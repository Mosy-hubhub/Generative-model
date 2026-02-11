import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_dim, frequency = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency, hidden_dim, bias=True),
                                 nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim, bias=True)
                                 )

        self.frequency = frequency
        

    @staticmethod
    def timestep_embedding(t, dim, max_period = 10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        frequency_list = torch.exp( - torch.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half).to(t.device)
        
        args = t[:,None].float() * frequency_list[None]
        cossin_wave = torch.cat([torch.sin(args), torch.cos(args)], dim = -1)
        if dim % 2 == 1:
            cossin_wave = torch.cat([cossin_wave, torch.zeros((len(t), 1), device = cossin_wave.device)], dim = 1)
            
        return cossin_wave
        
        

    def forward(self, t):
        cossin_wave = self.timestep_embedding(t, self.frequency)
        return self.mlp(cossin_wave)


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_label, hidden_dim, drop_prob):
        super().__init__()
        self.is_drop_prob = drop_prob > 0 
        self.drop_prob = drop_prob
        self.num_label = num_label
        self.hidden_dim = hidden_dim
        # total_num_label contains empty label
        total_num_label = num_label + self.is_drop_prob
        self.embedding = nn.Embedding(total_num_label, hidden_dim)
        

    def token_drop(self, labels, force_drop_ids = None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.drop_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_label, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        if (train and self.is_drop_prob) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding(labels)

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_dim, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine = False, eps = 1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine = False, eps = 1e-6)
        self.atten = Attention(hidden_dim, num_heads, qkv_bias = True, **block_kwargs)
        hidden_features = hidden_dim * mlp_ratio
        self.mlp = Mlp(hidden_dim, hidden_features, act_layer = approx_gelu)
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )
        
    def forward(self, x, c):
        '''
        c is mixture of time t and label
        x: [batch_size, seq_len, hidden_dim]
        gate_msa.unsqueeze(1): [batch_size, 1, hidden_dim]
        '''
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.atten(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_dim, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        
        # ouput.shape = (batch_size, num_patch, hidden_dim)
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_dim, bias=True)
        # output.shape = (batch_size, hidden_dim)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        # output.shape = (batch_size, hidden_dim)
        self.label_embedder = LabelEmbedder(num_classes, hidden_dim, class_dropout_prob)
        
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim), requires_grad=False)
        
        self.DiT_blocks = nn.ModuleList(
            DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.final_layer = FinalLayer(hidden_dim, patch_size, self.out_channels)
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize transformer layers:
                # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # output.shape = [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
        # = [grid_size*grid_size, hidden_dim] or [1+grid_size*grid_size, hidden_dim]
        pos_embed = get_2d_sincos_pos_embed(embed_dim = self.pos_embed.shape[-1], 
                                            grid_size = int(self.x_embedder.num_patches ** 0.5)
                                            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding:
        nn.init.normal_(self.label_embedder.embedding.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.DiT_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.label_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        # origin code here: eps = torch.cat([half_eps, half_eps], dim=0)
        eps = torch.cat([half_eps, uncond_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    output: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_w = np.arange(grid_size, dtype = np.float64)
    grid_h = np.arange(grid_size, dtype = np.float64)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # grid.shape = (2, grid_size, grid_size)
    grid = np.stack(grid, axis=0)
    
    # ouput.shape = (grid_size ** 2, D)
    output = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token and extra_tokens > 0:
        output = np.concatenate([np.zeros([extra_tokens, embed_dim]), output], axis=0)
        
    return output


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    '''
    out: (M, D), D is embed_dim, M is the half of the num of entries of grid
    '''
    assert embed_dim % 2 == 0
    
    w_embed = get_1d_sincos_pos_embed_from_grid(embed_dim / 2, grid[0])
    h_embed = get_1d_sincos_pos_embed_from_grid(embed_dim / 2, grid[1])
    
    output = np.concatenate([w_embed, h_embed], axis = 1)
    
    return output


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, max_frequency = 10000):
    """
    embed_dim: output dimension for each position (used as hidden_dim), D
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    
    # frequency matrix
    w = np.exp(- (np.arange(embed_dim / 2, dtype = np.float64) / (embed_dim / 2))
               * np.log(max_frequency))
    pos = pos.reshape(-1)
    
    w = np.einsum('m,d->md', pos , w)
    cos_array = np.cos(w)
    sin_array = np.sin(w)
    
    output = np.concatenate([cos_array, sin_array], axis = 1)
    
    return output

#################################################################################
#                                   DiT Configs                                  #
#################################################################################


