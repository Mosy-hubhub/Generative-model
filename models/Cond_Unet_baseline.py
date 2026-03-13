import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
from models.DiT import TimestepEmbedder, LabelEmbedder
from refinenet import ConvMeanPool, dilated_conv3x3, conv3x3


class ConditionalResidualBlock_AdaGN(nn.Module):
    '''
    Conditional Residual Block with AdaGN and Zero-Gating
    '''
    def __init__(self, input_dim, output_dim, embeding_size, resample=None,
                 act=nn.ELU(), dilation=None, adjust_padding=False, num_groups = 32):
        super().__init__()
        
        self.act = act
        self.resample = resample
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 【关键修复1】追踪第一层卷积后的真实通道数
        hidden_dim = input_dim if resample == 'down' else output_dim
        
        # --- Embedding 投影层 ---
        # 第一层：生成 shift1, scale1 (维度都是 input_dim)
        self.emb1 = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(embeding_size, 2 * input_dim)
        )
        
        # 第二层：生成 shift2, scale2 (维度是 hidden_dim) 以及最后的 gate (维度是 output_dim)
        self.emb2 = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(embeding_size, 2 * hidden_dim + output_dim)
        )
        
        # 【零初始化黑科技 (AdaLN-Zero 风格)】
        # 让网络在训练初期等效于 Identity 映射，极其稳定
        nn.init.zeros_(self.emb1[-1].weight)
        nn.init.zeros_(self.emb1[-1].bias)
        nn.init.zeros_(self.emb2[-1].weight)
        nn.init.zeros_(self.emb2[-1].bias)

        # --- 图像处理主干 ---
        self.normalize1 = nn.GroupNorm(num_groups, input_dim, affine = False)
        
        if resample == 'down':
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, hidden_dim, dilation=dilation)
                self.conv2 = ConvMeanPool(hidden_dim, output_dim, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)
            else:
                self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, stride=1, padding=1)
                self.conv2 = ConvMeanPool(hidden_dim, output_dim, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)

        elif resample is None:
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, hidden_dim, dilation=dilation)
                self.conv2 = dilated_conv3x3(hidden_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation)
            else:
                self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, stride=1, padding=1)
                conv_shortcut = partial(nn.Conv2d, kernel_size=3, padding=1)
        else:
            raise ValueError('invalid resample value')
                
        # 【关键修复2】第二层的 Norm 必须匹配 hidden_dim
        self.normalize2 = nn.GroupNorm(num_groups, hidden_dim, affine = False)

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, X, c):
        # 1. 计算第一层的条件参数 [B, 2 * input_dim]
        shift1, scale1 = self.emb1(c).chunk(2, dim=1)
        
        # 2. 计算第二层的条件参数 
        # 因为维度不同，用切片提取比 chunk 更安全
        emb2_out = self.emb2(c)
        shift2 = emb2_out[:, :self.conv1.out_channels]
        scale2 = emb2_out[:, self.conv1.out_channels : 2 * self.conv1.out_channels]
        gate   = emb2_out[:, 2 * self.conv1.out_channels :] 
        
        # 3. 【关键修复3】将一维条件向量广播为图像形状 [B, C, 1, 1]
        shift1 = shift1.unsqueeze(-1).unsqueeze(-1)
        scale1 = scale1.unsqueeze(-1).unsqueeze(-1)
        shift2 = shift2.unsqueeze(-1).unsqueeze(-1)
        scale2 = scale2.unsqueeze(-1).unsqueeze(-1)
        gate   = gate.unsqueeze(-1).unsqueeze(-1)
        
        # --- 前向传播：Layer 1 ---
        output = self.normalize1(X)
        output = output * (1 + scale1) + shift1
        output = self.act(output)
        output = self.conv1(output)
        
        # --- 前向传播：Layer 2 ---
        output = self.normalize2(output)
        output = output * (1 + scale2) + shift2
        output = self.act(output)
        output = self.conv2(output)
        
        # --- 门控与残差合并 ---
        # 最终的残差被 gate 控制，初期全为 0，网络逐渐学会打开大门
        output = output * gate

        return self.shortcut(X) + output


class CondRCUBlock_AdaGN(nn.Module):
    '''
    Conditional Residual Convolutional Unit Block with AdaGN and Zero-Gating
    
    Structure per block:
    Res = X
    For each stage:
        X = GroupNorm(X)
        X = AdaGN(X, scale, shift)
        X = ReLU(X)
        X = Conv3x3(X)
    X = Res + X * Gate
    '''
    def __init__(self, n_features, n_stages, n_blocks, embeding_size, act=nn.ReLU(), num_groups = 32):
        super().__init__()
        self.act = act
        self.n_stages = n_stages
        self.n_blocks = n_blocks
        self.n_features = n_features
        
        # 【关键修正 1：精准计算所需参数量】
        # 每个 stage 需要 2 组参数 (scale, shift)
        # 每个 block 需要 1 组门控参数 (gate)
        # 所以每个 block 总共需要: (2 * n_stages + 1) * n_features 个参数
        self.params_per_block = (2 * n_stages + 1) * n_features
        
        self.embed = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(embeding_size, self.n_blocks * self.params_per_block)
        )
        
        # 【关键修正 2：零初始化 (AdaLN-Zero)】
        nn.init.zeros_(self.embed[-1].weight)
        nn.init.zeros_(self.embed[-1].bias)
                
        # 【关键修正 3：抛弃 setattr，使用 ModuleList 嵌套】
        # 这能确保 PyTorch 完美追踪所有的权重梯度，并且 print(model) 时结构一目了然
        self.norms = nn.ModuleList([
            nn.ModuleList([nn.GroupNorm(num_groups, n_features, affine = False) for _ in range(n_stages)]) 
            for _ in range(n_blocks)
        ])
        
        self.convs = nn.ModuleList([
            nn.ModuleList([conv3x3(n_features, n_features) for _ in range(n_stages)]) 
            for _ in range(n_blocks)
        ])
        
    def forward(self, X, c):
        # 1. 一次性生成所有 block 和 stage 需要的条件参数
        # cond_params shape: [B, n_blocks * params_per_block]
        cond_params = self.embed(c)
        
        for i in range(self.n_blocks):
            Res = X
            
            # 2. 截取当前第 i 个 block 所专属的参数段
            start_idx = i * self.params_per_block
            end_idx = (i + 1) * self.params_per_block
            block_params = cond_params[:, start_idx:end_idx]
            
            for j in range(self.n_stages):
                # 3. 在当前 block 参数中，继续截取第 j 个 stage 的 scale 和 shift
                # 每个占 n_features 个通道
                scale_idx = j * 2 * self.n_features
                shift_idx = (j * 2 + 1) * self.n_features
                next_idx  = (j * 2 + 2) * self.n_features
                
                scale = block_params[:, scale_idx : shift_idx]
                shift = block_params[:, shift_idx : next_idx]
                
                # 广播到空间维度 [B, C, 1, 1]
                scale = scale.unsqueeze(-1).unsqueeze(-1)
                shift = shift.unsqueeze(-1).unsqueeze(-1)
                
                # 4. 执行核心的 AdaGN 运算
                X = self.norms[i][j](X)
                X = X * (1 + scale) + shift
                X = self.act(X)
                X = self.convs[i][j](X)
            
            # 5. 提取当前 block 的 gate 参数 (在参数段的最后)
            gate_idx = self.n_stages * 2 * self.n_features
            gate = block_params[:, gate_idx :]
            gate = gate.unsqueeze(-1).unsqueeze(-1)
            
            # 6. 带门控的残差连接
            X = Res + X * gate
            
        return X


class CondCRPBlock_AdaGN(nn.Module):
    '''
    Conditional Chained Residual Pooling Block with AdaGN and Zero-Gating
    
    Structure:
    X = act(X)
    path = X
    For each stage:
        path = GroupNorm(path)
        path = AdaGN(path, scale, shift)
        path = AvgPool(path)
        path = Conv3x3(path)
        X = X + path * gate
    '''
    def __init__(self, n_features, n_stages, embeding_size, act=nn.ReLU(), num_groups = 32):
        super().__init__()
        self.act = act
        self.n_stages = n_stages
        self.n_features = n_features
        
        # 【关键设计 1：参数分配】
        # 每一个 stage 需要 3 组参数：scale, shift, gate
        self.params_per_stage = 3 * n_features
        
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        
        self.embed = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(embeding_size, n_stages * self.params_per_stage)
        )
        
        # 【关键设计 2：零初始化】
        # 确保初始状态下 gate 为 0，等效于 Identity 映射
        nn.init.zeros_(self.embed[-1].weight)
        nn.init.zeros_(self.embed[-1].bias)
        
        # 使用 ModuleList 存储每一级的 norm 和 conv
        self.norms = nn.ModuleList([nn.GroupNorm(num_groups, n_features, affine = False) for _ in range(n_stages)])
        self.convs = nn.ModuleList([conv3x3(n_features, n_features) for _ in range(n_stages)])
        
    def forward(self, X, c):
        # CRP 的特点：先对全局做一次激活
        X = self.act(X)
        path = X
        
        # 一次性算出所有 stage 所需的条件参数
        # cond_params shape: [B, n_stages * 3 * n_features]
        cond_params = self.embed(c)
        
        for i in range(self.n_stages):
            # 1. 动态切片：提取当前 stage 的 scale, shift, gate
            start_idx = i * self.params_per_stage
            
            scale = cond_params[:, start_idx : start_idx + self.n_features]
            shift = cond_params[:, start_idx + self.n_features : start_idx + 2 * self.n_features]
            gate  = cond_params[:, start_idx + 2 * self.n_features : start_idx + 3 * self.n_features]
            
            # 2. 广播到空间维度 [B, C, 1, 1]
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            gate  = gate.unsqueeze(-1).unsqueeze(-1)
            
            # 3. 核心流转：Norm -> AdaGN 调制 -> Pool -> Conv
            path = self.norms[i](path)
            path = path * (1 + scale) + shift
            
            path = self.pool(path)
            path = self.convs[i](path)
            
            # 4. 带门控的链式残差叠加
            X = X + path * gate
            
        return X



class CondMSFBlock_AdaGN(nn.Module):
    '''
    Conditional Multi-Scale Fusion Block with AdaGN
    
    Structure per scale branch:
    tmp = GroupNorm(Xs[i])
    tmp = AdaGN(tmp, scale, shift)
    tmp = conv3x3(tmp)
    tmp = F.interpolate(tmp, shape)
    s = s + tmp
    '''
    def __init__(self, in_planes, n_features, embeding_size, num_groups = 32):
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        
        self.n_features = n_features
        self.in_planes = in_planes
        
        # 【关键修复 1：动态计算总参数量】
        # 由于每个输入分支的通道数 in_planes[i] 不同
        # 第 i 个分支需要 2 * in_planes[i] 个参数 (scale 和 shift)
        self.total_cond_params = sum(2 * c for c in in_planes)
        
        self.embed = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(embeding_size, self.total_cond_params)
        )
        
        # 【关键修复 2：AdaGN的零初始化】
        # 权重清零，使得 scale=0, shift=0，初始状态下 X * (1+0) + 0 = X
        # 完美保持恒等映射，且不会切断梯度流
        nn.init.zeros_(self.embed[-1].weight)
        nn.init.zeros_(self.embed[-1].bias)
        
        self.norm = nn.ModuleList([nn.GroupNorm(num_groups, c, affine = False) for c in in_planes])
        self.conv = nn.ModuleList([conv3x3(c, n_features) for c in in_planes])
            
    def forward(self, Xs, c, shape):
        # cond_params 包含了所有分支所需的 scale 和 shift
        cond_params = self.embed(c)
        
        s = 0  # 用于累加特征图
        idx = 0  # 动态切片游标
        
        for i in range(len(self.in_planes)):
            in_c = self.in_planes[i]
            
            # 1. 精准提取当前分支的 scale 和 shift
            scale = cond_params[:, idx : idx + in_c].unsqueeze(-1).unsqueeze(-1)
            idx += in_c
            
            shift = cond_params[:, idx : idx + in_c].unsqueeze(-1).unsqueeze(-1)
            idx += in_c
            
            # 2. 特征流转：Norm -> AdaGN 注入
            tmp = self.norm[i](Xs[i])
            tmp = tmp * (1 + scale) + shift
            
            # 3. 统一通道数
            tmp = self.conv[i](tmp)
            
            # 4. 统一分辨率 (建议 align_corners=False，这是扩散模型的通用标准)
            # 注意：传入 shape (H, W) 即可
            tmp = F.interpolate(tmp, size=shape, mode='bilinear', align_corners=False)
            
            # 5. 累加融合
            if i == 0:
                s = tmp
            else:
                s = s + tmp
    
        return s
    
    

class CondRefineBlock_AdaGN(nn.Module):
    '''
    Constructure:
    [CondRCUBlock_1, CondRCUBlock_2, ... , CondRCUBlock_n_block] - CondMSFBlock(if n_block > 1)
    - CondCRPBlock - output_convs
    '''
    def __init__(self, in_planes, n_features, embeding_size, act = nn.ReLU(), num_groups = 32, end = False):
        super().__init__()
        
        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = len(in_planes)
        
        self.n_features = n_features
        
        self.Cond_RCU = nn.ModuleList()
        for i in range(self.n_blocks):
            self.Cond_RCU.append(CondRCUBlock_AdaGN(in_planes[i], 2, 2, embeding_size, act, num_groups))
            
        if self.n_blocks > 1:
            self.Cond_MSF = CondMSFBlock_AdaGN(in_planes, n_features, embeding_size, num_groups)
            
        self.Cond_CRP = CondCRPBlock_AdaGN(n_features, 2, embeding_size, act, num_groups)
        self.output_convs = CondRCUBlock_AdaGN(n_features, 3 if end else 1, 2, embeding_size, act, num_groups)
        
    
    def forward(self, Xs, c, shape):
        assert isinstance(Xs, tuple) or isinstance(Xs, list)
        tmp = []
        
        for i in range(self.n_blocks):
            tmp.append(self.Cond_RCU[i](Xs[i], c))
            
        if self.n_blocks > 1:
            output = self.Cond_MSF(tmp, c, shape)
        else:
            output = tmp[0]
            
        output = self.Cond_CRP(output, c)
        output = self.output_convs(output, c)
        return output

    
class AdaGN_UNet_baseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.data.num_classes
        self.embeding_size = config.model.embeding_size
        self.class_dropout_prob = config.model.class_dropout_prob
        self.num_groups = config.model.num_groups
        self.act = act = nn.ELU()
        
        self.label_embedder = LabelEmbedder(self.num_classes, self.embeding_size, self.class_dropout_prob)
        self.t_embedder = TimestepEmbedder(self.embeding_size)
        
        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)
        
        self.normalizer = nn.GroupNorm(self.num_groups, ngf, affine=False)
        self.end_embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embeding_size, ngf * 2) 
        )
        # 保持零初始化
        nn.init.zeros_(self.end_embed[-1].weight)
        nn.init.zeros_(self.end_embed[-1].bias)
        
        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ConditionalResidualBlock_AdaGN(self.ngf, self.ngf, self.embeding_size, resample=None, act=act, num_groups=self.num_groups),
            ConditionalResidualBlock_AdaGN(self.ngf, self.ngf, self.embeding_size, resample=None, act=act, num_groups=self.num_groups),
        ])
            
        self.res2 = nn.ModuleList([
            ConditionalResidualBlock_AdaGN(self.ngf, 2 * self.ngf, self.embeding_size, resample='down', act=act, num_groups=self.num_groups),
            ConditionalResidualBlock_AdaGN(2 * self.ngf, 2 * self.ngf, self.embeding_size, resample=None, act=act, num_groups=self.num_groups)
        ])

        self.res3 = nn.ModuleList([
            ConditionalResidualBlock_AdaGN(2 * self.ngf, 2 * self.ngf, self.embeding_size, resample='down', act=act, dilation=2, num_groups=self.num_groups),
            ConditionalResidualBlock_AdaGN(2 * self.ngf, 2 * self.ngf, self.embeding_size, resample=None, act=act, dilation=2, num_groups=self.num_groups)
        ])

        if config.data.image_size == 28:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock_AdaGN(2 * self.ngf, 2 * self.ngf, self.embeding_size, resample='down', act=act, adjust_padding=True, dilation=4, num_groups=self.num_groups),
                ConditionalResidualBlock_AdaGN(2 * self.ngf, 2 * self.ngf, self.embeding_size, resample=None, act=act, dilation=4, num_groups=self.num_groups)
            ])
        else:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock_AdaGN(2 * self.ngf, 2 * self.ngf, self.embeding_size, resample='down', act=act, adjust_padding=False, dilation=4, num_groups=self.num_groups),
                ConditionalResidualBlock_AdaGN(2 * self.ngf, 2 * self.ngf, self.embeding_size, resample=None, act=act, dilation=4, num_groups=self.num_groups)
            ])


        self.refine1 = CondRefineBlock_AdaGN([2 * self.ngf], 2 * self.ngf, self.embeding_size, act=act, num_groups=self.num_groups)
        self.refine2 = CondRefineBlock_AdaGN([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, self.embeding_size, act=act, num_groups=self.num_groups)
        self.refine3 = CondRefineBlock_AdaGN([2 * self.ngf, 2 * self.ngf], self.ngf, self.embeding_size, act=act, num_groups=self.num_groups)
        self.refine4 = CondRefineBlock_AdaGN([self.ngf, self.ngf], self.ngf, self.embeding_size, act=act, num_groups=self.num_groups, end = True)

    def _compute_cond_module(self, module, x, c):
        for m in module:
            x = m(x, c)
        return x

    def forward(self, x, t, y):
        #if not self.logit_transform:
        #    x = 2 * x - 1.
            
        # 计算全局融合指令 c
        c = self.t_embedder(t) + self.label_embedder(y, self.training)

        output = self.begin_conv(x)

        # Encoder
        layer1 = self._compute_cond_module(self.res1, output, c)
        layer2 = self._compute_cond_module(self.res2, layer1, c)
        layer3 = self._compute_cond_module(self.res3, layer2, c)
        layer4 = self._compute_cond_module(self.res4, layer3, c)

        # Decoder (RefineNet)
        ref1 = self.refine1([layer4], c, layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], c, layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], c, layer2.shape[2:])
        output = self.refine4([layer1, ref3], c, layer1.shape[2:])


        output = self.normalizer(output)
        
        # 从 c 中抽出最后的 scale 和 shift
        scale, shift = self.end_embed(c).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        
        # 实施调制
        output = output * (1 + scale) + shift
        
        # 激活并输出到 3 通道 (或预测的噪音通道)
        output = self.act(output)
        output = self.end_conv(output)
        
        return output
    
    def forward_with_cfg(self, x, t, y, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        eps = self.forward(combined, t, y)
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        # Completely wrong: eps = torch.cat([half_eps, uncond_eps], dim=0) !
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps