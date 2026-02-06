import torch.nn as nn
import torch


class Act_bn_conv(nn.Module):
    def __init__(self, in_planes, num_features, affine = True, act = nn.ReLU(), padding = 1):
        super().__init__()
        
        self.act = act
        self.bn = nn.BatchNorm2d(num_features, affine= affine)
        self.conv = nn.Conv2d(in_planes, num_features, 3, padding = padding)

    def forward(self, X):
        return self.act(self.bn(self.conv(X))) 
    
    
class Act_bn_conv_block(nn.Module):
    def __init__(self, in_planes,
                 affine = True,
                 act = nn.ReLU(),
                 padding = 1,
                 num_act_bn_conv = 2):
        super().__init__()
        self.num_act_bn_conv = num_act_bn_conv
        self.network = nn.ModuleList()
        
        for _ in range(num_act_bn_conv):
            self.network.append(Act_bn_conv(in_planes, in_planes,
                                            affine = affine,
                                            act = act, 
                                            padding = padding))

    def forward(self, X):
        for i in range(self.num_act_bn_conv):
            X = self.network[i](X) + X
        return X
    
    

class VAE_Scaler(nn.Module):
    """用于 VAE 防止 KL 消失的特殊缩放层
    Args:
        latent_dim: 潜在空间维度
        tau: 超参数，默认 0.5, 确保 gamma_mu > sqrt(tau)
        init_scale: 初始化 scale 参数的值
    """
    def __init__(self, latent_dim, tau=0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.tau = tau
        
        # 可训练参数 theta
        self.scale = nn.Parameter(torch.zeros(latent_dim))
        
    def forward(self, x, mode='positive'):
        """
        Args:
            x: 输入张量 [batch_size, latent_dim]
            mode: 'positive' 用于 mu, 'negative' 用于 log_var
        """
        if mode == 'positive':
            # gamma_mu = sqrt(tau + (1-tau) * sigmoid(theta))
            scale = self.tau + (1 - self.tau) * torch.sigmoid(self.scale)
        elif mode == 'negative':
            # gamma_sigma = sqrt((1-tau) * sigmoid(-theta))
            scale = (1 - self.tau) * torch.sigmoid(-self.scale)
        else:
            raise ValueError("mode must be 'positive' or 'negative'")
        
        # 确保数值稳定性
        scale = torch.clamp(scale, min=1e-8)
        
        # 应用缩放: x * sqrt(scale)
        scale_sqrt = torch.sqrt(scale)
        # 扩展维度以匹配 batch_size
        # unsqueeze(0):在第0维增加一个维度，这样才能用expand
        # expand(x.size(0), -1), 第零个维度广播成x.size(0), 第二个维度不变
        scale_sqrt = scale_sqrt.unsqueeze(0).expand(x.size(0), -1)
        
        return x * scale_sqrt
    
    def get_gamma_values(self):
        """获取当前的 gamma_mu 和 gamma_sigma 值"""
        with torch.no_grad():
            gamma_mu = torch.sqrt(self.tau + (1 - self.tau) * torch.sigmoid(self.scale))
            gamma_sigma = torch.sqrt((1 - self.tau) * torch.sigmoid(-self.scale))
            return gamma_mu.mean().item(), gamma_sigma.mean().item()