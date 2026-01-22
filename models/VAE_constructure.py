import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
from .refinenet import CRPBlock, MeanPoolConv, UpsampleConv, InstanceNorm2dPlus


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
        scale_sqrt = scale_sqrt.unsqueeze(0).expand(x.size(0), -1)
        
        return x * scale_sqrt
    
    def get_gamma_values(self):
        """获取当前的 gamma_mu 和 gamma_sigma 值"""
        with torch.no_grad():
            gamma_mu = torch.sqrt(self.tau + (1 - self.tau) * torch.sigmoid(self.scale))
            gamma_sigma = torch.sqrt((1 - self.tau) * torch.sigmoid(-self.scale))
            return gamma_mu.mean().item(), gamma_sigma.mean().item()





class VAE_encoder_ver1(nn.Module):
    '''
    input dimension: sample_dimension
    output dimension: latent_dimension * 2
    '''
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.ngf = ngf = config.model.ngf
        self.act = act = nn.ELU()
        self.latent_dim = latent_dim = config.model.latent_dimension
        self.norm = InstanceNorm2dPlus
        # self.act = act = nn.ReLU(True)
        
        
        self.cnn = nn.Sequential(nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1),
                                 CRPBlock(ngf, 2, act),
                                 MeanPoolConv(ngf, 2 * ngf),
                                 CRPBlock(2 * ngf, 2, act),
                                 MeanPoolConv(2 * ngf,  4 * ngf),
                                 CRPBlock(4 * ngf, 2, act),
                                 MeanPoolConv(ngf * 4, ngf * 2),
                                 CRPBlock(ngf * 2, 2, act),
                                 MeanPoolConv(ngf * 2, ngf),
                                 CRPBlock(ngf, 2, act),
                                 MeanPoolConv(ngf, ngf // 2),
                                 )
        
        self.mean = nn.Sequential(nn.BatchNorm1d(ngf // 2, affine=True),
                                  act,
                                  nn.Linear(ngf // 2, latent_dim),
                                  nn.BatchNorm1d(latent_dim, affine=True),
                                  act,
                                  )
        
        self.log_var = nn.Sequential(nn.BatchNorm1d(ngf // 2, affine=True),
                                 act,
                                 nn.Linear(ngf // 2, latent_dim),
                                 nn.BatchNorm1d(latent_dim, affine=True),
                                 act,
                                 )
    
    
    def forward(self, X):
        temp = self.cnn(X)
        temp = temp.squeeze(dim=[2, 3])
        mean = self.mean(temp)
        var = self.var(temp)
        return (mean, var)
    
    
class VAE_encoder_BN_ver1(nn.Module):
    '''
    input dimension: sample_dimension
    output dimension: latent_dimension * 2
    plus batchnorm at the end of the output head of mean
    '''
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.ngf = ngf = config.model.ngf
        self.act = act = nn.ELU()
        self.latent_dim = latent_dim = config.model.latent_dimension
        self.norm = InstanceNorm2dPlus
        # self.act = act = nn.ReLU(True)
        
        
        self.cnn = nn.Sequential(nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1),
                                 CRPBlock(ngf, 2, act),
                                 MeanPoolConv(ngf, 2 * ngf),
                                 CRPBlock(2 * ngf, 2, act),
                                 MeanPoolConv(2 * ngf,  4 * ngf),
                                 CRPBlock(4 * ngf, 2, act),
                                 MeanPoolConv(ngf * 4, ngf * 2),
                                 CRPBlock(ngf * 2, 2, act),
                                 MeanPoolConv(ngf * 2, ngf),
                                 CRPBlock(ngf, 2, act),
                                 MeanPoolConv(ngf, ngf // 2),
                                 )
        
        self.mean = nn.Sequential(nn.BatchNorm1d(ngf // 2, affine=True),
                                  act,
                                  nn.Linear(ngf // 2, latent_dim),
                                  nn.BatchNorm1d(latent_dim, affine=True),
                                  act,
                                  nn.BatchNorm1d(latent_dim, affine=True),
                                  )
        
        self.log_var = nn.Sequential(nn.BatchNorm1d(ngf // 2, affine=True),
                                 act,
                                 nn.Linear(ngf // 2, latent_dim),
                                 nn.BatchNorm1d(latent_dim, affine=True),
                                 act,
                                 )
    
    
    def forward(self, X):
        temp = self.cnn(X)
        temp = temp.squeeze(dim=[2, 3])
        mean = self.mean(temp)
        var = self.var(temp)
        return (mean, var)
    
    
    
    
    
class VAE_decoder_ver1(nn.Module):
    '''
    input dimension: latent_dimension * 2
    output dimension: sample_dimension
    because the variance of the component of ouput defaults to 1
    '''
    def __init__(self, config):
        super().__init__()
        self.latent_dim = latent_dim = config.model.latent_dimension
        self.logit_transform = config.data.logit_transform
        self.ngf = ngf = config.model.ngf
        self.act = act = nn.ELU()
        self.norm = InstanceNorm2dPlus
        # self.act = act = nn.ReLU(True)
        
        self.decoder_projection = nn.Sequential(nn.Linear(latent_dim, ngf // 2),
                                                act,
                                                nn.BatchNorm1d(ngf // 2, affine=True),
                                                )
        
        self.network = nn.Sequential(UpsampleConv(ngf // 2, ngf),
                                     CRPBlock(ngf, 2, act),
                                     UpsampleConv(ngf, ngf * 2),
                                     CRPBlock(ngf * 2, 2, act),
                                     UpsampleConv(ngf * 2, ngf * 4),
                                     CRPBlock(ngf * 4, 2, act),
                                     UpsampleConv(ngf * 4, ngf * 2),
                                     CRPBlock(ngf * 2, 2, act),
                                     UpsampleConv(ngf * 2, ngf),
                                     CRPBlock(ngf, 2, act),
                                     nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1),
                                     )
    
    
    def forward(self, z):
        temp = self.decoder_projection(z)
        temp = temp.reshape(-1, self.ngf // 2, 1, 1)
        output = self.network(temp)
        return output
    
    
    
#=========================================================================================
    
    
#=========================================================================================
    
    
    
class VAE_encoder_ver2(nn.Module):
    '''
    input dimension: sample_dimension
    output dimension: latent_dimension * 2
    '''
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.ngf = ngf = config.model.ngf
        self.act = act = nn.ELU()
        self.latent_dim = latent_dim = config.model.latent_dimension
        self.norm = InstanceNorm2dPlus
        # self.act = act = nn.ReLU(True)
        
        
        self.cnn = nn.Sequential(nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1),
                                 CRPBlock(ngf, 2, act),
                                 MeanPoolConv(ngf, 2 * ngf),
                                 CRPBlock(2 * ngf, 2, act),
                                 MeanPoolConv(2 * ngf,  4 * ngf),
                                 CRPBlock(4 * ngf, 2, act),
                                 MeanPoolConv(ngf * 4, ngf * 2),
                                 CRPBlock(ngf * 2, 2, act),
                                 MeanPoolConv(ngf * 2, ngf),
                                 CRPBlock(ngf, 2, act),
                                 )
        
        self.mean = nn.Sequential(nn.BatchNorm1d(ngf *  4, affine=True),
                                  nn.ELU(),
                                  nn.Linear(ngf * 4, latent_dim),
                                  nn.BatchNorm1d(latent_dim, affine=True),
                                  nn.ELU(),
                                  )
        
        self.log_var = nn.Sequential(nn.BatchNorm1d(ngf * 4, affine=True),
                                 nn.ReLU(),
                                 nn.Linear(ngf * 4, latent_dim),
                                 nn.BatchNorm1d(latent_dim, affine=True),
                                 nn.ReLU(),
                                 )
    
    
    def forward(self, X):
        temp = self.cnn(X)
        temp = temp.squeeze(dim=[2, 3])
        mean = self.mean(temp)
        var = self.var(temp)
        return (mean, var)
    
    
    
    
    
class VAE_encoder_BN_ver2(nn.Module):
    '''
    input dimension: sample_dimension
    output dimension: latent_dimension * 2
    '''
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.ngf = ngf = config.model.ngf
        self.act = act = nn.ELU()
        self.latent_dim = latent_dim = config.model.latent_dimension
        self.norm = InstanceNorm2dPlus
        # self.act = act = nn.ReLU(True)
        
        
        self.cnn = nn.Sequential(nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1),
                                 CRPBlock(ngf, 2, act),
                                 MeanPoolConv(ngf, 2 * ngf),
                                 CRPBlock(2 * ngf, 2, act),
                                 MeanPoolConv(2 * ngf,  4 * ngf),
                                 CRPBlock(4 * ngf, 2, act),
                                 MeanPoolConv(ngf * 4, ngf * 2),
                                 CRPBlock(ngf * 2, 2, act),
                                 MeanPoolConv(ngf * 2, ngf),
                                 CRPBlock(ngf, 2, act),
                                 )
        
        self.mean = nn.Sequential(nn.BatchNorm1d(ngf *  4, affine=True),
                                  nn.ELU(),
                                  nn.Linear(ngf * 4, latent_dim),
                                  nn.BatchNorm1d(latent_dim, affine=True),
                                  nn.ELU(),
                                  nn.BatchNorm1d(latent_dim, affine=True),
                                  )
        
        self.log_var = nn.Sequential(nn.BatchNorm1d(ngf * 4, affine=True),
                                 nn.ReLU(),
                                 nn.Linear(ngf * 4, latent_dim),
                                 nn.BatchNorm1d(latent_dim, affine=True),
                                 nn.ReLU(),
                                 )
    
    
    def forward(self, X):
        temp = self.cnn(X)
        temp = temp.squeeze(dim=[2, 3])
        mean = self.mean(temp)
        var = self.var(temp)
        return (mean, var)
    
    
    

class VAE_decoder_ver2(nn.Module):
    '''
    input dimension: latent_dimension * 2
    output dimension: sample_dimension
    because the variance of the component of ouput defaults to 1
    '''
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.ngf = ngf = config.model.ngf
        self.act = act = nn.ELU()
        self.norm = InstanceNorm2dPlus
        self.latent_dim = latent_dim = config.model.latent_dimension
        # self.act = act = nn.ReLU(True)
        
        self.decoder_projection = nn.Sequential(nn.Linear(latent_dim, ngf * 4),
                                                act,
                                                nn.BatchNorm1d(ngf * 4, affine = True),
                                                )  
        
        self.network = nn.Sequential(CRPBlock(ngf, 2, act),
                                     UpsampleConv(ngf, ngf * 2),
                                     CRPBlock(ngf * 2, 2, act),
                                     UpsampleConv(ngf * 2, ngf * 4),
                                     CRPBlock(ngf * 4, 2, act),
                                     UpsampleConv(ngf * 4, ngf * 2),
                                     CRPBlock(ngf * 2, 2, act),
                                     UpsampleConv(ngf * 2, ngf),
                                     CRPBlock(ngf, 2, act),
                                     nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)
                                     )
    
    
    def forward(self, z):
        tmp = self.decoder_projection(z)
        tmp = tmp.reshape(-1, self.ngf, 2, 2)
        output = self.network(tmp)
        return output
    
    
#======================================================================================


#======================================================================================


    
class VAE_model_BN_ver1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = VAE_encoder_BN_ver1(config)
        self.decoder = VAE_decoder_ver1(config)
            
            
    def forward(self, X, epsilon):
        (mean, log_var) = self.encoder(X)
        z = mean + torch.exp(0.5 * log_var) * epsilon
        output = self.decoder(z)
        return output, mean, log_var
        
        
class VAE_model_ver1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = VAE_encoder_ver1(config)
        self.decoder = VAE_decoder_ver1(config)
            
            
    def forward(self, X, epsilon):
        (mean, log_var) = self.encoder(X)
        z = mean + torch.exp(0.5 * log_var) * epsilon
        output = self.decoder(z)
        return output, mean, log_var
    
    
class VAE_model_BN_ver2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = VAE_encoder_BN_ver2(config)
        self.decoder = VAE_decoder_ver2(config)
            
            
    def forward(self, X, epsilon):
        (mean, log_var) = self.encoder(X)
        z = mean + torch.exp(0.5 * log_var) * epsilon
        output = self.decoder(z)
        return output, mean, log_var
            
            
class VAE_model_ver2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = VAE_encoder_ver2(config)
        self.decoder = VAE_decoder_ver2(config)
            
            
    def forward(self, X, epsilon):
        (mean, log_var) = self.encoder(X)
        z = mean + torch.exp(0.5 * log_var) * epsilon
        output = self.decoder(z)
        return output, mean, log_var
            
            


