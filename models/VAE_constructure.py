import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
from .refinenet import CRPBlock, MeanPoolConv, UpsampleConv, InstanceNorm2dPlus
from .layers import Act_bn_conv_block, VAE_Scaler


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
        
        
        self.cnn = nn.Sequential(nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1),
                                 Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(ngf, 2 * ngf),
                                 Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(2 * ngf,  4 * ngf),
                                 Act_bn_conv_block(4 * ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(ngf * 4, ngf * 2),
                                 Act_bn_conv_block(2 * ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(ngf * 2, ngf),
                                 Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
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
        
        
        self.cnn = nn.Sequential(nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1),
                                 Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(ngf, 2 * ngf),
                                 Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(2 * ngf,  4 * ngf),
                                 Act_bn_conv_block(4 * ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(ngf * 4, ngf * 2),
                                 Act_bn_conv_block(2 * ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(ngf * 2, ngf),
                                 Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
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
                                     Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
                                     UpsampleConv(ngf, ngf * 2),
                                     Act_bn_conv_block(ngf * 2, act = act, num_act_bn_conv = 1),
                                     UpsampleConv(ngf * 2, ngf * 4),
                                     Act_bn_conv_block(ngf * 4, act = act, num_act_bn_conv = 1),
                                     UpsampleConv(ngf * 4, ngf * 2),
                                     Act_bn_conv_block(ngf * 2, act = act, num_act_bn_conv = 1),
                                     UpsampleConv(ngf * 2, ngf),
                                     Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
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
        
        
        self.cnn = nn.Sequential(nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1),
                                 Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(ngf, 2 * ngf),
                                 Act_bn_conv_block(2 * ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(2 * ngf,  4 * ngf),
                                 Act_bn_conv_block(4 * ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(ngf * 4, ngf * 2),
                                 Act_bn_conv_block(2 * ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(ngf * 2, ngf),
                                 Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
                                 )
        
        self.mean = nn.Sequential(nn.BatchNorm1d(ngf *  4, affine=True),
                                  act,
                                  nn.Linear(ngf * 4, latent_dim),
                                  nn.BatchNorm1d(latent_dim, affine=True),
                                  act,
                                  )
        
        self.log_var = nn.Sequential(nn.BatchNorm1d(ngf * 4, affine=True),
                                 act,
                                 nn.Linear(ngf * 4, latent_dim),
                                 nn.BatchNorm1d(latent_dim, affine=True),
                                 act,
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
        
        
        self.cnn = nn.Sequential(nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1),
                                 Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(ngf, 2 * ngf),
                                 Act_bn_conv_block(2 * ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(2 * ngf,  4 * ngf),
                                 Act_bn_conv_block(4 * ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(ngf * 4, ngf * 2),
                                 Act_bn_conv_block(2 * ngf, act = act, num_act_bn_conv = 1),
                                 MeanPoolConv(ngf * 2, ngf),
                                 Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
                                 )
        
        self.mean = nn.Sequential(nn.BatchNorm1d(ngf *  4, affine=True),
                                  act,
                                  nn.Linear(ngf * 4, latent_dim),
                                  nn.BatchNorm1d(latent_dim, affine=True),
                                  act,
                                  nn.BatchNorm1d(latent_dim, affine=True),
                                  )
        
        self.log_var = nn.Sequential(nn.BatchNorm1d(ngf * 4, affine=True),
                                 act,
                                 nn.Linear(ngf * 4, latent_dim),
                                 nn.BatchNorm1d(latent_dim, affine=True),
                                 act,
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
        
        self.network = nn.Sequential(Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
                                     UpsampleConv(ngf, ngf * 2),
                                     Act_bn_conv_block(2 * ngf, act = act, num_act_bn_conv = 1),
                                     UpsampleConv(ngf * 2, ngf * 4),
                                     Act_bn_conv_block(4 * ngf, act = act, num_act_bn_conv = 1),
                                     UpsampleConv(ngf * 4, ngf * 2),
                                     Act_bn_conv_block(2 * ngf, act = act, num_act_bn_conv = 1),
                                     UpsampleConv(ngf * 2, ngf),
                                     Act_bn_conv_block(ngf, act = act, num_act_bn_conv = 1),
                                     nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)
                                     )
    
    
    def forward(self, z):
        tmp = self.decoder_projection(z)
        tmp = tmp.reshape(-1, self.ngf, 2, 2)
        output = self.network(tmp)
        return output
    
    
#======================================================================================

#======================================================================================

class VAE_model(nn.Module):
    def __init__(self, config, version):
        super().__init__()
        if version == 'BN_ver1':
            self.encoder = VAE_encoder_BN_ver1(config)
            self.decoder = VAE_decoder_ver1(config)
        elif version == 'ver1':
            self.encoder = VAE_encoder_ver1(config)
            self.decoder = VAE_decoder_ver1(config)
        elif version == 'BN_ver2':
            self.encoder = VAE_encoder_BN_ver2(config)
            self.decoder = VAE_decoder_ver2(config)
        elif version == 'ver2':
            self.encoder = VAE_encoder_ver2(config)
            self.decoder = VAE_decoder_ver2(config)
        else:
            raise NotImplementedError('VAE model version {} not understood.'.format(version))
        
            
    def forward(self, X, epsilon):
        (mean, log_var) = self.encoder(X)
        z = mean + torch.exp(0.5 * log_var) * epsilon
        output = self.decoder(z)
        return output, mean, log_var



            
            


