import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    '''
    in_planes: the number of input channels
    out_planes: the number of output channels
    this layer keep image resolution (size)
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)
    
def conv1x1(in_planes, out_planes, stride=1, bias=False):
    '''
    this layer keep image resolution (size)
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


def dilated_conv3x3(in_planes, out_planes, dilation, bias=True):
    '''
    dilation: the kernel size(receptive field) actually is (3 + 2 * (dilation - 1))
    this layer keep image resolution (size) (2 + 2 * (dilation - 1) = 2 * dilation )
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, dilation=dilation, bias=bias)


class ConditionalBatchNorm2d(nn.Module):
    '''
    this layer is batch normalization conditionally with different noise scale
    this is one of diffence of different scorenet with different noise scale(in diffusion)
    
    remark:num_features means the number of feature graph here,when we do batchnorm,we don't fuse
    different position, because it's dimension is too high and we lose their position information.
    '''
    def __init__(self, num_features, num_classes, bias = True):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.bias = bias
        # affine = False because we already have embed layer
        self.bn = nn.BatchNorm2d(num_features, affine = False)
        if self.bias:
            self.embed = nn.Embedding(num_classes, 2 * num_features)
            self.embed.weight.data[:,:num_features].uniform_()
            self.embed.weight.data[:,num_features:].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features)
            self.embed.weight.data.uniform_()
        
        
        
    def forward(self, X, y):
        X = self.bn(X)
        
        if self.bias:
            beta, gamma = self.embed(y).chunk(2, dim = 1)
            X = beta.view(-1, self.num_features, 1, 1) * X + gamma.view(-1, self.num_features, 1, 1)
            
        else:
            beta = self.embed(y)
            X = beta.view(-1, self.num_features, 1, 1) * X
        return X
    
    
class CRPBlock(nn.Module):
    '''
    Chained Residual Pooling Block 
    n_features: num of feature graph
    Constructure:
    (maxpool-conv3x3-) * n_stages
    act(activate function) = ReLU
    this block keeps planes staying the same
    '''
    def __init__(self, n_features, n_stages, act = nn.ReLU()):
        super().__init__()
        self.act = act
        self.n_stages = n_stages
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv = nn.ModuleList()
        for i in range(n_stages):
            self.conv.append(conv3x3(n_features, n_features))
        
    def forward(self, X):
        X = self.act(X)
        path = X
        for i in range(self.n_stages):
            path = self.pool(path)
            path = self.conv[i](path)
            X = path + X
        return X
    
    
class CondCRPBlock(nn.Module):
    '''
    Conditional Chained Residual Pooling Block
    the difference is that it uses conditional normalization
    Constructure:
    (cond_norm-Avgpool-con3x3-) * n_stages
    act(activate function) = ReLU
    '''
    def __init__(self, n_features, n_stages, normalizer ,num_classes ,act = nn.ReLU()):
        super().__init__()
        self.act = act
        self.n_stages = n_stages
        self.pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.conv = nn.ModuleList()
        self.condnorm = nn.ModuleList()
        for i in range(n_stages):
            self.conv.append(conv3x3(n_features, n_features))
            self.condnorm.append(normalizer(n_features, num_classes))
        
    def forward(self, X, y):
        X = self.act(X)
        path = X
        for i in range(self.n_stages):
            path = self.condnorm[i](path, y)
            path = self.pool(path)
            path = self.conv[i](path)
            X = path + X
        return X
    
    
class CondRCUBlock(nn.Module):
    '''
    Residual Convolutional Unit Block
    Constructure:
    ((cond_norm- ReLU(activation)-conv3x3- ) * self.n_stages (with Res Connect)) * self.n_blocks
    
    Remark: the way of Res Connection in thid block is different from CRPBlock
    '''
    def __init__(self, n_features, n_stages, n_blocks, normalizer , n_classes, act = nn.ReLU()):
        super().__init__()
        self.act = act
        self.n_stages = n_stages
        self.n_blocks = n_blocks
                
        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(self, '{}_{}_norm'.format(i, j), normalizer(n_features, n_classes))
                setattr(self, '{}_{}_conv'.format(i, j), conv3x3(n_features, n_features))
                
        
    def forward(self, X, y):
        for i in range(self.n_blocks):
            Res = X
            for j in range(self.n_stages):
                X = getattr(self, '{}_{}_norm'.format(i, j))(X, y)
                X = self.act(X)
                X = getattr(self, '{}_{}_conv'.format(i, j))(X)
            X = Res + X  
        return X
    
    
class CondMSFBlock(nn.Module):
    '''
    Multi-Scale Fusion Block
    Constructure:
    (normalizar-conv3x3-F.interpolate +) * len(in_planes)
    
    features: num of feature graph after Fusion
    in_planes: a tuple or a list of num of input feature graph(planes) of each scale
    Xs:a list of input , Xs[i]:input of i-th scale 
    '''
    def __init__(self, in_planes, n_features, n_classes, normalizer):
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)
        
        self.n_features = n_features
        self.in_planes = in_planes
        
        self.norm = nn.ModuleList()
        self.conv = nn.ModuleList()
        for i in range(len(in_planes)):
            self.norm.append(normalizer(in_planes[i], n_classes))
            self.conv.append(conv3x3(in_planes[i], n_features))
            
    def forward(self, Xs, y, shape):
        # Xs[0].shape[0] = batch_size
        s = torch.zeros(Xs[0].shape[0], self.n_features, *shape, device=Xs[0].device)
        for i in range(len(self.conv)):
            tmp = self.norm[i](Xs[i],y)
            tmp = self.conv[i](tmp)
            tmp = F.interpolate(tmp, shape, mode = 'bilinear', align_corners=True)
            s = tmp + s
    
        return s
    
    
class CondRefineBlock(nn.Module):
    '''
    Constructure:
    [CondRCUBlock_1, CondRCUBlock_2, ... , CondRCUBlock_n_block] - CondMSFBlock(if n_block > 1)
    - CondCRPBlock - output_convs
    '''
    def __init__(self, in_planes, n_features, n_classes, normalizer, act = nn.ReLU()):
        super().__init__()
        
        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = len(in_planes)
        
        self.n_features = n_features
        self.n_classes = n_classes
        
        self.Cond_RCU = nn.ModuleList()
        for i in range(self.n_blocks):
            self.Cond_RCU.append(CondRCUBlock(in_planes[i], 2, 2, normalizer, n_classes, act))
            
        if self.n_blocks > 1:
            self.Cond_MSF = CondMSFBlock(in_planes, n_features, n_classes, normalizer)
            
        self.Cond_CRP = CondCRPBlock(n_features, 2, normalizer, n_classes, act)
        self.output_convs = CondRCUBlock(n_features, 1, 2, normalizer, n_classes, act)
        
    
    def forward(self, Xs, y, shape):
        assert isinstance(Xs, tuple) or isinstance(Xs, list)
        tmp = []
        
        for i in range(self.n_blocks):
            tmp.append(self.Cond_RCU[i](Xs[i], y))
            
        if self.n_blocks > 1:
            output = self.Cond_MSF(tmp, y, shape)
        else:
            output = tmp[0]
            
        output = self.Cond_CRP(output, y)
        output = self.output_convs(output, y)
        return output
    
    
    
class ConvMeanPool(nn.Module):
    '''
    Convolution followed by Mean Pooling
    2x downsampling, Halving the resolution
    '''
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False):
        super().__init__()
        if not adjust_padding:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        else:
            self.conv = nn.Sequential(
                nn.ZeroPad2d((1, 0, 1, 0)),
                nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
            )

    def forward(self, inputs):
        output = self.conv(inputs)
        output = sum(
            [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return output


class MeanPoolConv(nn.Module):
    '''
    Mean Pooling followed by Convolution
    2x downsampling, Halving the resolution
    '''
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)

    def forward(self, inputs):
        output = inputs
        output = sum(
            [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
        return self.conv(output)
    
    
class UpsampleConv(nn.Module):
    '''
    Contructure:
    (conv2d - (features x 4) - PixelShuffle)
    '''
    def __init__(self, input_dim, output_dim, kernel_size=3, biases=True):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=kernel_size // 2, bias=biases)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, inputs):
        output = inputs
        output = torch.cat([output, output, output, output], dim=1)
        output = self.pixelshuffle(output)
        return self.conv(output)
    
    
class ConditionalResidualBlock(nn.Module):
    '''
    Conditional Residual Block with optional resampling
    
    resample: 'down' or None -> this class support downsampling or keep resolution
    dilation: int or None ->None means that using conv3x3, int means that using dilated_conv3x3 with dilation rate
    input_dim: in_planes num of input feature graph
    output_dim: out_planes num of output feature graph
    
    
    Constructure:
    normalize1 - activation - conv1 - normalize2 - activation - conv2 ---->output
    identity / shortcut() ------------------------------------------------>
    '''
    
    def __init__(self, input_dim ,output_dim, num_classes ,normalization, 
                 resample = None, act = nn.ELU(), dilation = None, adjust_padding = False):
        super().__init__()
        
        self.act = act
        self.dilation = dilation
        self.resample = resample
        self.norm1 = normalization(input_dim, num_classes)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # downsample
        if resample == 'down':
            # dilation_conv is be used
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)
            # dilation_conv isn't be used, replace by conv
            else:
                self.conv1 = nn.Conv2d(input_dim, input_dim, 3, stride=1, padding=1)
                self.normalize2 = normalization(input_dim, num_classes)
                self.conv2 = ConvMeanPool(input_dim, output_dim, 3, adjust_padding=adjust_padding)
                conv_shortcut = partial(ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding)

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = partial(nn.Conv2d, kernel_size = 3, padding = 1)
                self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
                self.normalize2 = normalization(output_dim, num_classes)
                self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        else:
            raise Exception('invalid resample value')
                
        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim, num_classes)
        
    def forward(self, X, y):
        output = self.normalize1(X, y)
        output = self.act(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.act(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = X
        else:
            shortcut = self.shortcut(X)

        return shortcut + output


class InstanceNorm2dPlus(nn.Module):
    '''
    InstanceNorm2d + learnable Affine Map 
    InstanceNorm means norm in each sample and each feature graph
    '''
    def __init__(self, num_features, num_classes, bias=True):
        self.bias = bias
        self.num_features = num_features
        self.num_classes = num_classes
        
        self.norm = nn.InstanceNorm2d(num_features, affine = False, track_running_stats = False)
        
        self.alpha = nn.Parameter(torch.zeros((num_features,)))
        self.gamma = nn.Parameter(torch.zeros((num_features, )))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        
        if bias is True:
            self.beta = nn.Parameter(torch.zeros((num_features,)))
            
            
    def forward(self, X):
        # mean.shape = (batch_size, features)
        means = torch.mean(X, dim = (2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m)/(torch.sqrt(1e-5 + v))
        
        X = self.norm(X)
        X = means[...,None,None] * self.alpha.view(1, self.num_features, 1, 1) + X
        
        if self.bias is True:
            X = X * self.gamma.view(1, self.num_features, 1, 1) + self.beta.view(1, self.num_features, 1, 1)
        else:
            X = X * self.gamma.view(1, self.num_features, 1, 1)
            
        return X 
        

class ConditionalInstanceNorm2dPlus(nn.Module):
    '''
    ConditionalInstanceNorm2dPlus is a InstanceNorm2dPlus replace constant alpha, beta and gamma
    by nn.embedding (this allow us normalize conditionally)
    '''
    def __init__(self, num_features, num_classes, bias = True):
        super().__init__()
        self.bias = bias
        self.num_features = num_features
        self.num_classes = num_classes
        self.norm = nn.InstanceNorm2d(num_features ,affine = False ,track_running_stats = False)

        if bias is True:
            self.embed = nn.Embedding(num_classes, num_features * 3)
            self.embed.weight.data[:,: (num_features * 2)].normal_(1, 0.02)
            self.embed.weight.data[:,(num_features * 2) :].zero_()
        else:
            self.embed = nn.Embedding(num_classes, num_features *2)
            self.embed.weight.data[:,: (num_features * 2)].normal_(1, 0.02)
            
    def forward(self, X, y):
        mean = torch.mean(X, dim = (2, 3))
        m = torch.mean(mean, dim=-1, keepdim=True)
        v = torch.var(mean, dim=-1, keepdim=True)
        mean = (mean - m)/ torch.sqrt(1e-5 + v)
        X = self.norm(X)
        
        if self.bias is True:
            alpha, gamma, beta = self.embed(y).chunk(3, dim = -1) 
            X = mean[...,None, None] * alpha[...,None, None] + X
            X = X * gamma[...,None, None] + beta[...,None, None]
        else:
            alpha, gamma = self.embed(y).chunk(2, dim = -1)
            X = mean[...,None, None] * alpha[...,None, None] + X
            X = X * gamma[...,None, None]
            
        return X






class RefineNetDilated(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.norm = InstanceNorm2dPlus
        # ngf is the number of generator filters in the first conv layer
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = nn.ELU()
        # self.act = act = nn.ReLU(True)

        # 3 means kernal size 3,config.data.channels means input image channels, ngf means output channels
        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act, normalization=self.norm)
            
        self.res2 = ConditionalResidualBlock(self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act, normalization=self.norm)
        

        self.res3 = nn.ModuleList([
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, dilation=2),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, dilation=2)]
        )

        if config.data.image_size == 28:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                         normalization=self.norm, adjust_padding=True, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                         normalization=self.norm, dilation=4)]
            )
        else:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                         normalization=self.norm, adjust_padding=False, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                         normalization=self.norm, dilation=4)]
            )

        self.refine1 = CondRefineBlock([2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act, start=True)
        self.refine2 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act)
        self.refine3 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, self.num_classes, self.norm, act=act)
        self.refine4 = CondRefineBlock([self.ngf, self.ngf], self.ngf, self.num_classes, self.norm, act=act, end=True)

    def _compute_cond_module(self, module, x, y):
        for m in module:
            x = m(x, y)
        return x

    def forward(self, x):
        if not self.logit_transform:
            x = 2 * x - 1.

        y = None
        output = self.begin_conv(x)

        layer1 = self._compute_cond_module(self.res1, output, y)
        layer2 = self._compute_cond_module(self.res2, layer1, y)
        layer3 = self._compute_cond_module(self.res3, layer2, y)
        layer4 = self._compute_cond_module(self.res4, layer3, y)

        ref1 = self.refine1([layer4], y, layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], y, layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], y, layer2.shape[2:])
        output = self.refine4([layer1, ref3], y, layer1.shape[2:])

        output = self.normalizer(output, y)
        output = self.act(output)
        output = self.end_conv(output)
        return output
    
    
    
class CondRefineNetDilated(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        # self.norm = ConditionalInstanceNorm2d
        self.norm = ConditionalInstanceNorm2dPlus
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = nn.ELU()
        # self.act = act = nn.ReLU(True)

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ConditionalResidualBlock(self.ngf, self.ngf, self.num_classes, resample=None, act=act, normalization=self.norm)
        ])
            
        self.res2 = nn.ModuleList([ 
            ConditionalResidualBlock(self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,           normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                     normalization=self.norm, dilation=2),
            ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                     normalization=self.norm, dilation=2)]
        )

        if config.data.image_size == 28:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                         normalization=self.norm, adjust_padding=True, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                         normalization=self.norm, dilation=4)]
            )
        else:
            self.res4 = nn.ModuleList([
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample='down', act=act,
                                         normalization=self.norm, adjust_padding=False, dilation=4),
                ConditionalResidualBlock(2 * self.ngf, 2 * self.ngf, self.num_classes, resample=None, act=act,
                                         normalization=self.norm, dilation=4)]
            )

        self.refine1 = CondRefineBlock([2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act)
        self.refine2 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, self.num_classes, self.norm, act=act)
        self.refine3 = CondRefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, self.num_classes, self.norm, act=act)
        self.refine4 = CondRefineBlock([self.ngf, self.ngf], self.ngf, self.num_classes, self.norm, act=act)

    def _compute_cond_module(self, module, x, y):
        for m in module:
            x = m(x, y)
        return x

    def forward(self, x, y):
        if not self.logit_transform:
            x = 2 * x - 1.

        output = self.begin_conv(x)

        layer1 = self._compute_cond_module(self.res1, output, y)
        layer2 = self._compute_cond_module(self.res2, layer1, y)
        layer3 = self._compute_cond_module(self.res3, layer2, y)
        layer4 = self._compute_cond_module(self.res4, layer3, y)

        ref1 = self.refine1([layer4], y, layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], y, layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], y, layer2.shape[2:])
        output = self.refine4([layer1, ref3], y, layer1.shape[2:])

        output = self.normalizer(output, y)
        output = self.act(output)
        output = self.end_conv(output)
        return output
