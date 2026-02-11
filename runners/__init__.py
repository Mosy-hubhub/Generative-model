from .anneal_runner import AnnealRunner
from .VAE_runner import VAERunner
import torch
import torch.optim as optim
from abc import ABC, abstractmethod

class model_runner(ABC):
    def logit_trans(self, image, lamb = 1e-6):
        '''
        to make data more stable
        y = ln[(lamb + (1 - 2 * lamb) * x) / 1 - (lamb + (1 - 2 * lamb) * x)]
        '''
        image = lamb + (1 - 2 * lamb) * image 
        return torch.log(image) - torch.log1p(-image)
    
    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))
        
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def test(self):
        pass
