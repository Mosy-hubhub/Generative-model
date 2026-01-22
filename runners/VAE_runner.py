import torch
import numpy as np
from torchvision.datasets import CIFAR10
import torch.optim as optim
import torchvision.transforms as transforms
import os
import logging
from torch.utils.data import DataLoader
import tensorboardX
import shutil
from models.refinenet import CondRefineNetDilated
import tqdm
from torchvision.utils import save_image, make_grid
from PIL import Image
from models.VAE_constructure import VAE_model_BN_ver1, VAE_model_BN_ver2, VAE_model_ver1, VAE_model_ver2
from losses.VAE_loss import ELBO


class AnnealRunner():
    def __init__(self, args, config):
        self.config = config
        self.args = args
        
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)
        self.sigmas = sigmas
    
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
        
        
    def train(self):
        '''
        step1: create two function for data preprocessing (self.config.data.random_flip:true or false)
        step2: get dataset
        step3: create dataloader, testloader, tg_logger, score(scorenet to train), optimizer
        step4: create sigmas as noise scale list
        step5: training loop, write log, get loss function and update parameter
        '''
        if self.config.data.random_flip is False:
            train_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
            
        
        if self.config.data.dataset == 'CIFAR10':
            train_dataset =  CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                                    transform = train_transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True,
                                    transform = test_transform)
        else:
            raise NotImplementedError('dataset {} not understood.'.format(self.config.data.dataset))
        
        
        train_loader = DataLoader(train_dataset, self.config.training.batch_size, shuffle = True, num_workers= 4, drop_last = True)
        test_loader = DataLoader(test_dataset, self.config.training.batch_size, shuffle = True, num_workers= 4, drop_last = True)
        test_iter = iter(test_loader)
        
        
        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        tb_logger = tensorboardX.SummaryWriter(log_dir = tb_path)
        
        if self.config.model.model_type == 'VAE_model_ver2':
            VAE_model = VAE_model_ver2(self.config).to(device = self.config.device)
        elif self.config.model.model_type == 'VAE_model_BN_ver2':
            VAE_model = VAE_model_BN_ver2(self.config).to(device = self.config.device)
        elif self.config.model.model_type == 'VAE_model_BN_ver1':
            VAE_model = VAE_model_BN_ver1(self.config).to(device = self.config.device)
        elif self.config.model.model_type == 'VAE_model_ver1':
            VAE_model = VAE_model_ver1(self.config).to(device = self.config.device)
        else:
            raise NotImplementedError('model type {} not understood.'.format(self.config.model.model_type))
            
        VAE_model = torch.nn.DataParallel(VAE_model)
        
        optimizer = self.get_optimizer(VAE_model.parameters())
        
        
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            VAE_model.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
        
        step = 0
        
       
        for epoch in range(self.config.training.n_epochs):
            for i, (X,y) in enumerate(train_loader):
                step += 1
                VAE_model.train()
                
                X = X.to(self.config.device)
                if self.config.data.logit_transform is True:
                    X = self.logit_transform(X)
                
                
                if self.config.training.algo == 'ELBO':
                    loss = ELBO(VAE_model, X)
                else:
                    raise NotImplementedError('loss_function {} not understood.'.format(self.config.training.algo))
        
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))
                
                if step % 100 == 0:
                    VAE_model.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)
                        
                    test_X = test_X.to(self.config.device)
                    test_y = test_y.to(self.config.device)
                        
                    with torch.no_grad():
                        if self.config.training.algo == 'ELBO':
                            loss = ELBO(VAE_model, test_X)
                        else:
                            raise NotImplementedError('loss_function {} not understood.'.format(self.config.training.algo))
        
        
                    tb_logger.add_scalar('test_{}_loss'.format(self.config.training.algo), loss, global_step=step)
                    
                if step % self.config.training.snapshot_freq == 0:
                    # save model checkpoint
                    states = [
                        VAE_model.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))
    
    
    
    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        if self.config.model.model_type == 'VAE_model_ver2':
            VAE_model = VAE_model_ver2(self.config).to(device = self.config.device)
        elif self.config.model.model_type == 'VAE_model_BN_ver2':
            VAE_model = VAE_model_BN_ver2(self.config).to(device = self.config.device)
        elif self.config.model.model_type == 'VAE_model_BN_ver1':
            VAE_model = VAE_model_BN_ver1(self.config).to(device = self.config.device)
        elif self.config.model.model_type == 'VAE_model_ver1':
            VAE_model = VAE_model_ver1(self.config).to(device = self.config.device)
        else:
            raise NotImplementedError('model type {} not understood.'.format(self.config.model.model_type))
            
        VAE_model = torch.nn.DataParallel(VAE_model)
        
        VAE_model.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        grid_size = 5
        
        VAE_model.eval()
        
        if self.config.data.dataset == 'CIFAR10':
            z = torch.randn((grid_size ** 2, self.config.model.latent_dimension), device = self.config.device)
            if self.config.model.generater == 'VAE':
                with torch.no_grad():
                    output_mean = VAE_model.decoder(z)
                all_samples = torch.normal(output_mean, torch.ones_like(output_mean))
            else:
                raise NotImplementedError('generater {} not understood.'.format(self.config.model.generater))
        
            all_samples = torch.clamp(all_samples, 0,  1).to('cpu')
            all_samples = all_samples.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size, 
                                           self.config.data.image_size)

            if self.config.data.logit_transform:
                all_samples = torch.sigmoid(all_samples)

            image_grid = make_grid(all_samples, nrow=grid_size)

            save_image(image_grid, os.path.join(self.args.image_folder, 'image.png'))
            torch.save(all_samples, os.path.join(self.args.image_folder, 'image_raw.pth'))
        
        else: 
            raise NotImplementedError('dataset {} not understood.'.format(self.config.data.dataset))


