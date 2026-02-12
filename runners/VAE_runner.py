import torch
import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import os
import logging
from torch.utils.data import DataLoader
import tensorboardX
import shutil
from torchvision.utils import save_image, make_grid
from models.VAE_constructure import VAE_model
from losses.VAE_loss import ELBO
from .abstruct_runner import model_runner

class VAERunner(model_runner):
    def __init__(self, args, config):
        self.config = config
        self.args = args
    
       
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
        
        VAE_network = VAE_model(self.config, self.config.model.model_type).to(device = self.config.device)
        VAE_network = torch.nn.DataParallel(VAE_network)
        optimizer = self.get_optimizer(VAE_network.parameters())
        
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            VAE_network.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
        
        step = 0
        
       
        for epoch in range(self.config.training.n_epochs):
            for i, (X,y) in enumerate(train_loader):
                step += 1
                VAE_network.module.train()
                
                X = X.to(self.config.device)
                if self.config.data.logit_transform is True:
                    X = self.logit_transform(X)
                
                
                if self.config.training.algo == 'ELBO':
                    loss, KL_divergence = ELBO(X, VAE_network)
                else:
                    raise NotImplementedError('loss_function {} not understood.'.format(self.config.training.algo))
        
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}, KL_divergence: {}".format(step, loss.item(), KL_divergence.item()))
                
                if step >= self.config.training.n_iters:
                    return 0
                
                if step % 100 == 0:
                    VAE_network.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)
                        
                    test_X = test_X.to(self.config.device)
                    test_y = test_y.to(self.config.device)
                        
                    with torch.no_grad():
                        if self.config.training.algo == 'ELBO':
                            loss, KL_divergence = ELBO(test_X, VAE_network)
                        else:
                            raise NotImplementedError('loss_function {} not understood.'.format(self.config.training.algo))
        
        
                    tb_logger.add_scalar('test_{}_loss'.format(self.config.training.algo), loss, global_step=step)
                    tb_logger.add_scalar('test_KL_divergence', KL_divergence, global_step=step)
                    
                if step % self.config.training.snapshot_freq == 0:
                    # save model checkpoint
                    states = [
                        VAE_network.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))
    
    
    
    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
       
        VAE_network = VAE_model(self.config, self.config.model.model_type).to(device = self.config.device)
        VAE_network = torch.nn.DataParallel(VAE_network)
        VAE_network.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        grid_size = 5
        
        VAE_network.module.eval()
        
        if self.config.data.dataset == 'CIFAR10':
            z = torch.randn((grid_size ** 2, self.config.model.latent_dimension), device = self.config.device)
            with torch.no_grad():
                output_mean = VAE_network.module.decoder(z)
            all_samples = torch.normal(output_mean, torch.ones_like(output_mean))
            
            if self.config.data.logit_transform:
                all_samples = torch.sigmoid(all_samples)
            
            all_samples = torch.clamp(all_samples, 0,  1).to('cpu')
            all_samples = all_samples.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size, 
                                           self.config.data.image_size)

            image_grid = make_grid(all_samples, nrow=grid_size)

            save_image(image_grid, os.path.join(self.args.image_folder, self.args.doc, 'image.png'))
            torch.save(all_samples, os.path.join(self.args.image_folder, self.args.doc, 'image_raw.pth'))
        
        else: 
            raise NotImplementedError('dataset {} not understood.'.format(self.config.data.dataset))