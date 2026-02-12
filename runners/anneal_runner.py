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
from losses.dsm import dsm_baseline, dsm_anneal
from losses.sliced_sm import ssm_baseline, ssm_anneal_vr
import tqdm
from torchvision.utils import save_image, make_grid
from PIL import Image
from .abstruct_runner import model_runner

class AnnealRunner(model_runner):
    def __init__(self, args, config):
        self.config = config
        self.args = args
        
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)
        self.sigmas = sigmas
            
        
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
        
        scorenet = CondRefineNetDilated(self.config)
        scorenet = torch.nn.DataParallel(scorenet) 
        
        optimizer = self.get_optimizer(scorenet.parameters())
        
        
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            scorenet.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
        
        step = 0
        
       
        for epoch in range(self.config.training.n_epochs):
            for i, (X,y) in enumerate(train_loader):
                step += 1
                scorenet.train()
                
                X = X.to(self.config.device)
                if self.config.data.logit_transform is True:
                    X = self.logit_transform(X)
                
                
                labels = torch.randint(0, self.config.model.num_classes - 1, (X.shape[0],), device = X.device)
                
                if self.config.training.algo == 'ssm_annel':
                    loss = ssm_anneal_vr(scorenet, X)
                elif self.config.training.algo == 'dsm_anneal':
                    loss = dsm_anneal(scorenet, X, self.sigmas, labels)
                else:
                    raise NotImplementedError('loss_function {} not understood.'.format(self.config.training.algo))
        
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))
                
                if step >= self.config.training.n_iters:
                    return 0
                
                if step % 100 == 0:
                    scorenet.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)
                        
                    test_X = test_X.to(self.config.device)
                    test_y = test_y.to(self.config.device)
                        
                    with torch.no_grad():
                        if self.config.training.algo == 'ssm_annel':
                            loss = ssm_anneal_vr(scorenet, test_X)
                        elif self.config.training.algo == 'dsm_anneal':
                            loss = dsm_anneal(scorenet, test_X, self.sigmas, labels, self.config.training.anneal_power)
                        else:
                            raise NotImplementedError('loss_function {} not understood.'.format(self.config.training.algo))
        
        
                    tb_logger.add_scalar('test_{}_loss'.format(self.config.training.algo), loss, global_step=step)
                    
                if step % self.config.training.snapshot_freq == 0:
                    # save model checkpoint
                    states = [
                        scorenet.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))

        
    def Langevin_dynamics(self, scorenet, x_mod, n_steps=2000, step_lr=2e-5):
        images = []

        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 9
        labels = labels.long()

        with torch.no_grad():
            for _ in range(n_steps):
                images.append(torch.clamp(x_mod, 0.0, 1.0).to('cpu'))
                noise = torch.randn_like(x_mod) * np.sqrt(step_lr * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_lr * grad + noise
                x_mod = x_mod
                print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

            return images
        
        
    def anneal_Langevin_dynamics(self, scorenet, X_mod, num_steps_each_scale, epsilon = 2e-5):
        '''
        Langevin method recursively
        x_t = x_{t-1}+ 1/2 * \epsilon * s(x_{t-1}, noise_scale)+ sqrt{\epsilon} * z_{t-1}
        '''        
        image = []
           
        with torch.no_grad():
            for i , sigma in tqdm.tqdm(enumerate(self.sigmas), total=self.config.model.num_classes, desc='annealed Langevin dynamics sampling'):
                step_length = epsilon * (sigma / self.sigmas[-1]) ** 2
                label = torch.ones((X_mod.shape[0],), device = self.config.device) * i
                label = label.long()
                
                for _ in range(num_steps_each_scale):
                    image.append(torch.clamp(X_mod, 0,  1).to('cpu'))
                    noise = torch.randn_like(X_mod)
                    grad = scorenet(X_mod, label)
                    X_mod = X_mod + 1/2. * step_length * grad  + torch.sqrt(step_length) * noise
                    print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))
        
        return image
    
    
    
    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(device = self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        
        scorenet.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)
        
        imgs = []
        grid_size = 5
        
        scorenet.eval()
        
        if self.config.data.dataset == 'CIFAR10':
            X_mod = torch.rand((grid_size ** 2, 3, 32, 32), device = self.config.device)
            if self.config.model.generater == 'Langevin_dynamics':
                all_samples = self.Langevin_dynamics(scorenet, X_mod, 2000,  4e-5)
            elif self.config.model.generater == 'anneal_Langevin_dynamics':
                all_samples = self.anneal_Langevin_dynamics(scorenet, X_mod, 200, 4e-5)
            else:
                raise NotImplementedError('generater {} not understood.'.format(self.config.model.generater))
        
            for i, sample in enumerate(tqdm.tqdm(all_samples, total=len(all_samples), desc='saving images')):
                sample = sample.view(grid_size ** 2, self.config.data.channels, self.config.data.image_size,
                                     self.config.data.image_size)

                if self.config.data.logit_transform:
                    sample = torch.sigmoid(sample)

                image_grid = make_grid(sample, nrow=grid_size)
                if i % 10 == 0:
                    im = Image.fromarray(image_grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())
                    imgs.append(im)

                save_image(image_grid, os.path.join(self.args.image_folder, self.args.doc, 'image_{}.png'.format(i)))
                torch.save(sample, os.path.join(self.args.image_folder, self.args.doc, 'image_raw_{}.pth'.format(i)))
        
        else: 
            raise NotImplementedError('dataset {} not understood.'.format(self.config.data.dataset))
        
        
        imgs[0].save(os.path.join(self.args.image_folder, self.args.doc, "movie.gif"), save_all=True, append_images=imgs[1:], duration=1, loop=0)
