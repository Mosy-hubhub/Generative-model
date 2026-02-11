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
from models.DiT import DiT
from .import model_runner
from diffusers.models import AutoencoderKL
from utils.diffusion import create_diffusion


class DiTRunner(model_runner):
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
            # transform into [0, 1]
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
        
        latent_size = self.config.data.image_size // 8
        model = DiT(input_size = latent_size,
                    patch_size = self.config.model.patch_size,
                    depth = self.config.model.depth, 
                    hidden_dim = self.config.model.hidden_dim,  
                    num_heads = self.config.model.num_heads,
                    num_classes = self.config.data.num_classes,
                    mlp_ratio = self.config.model.mlp_ratio,
                    learn_sigma = self.config.model.learn_sigma
                    ).to(self.config.device)
        model = torch.nn.DataParallel(model) 

        diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{self.config.model.vae}").to(self.config.device)
        logging.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        optimizer = self.get_optimizer(model.parameters())
        
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            model.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
        
        step = 0
       
        for epoch in range(self.config.training.n_epochs):
            for i, (X,y) in enumerate(train_loader):
                step += 1
                model.module.train()
                
                X = X.to(self.config.device)
                y = y.to(self.config.device)
                if self.config.data.logit_transform is True:
                    X = self.logit_transform(X)
                
                with torch.no_grad():
                # Map input images to latent space + normalize latents:
                    X = vae.encode(X).latent_dist.sample().mul_(0.18215)
                
                t = torch.randint(0, diffusion.num_timesteps, (X.shape[0],), device=self.config.device)
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, X, t, model_kwargs)
                loss = loss_dict["loss"].mean()
        
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                tb_logger.add_scalar('loss', loss, global_step=step)
                logging.info("step: {}, loss: {}".format(step, loss.item()))
                
                if step >= self.config.training.n_iters:
                    return 0
                
                if step % 100 == 0:
                    model.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)
                        
                    test_X = test_X.to(self.config.device)
                    test_y = test_y.to(self.config.device)
                        
                    if self.config.data.logit_transform is True:
                        X = self.logit_transform(X)
                        
                    with torch.no_grad():
                        X = vae.encode(X).latent_dist.sample().mul_(0.18215)
                        t = torch.randint(0, diffusion.num_timesteps, (test_X.shape[0],), device=self.config.device)
                        model_kwargs = dict(y = y)
                        loss_dict = diffusion.training_losses(model, test_X, t, model_kwargs)
                    
                    tb_logger.add_scalar('test_loss', loss_dict['loss'], global_step=step)
                               
                if step % self.config.training.snapshot_freq == 0:
                    # save model checkpoint
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))
    
    
    
    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
       
        model = DiT(input_size = latent_size,
                    patch_size = self.config.model.patch_size,
                    depth = self.config.model.depth, 
                    hidden_dim = self.config.model.hidden_dim,  
                    num_heads = self.config.model.num_heads,
                    num_classes = self.config.data.num_classes,
                    mlp_ratio = self.config.model.mlp_ratio,
                    learn_sigma = self.config.model.learn_sigma
                    ).to(self.config.device)
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{self.config.model.vae}").to(self.config.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0])
        diffusion = create_diffusion(str(self.config.sampling.num_sampling_steps))

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        grid_size = 4
        
        model.module.eval()
        
        if self.config.data.dataset == 'CIFAR10':
            # Labels to condition the model with (feel free to change):
            class_labels = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
            assert len(class_labels) == grid_size ** 2, f'the num of class labels must equal to the square of grid to become a image grid' 
            
        else: 
            raise NotImplementedError('dataset {} not understood.'.format(self.config.data.dataset))

        # Create sampling noise:
        latent_size = self.config.data.image_size // 8
        n = len(class_labels)
        z = torch.randn(n, 4, latent_size, latent_size, device = self.config.device)
        y = torch.tensor(class_labels, device = self.config.device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([self.config.data.num_classes] * n, device = self.config.device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=self.config.sampling.cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(model = model.module.forward_with_cfg,
                                          shape = z.shape,
                                          noise = z,
                                          clip_denoised = False,
                                          model_kwargs = model_kwargs,
                                          progress = True,
                                          device = self.config.device)
        samples = vae.decode(samples / 0.18215).sample
        if self.config.data.logit_transform:
                samples = torch.sigmoid(samples)
        samples_cfg, samples_null = samples.chunk(2, dim=0)

        image_cfg_grid = make_grid(samples_cfg, nrow = grid_size)
        samples_null_grid = make_grid(samples_null, nrow = grid_size)

        save_image(image_cfg_grid, os.path.join(self.args.image_folder, 'image_cfg.png'),
                   normalize = True, value_range = (-1, 1))
        torch.save(samples_cfg, os.path.join(self.args.image_folder, 'image_cfg_raw.pth'))
        save_image(samples_null_grid, os.path.join(self.args.image_folder, 'image_null.png'),
                   normalize = True, value_range = (-1, 1))
        torch.save(samples_null, os.path.join(self.args.image_folder, 'image_null_raw.pth'))       