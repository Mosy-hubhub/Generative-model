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
from .abstruct_runner import model_runner
from losses.dsm import cond_dsm_anneal, cond_vpred_anneal
from utils.diffusion import create_diffusion
import tqdm
from PIL import Image


class DiT_pixel_Runner(model_runner):
    """
    this version only has DiT, take diffusion in pixel space
    """
    def __init__(self, args, config):
        self.config = config
        self.args = args        
        if config.sampling.sample_way == 'SMLD':
            sigmas = torch.tensor(
            np.exp(np.linspace(np.log(50), np.log(0.01), self.config.model.num_timesteps))).float().to(self.config.device)
            self.sigmas = sigmas
        
    def train(self):
        '''
        step1: create two function for data preprocessing (self.config.data.random_flip:true or false)
        step2: get dataset
        step3: create dataloader, testloader, tg_logger, score(scorenet to train), optimizer
        step4: create sigmas as noise scale list
        step5: training loop, write log, get loss function and update parameter
        '''
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if self.config.data.random_flip is False:
            # transform into [0, 1]
            train_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor(),
                normalize
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
        
        latent_size = self.config.data.image_size
        model = DiT(input_size = latent_size,
                    patch_size = self.config.model.patch_size,
                    in_channels = 3,
                    depth = self.config.model.depth, 
                    hidden_dim = self.config.model.hidden_dim,  
                    num_heads = self.config.model.num_heads,
                    num_classes = self.config.data.num_classes,
                    mlp_ratio = self.config.model.mlp_ratio,
                    learn_sigma = self.config.model.learn_sigma,
                    ).to(self.config.device)
        # model = torch.nn.DataParallel(model) 

        if self.config.sampling.sample_way == 'DDPM':
            diffusion = create_diffusion(timestep_respacing="",
                                        learn_sigma = False,
                                        predict_xstart = self.config.model.predict_xstart,
                                        diffusion_steps = self.config.model.num_timesteps
                                        )  # default: 1000 steps, linear noise schedule
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
                model.train()
                
                X = X.to(self.config.device)
                y = y.to(self.config.device)
                if self.config.data.logit_transform is True:
                    X = self.logit_transform(X)
                
                if self.config.sampling.sample_way == 'DDPM':
                    t = torch.randint(0, diffusion.num_timesteps, (X.shape[0],), device=self.config.device)
                    model_kwargs = dict(y=y)
                    loss_dict = diffusion.training_losses(model, X, t, model_kwargs)
                    loss = loss_dict["loss"].mean()
                elif self.config.sampling.sample_way == 'SMLD':
                    t = torch.randint(0, self.config.model.num_timesteps, (X.shape[0],), device=self.config.device)
                    # loss = cond_dsm_anneal(model, X, y, t, self.sigmas)
                    loss = cond_vpred_anneal(model, X, y, t, self.sigmas)
                    
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
                        test_X = self.logit_transform(test_X)
                        
                    with torch.no_grad():
                        if self.config.sampling.sample_way == 'DDPM':
                            t = torch.randint(0, diffusion.num_timesteps, (test_X.shape[0],), device=self.config.device)
                            model_kwargs = dict(y = test_y)
                            loss_dict = diffusion.training_losses(model, test_X, t, model_kwargs)
                            tb_logger.add_scalar('test_loss', loss_dict["loss"].mean(), global_step=step)
                        elif self.config.sampling.sample_way == 'SMLD':
                            t = torch.randint(0, self.config.model.num_timesteps, (test_X.shape[0],), device=self.config.device)
                            loss = cond_dsm_anneal(model, test_X, test_y, t, self.sigmas)
                            tb_logger.add_scalar('test_loss', loss, global_step=step)
                    
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
        
        latent_size = self.config.data.image_size
       
        model = DiT(input_size = latent_size,
                    patch_size = self.config.model.patch_size,
                    in_channels = 3,
                    depth = self.config.model.depth, 
                    hidden_dim = self.config.model.hidden_dim,  
                    num_heads = self.config.model.num_heads,
                    num_classes = self.config.data.num_classes,
                    mlp_ratio = self.config.model.mlp_ratio,
                    learn_sigma = self.config.model.learn_sigma
                    ).to(self.config.device)
        # model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0])
        diffusion = create_diffusion(str(self.config.sampling.num_sampling_steps),
                                     learn_sigma = False,
                                     predict_xstart = True)

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        grid_size = 4
        
        model.eval()
        
        if self.config.data.dataset == 'CIFAR10':
            # Labels to condition the model with (feel free to change):
            class_labels = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
            assert len(class_labels) == grid_size ** 2, f'the num of class labels must equal to the square of grid to become a image grid' 
            
        else: 
            raise NotImplementedError('dataset {} not understood.'.format(self.config.data.dataset))

        # Create sampling noise:
        latent_size = self.config.data.image_size
        n = len(class_labels)
        z = torch.randn(n, 3, latent_size, latent_size, device = self.config.device)
        y = torch.tensor(class_labels, device = self.config.device)

        # Setup classifier-free guidance:
        z = torch.cat([z, z], 0)
        y_null = torch.tensor([self.config.data.num_classes] * n, device = self.config.device)
        y = torch.cat([y, y_null], 0)
        model_kwargs = dict(y=y, cfg_scale=self.config.sampling.cfg_scale)
        
                
        if self.config.sampling.sample_way == 'DDPM':
            # Sample images:
            samples = diffusion.p_sample_loop(model = model.forward_with_cfg,
                                            shape = z.shape,
                                            noise = z,
                                            clip_denoised = False,
                                            model_kwargs = model_kwargs,
                                            progress = True,
                                            device = self.config.device)
            if self.config.data.logit_transform:
                samples = torch.sigmoid(samples)
            samples_cfg, samples_null = samples.chunk(2, dim=0)

            image_cfg_grid = make_grid(samples_cfg, nrow = grid_size)
            samples_null_grid = make_grid(samples_null, nrow = grid_size)
        
            save_dir = os.path.join(self.args.image_folder, self.args.doc)
            os.makedirs(save_dir, exist_ok=True)

            save_image(image_cfg_grid, os.path.join(self.args.image_folder, self.args.doc, 'image_cfg.png'),
                    normalize = True, value_range = (-1, 1))
            save_image(samples_null_grid, os.path.join(self.args.image_folder, self.args.doc, 'image_null.png'),
                    normalize = True, value_range = (-1, 1))    
            
        elif self.config.sampling.sample_way == 'SMLD':
            self.cond_anneal_Langevin_dynamics(model, z, y, 50, 4e-5)
            
    
    def cond_anneal_Langevin_dynamics(self, v_net, X_mod, y, num_steps_each_scale, epsilon=2e-5):
        '''
        Langevin method recursively with GIF frame collection and final result preservation.
        '''        
        gif_frames_null = []
        gif_frames_cfg = []
        grid_size = int((X_mod.shape[0] / 2) ** 0.5)
    
        # 比如总步数 = len(sigmas) * num_steps_each_scale，我们只取其中 50-100 帧
        total_total_steps = len(self.sigmas) * num_steps_each_scale
        save_interval = max(1, total_total_steps // 100) 
        curr_step = 0
        
        save_dir = os.path.join(self.args.image_folder, self.args.doc)
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            for i, sigma in tqdm.tqdm(enumerate(self.sigmas), total=len(self.sigmas), desc='annealed Langevin dynamics sampling'):
                step_length = epsilon * (sigma / self.sigmas[-1]) ** 2
                t = torch.tensor([i] * X_mod.shape[0], device = X_mod.device)
            
                for _ in range(num_steps_each_scale):
                    # --- GIF 帧收集逻辑 ---
                    if curr_step % save_interval == 0:
                        sample_to_show = X_mod.clone()
                        if self.config.data.logit_transform:
                            sample_to_show = torch.sigmoid(sample_to_show)
                        sample_to_show = torch.clamp(sample_to_show, 0, 1)
                        sample_to_show_cfg, sample_to_show_null = sample_to_show.chunk(2, dim=0)
                    
                        grid_null = make_grid(sample_to_show_null, nrow=grid_size)
                        grid_cfg = make_grid(sample_to_show_cfg, nrow=grid_size)
                        im_data_null = grid_null.mul(255).add_(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        im_data_cfg = grid_cfg.mul(255).add_(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        
                        gif_frames_null.append(Image.fromarray(im_data_null))
                        gif_frames_cfg.append(Image.fromarray(im_data_cfg))
                
                    noise = torch.randn_like(X_mod)
                    # grad =  - noisenet.forward_with_cfg(X_mod / torch.sqrt(1 + sigma ** 2), t, y, self.config.sampling.cfg_scale) / sigma
                    
                    #######
                    alpha = 1.0 / torch.sqrt(1.0 + sigma ** 2)
                    sigma_vp = sigma / torch.sqrt(1.0 + sigma ** 2)
                    
                    # 2. 构造网络的安全输入 (方差恒定为 1)
                    X_input = X_mod * alpha
                    
                    # 3. v_net 吐出带 CFG 的 v_pred！
                    # (注意这里把你的 noisenet 改成了 v_net，请和你代码里的变量名保持一致)
                    v_pred = v_net.forward_with_cfg(X_input, t, y, self.config.sampling.cfg_scale)
                    
                    # 4. 【数学魔法解方程】反推出纯正的预测噪声 epsilon
                    noise_pred = alpha * v_pred + sigma_vp * X_input
                    
                    # 5. 算 SMLD 的梯度 (Score = -epsilon / sigma)
                    grad = - noise_pred / sigma
                    
                    ######
                    X_mod = X_mod + 1/2. * step_length * grad + torch.sqrt(step_length) * noise
                
                    curr_step += 1
                    print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

        final_sample = X_mod.clone()
        if self.config.data.logit_transform:
            final_sample = torch.sigmoid(final_sample)
        final_sample = torch.clamp(final_sample, 0, 1)
        final_sample_cfg, final_sample_null = final_sample.chunk(2, dim=0)

        if gif_frames_null:
            gif_frames_null[0].save(os.path.join(self.args.image_folder, self.args.doc, 'diffusion_process.gif'),
                                    save_all=True,
                                    append_images=gif_frames_null[1:],
                                    duration=10,
                                    loop=0)
            gif_frames_cfg[0].save(os.path.join(self.args.image_folder, self.args.doc, 'diffusion_process_cfg.gif'),
                                   save_all=True,
                                   append_images=gif_frames_cfg[1:],
                                   duration=10,
                                   loop=0)
            print(f"\n[Saved] GIF saved to {self.args.image_folder}")
            
        torch.save(final_sample_cfg, os.path.join(self.args.image_folder, self.args.doc, 'image_raw_cfg.pth'))
        torch.save(final_sample_null, os.path.join(self.args.image_folder, self.args.doc, 'image_raw.pth'))