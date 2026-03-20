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
from models.Cond_Unet_baseline import AdaGN_UNet_baseline 
from models.Cond_unet_attention import Baseline_UNet_attention_wrapped
from losses.dsm import cond_score_anneal, cond_dsm_anneal
import tqdm
from torchvision.utils import save_image, make_grid
from PIL import Image
from .abstruct_runner import model_runner
from utils.EMA import EMA

class AdaGN_Unet_Runner(model_runner):
    def __init__(self, args, config):
        self.config = config
        self.args = args
        
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_timesteps))).float().to(self.config.device)
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
        
        if self.config.model.model_name == 'AdaGN_Unet_baseline':
            scorenet = AdaGN_UNet_baseline(self.config).to(self.config.device)
        elif self.config.model.model_name == 'AdaGN_Unet_attention':
            scorenet = Baseline_UNet_attention_wrapped(self.config).to(self.config.device)
        else:
            raise NotImplementedError('model {} not understood.'.format(self.config.model.model_name))

        optimizer = self.get_optimizer(scorenet.parameters())

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            scorenet.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
            
        ema = EMA(scorenet, decay = 0.9999)
        step = 0
               
        for epoch in range(self.config.training.n_epochs):
            for i, (X,y) in enumerate(train_loader):
                step += 1
                scorenet.train()
                X = X.to(self.config.device)
                y = y.to(self.config.device)
                X = X * (255. / 256.) + (torch.rand_like(X) * 2) / 256.
                times = torch.randint(0, self.config.model.num_timesteps, (X.shape[0],), device = X.device)
                if self.config.data.logit_transform is True:
                    X = self.logit_transform(X)
                
                loss = cond_score_anneal(scorenet, X, y, times, self.sigmas)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                ema.update()
                
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
                        
                    test_times = torch.randint(0, self.config.model.num_timesteps, (test_X.shape[0],), device=test_X.device)
                    
                    with torch.no_grad():
                        loss = cond_score_anneal(scorenet, test_X, test_y, test_times, self.sigmas)
                        
                    tb_logger.add_scalar('test_loss', loss, global_step=step)
                    
                if step % self.config.training.snapshot_freq == 0:
                    # save model checkpoint
                    ema.apply_shadow()
                    states = [
                        scorenet.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))
                    ema.restore()
    
    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        if self.config.model.model_name == 'AdaGN_Unet_baseline':
            scorenet = AdaGN_UNet_baseline(self.config).to(self.config.device)
        elif self.config.model.model_name == 'AdaGN_Unet_attention':
            scorenet = Baseline_UNet_attention_wrapped(self.config).to(self.config.device)
        else:
            raise NotImplementedError('model {} not understood.'.format(self.config.model.model_name))
        scorenet.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        scorenet.eval()
        
        if self.config.data.dataset == 'CIFAR10':
            if getattr(self.args, 'fid_mode', False):
                self.generate_fid_samples(scorenet, total_samples=50000, batch_size=128)
            else:
                grid_size = 4
                class_labels = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
                assert len(class_labels) == grid_size ** 2, f'the num of class labels must equal to the square of grid to become a image grid' 
                n = len(class_labels)
                z = torch.randn(n, 3, 32, 32, device = self.config.device) * self.sigmas[0]
                y = torch.tensor(class_labels, device = self.config.device)
            
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([self.config.data.num_classes] * n, device = self.config.device)
                y = torch.cat([y, y_null], 0)
            
                self.cond_anneal_Langevin_dynamics(scorenet, z, y, 100, 4e-5)
            
            
            
    def cond_anneal_Langevin_dynamics(self, score_net, X_mod, y, num_steps_each_scale, epsilon=2e-5):
        '''
        Langevin method recursively with GIF frame collection and final result preservation.
        '''        
        gif_frames= []
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
                        sample_to_show = torch.clamp((sample_to_show + 1)/2, 0, 1)
                        sample_to_show_cfg, _ = sample_to_show.chunk(2, dim=0)
                        grid_cfg = make_grid(sample_to_show_cfg, nrow=grid_size)
                        im_data_cfg = grid_cfg.mul(255).add_(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        gif_frames.append(Image.fromarray(im_data_cfg))
                
                    noise = torch.randn_like(X_mod)
                    grad =  score_net.forward_with_cfg(X_mod, t, y, self.config.sampling.cfg_scale)
                    X_mod = X_mod + 1/2. * step_length * grad + torch.sqrt(step_length) * noise
                
                    curr_step += 1
                    print("modulus of grad components: mean {}, max {}".format(grad.abs().mean(), grad.abs().max()))

        final_sample = X_mod.clone()
        if self.config.data.logit_transform:
            final_sample = torch.sigmoid(final_sample)
        final_sample = torch.clamp((final_sample + 1)/2, 0, 1)
        final_sample_cfg, _ = final_sample.chunk(2, dim=0)

        if gif_frames:
            gif_frames[0].save(os.path.join(self.args.image_folder, self.args.doc, 'diffusion_process_cfg.gif'),
                                   save_all=True,
                                   append_images=gif_frames[1:],
                                   duration=10,
                                   loop=0)
            print(f"\n[Saved] GIF saved to {self.args.image_folder}")
            
        torch.save(final_sample_cfg, os.path.join(self.args.image_folder, self.args.doc, 'image_raw_cfg.pth'))



    @torch.no_grad()
    def _pure_langevin_sampling(self, score_net, X_mod, y, num_steps_each_scale, epsilon=2e-5):
        """
        纯净版的采样核，剥离了所有 GIF 和打印逻辑，只管以最快速度算出结果。
        注意：传入的 X_mod 和 y 依然是 2B 结构。
        """
        for i, sigma in enumerate(self.sigmas):
            step_length = epsilon * (sigma / self.sigmas[-1]) ** 2
            t = torch.tensor([i] * X_mod.shape[0], device=X_mod.device)
            
            for _ in range(num_steps_each_scale):
                noise = torch.randn_like(X_mod)
                grad = score_net.forward_with_cfg(X_mod, t, y, self.config.sampling.cfg_scale)
                X_mod = X_mod + 1/2. * step_length * grad + torch.sqrt(step_length) * noise

        final_sample = X_mod
        if self.config.data.logit_transform:
            final_sample = torch.sigmoid(final_sample)
        else:
            final_sample = (final_sample + 1) / 2.0
            
        final_sample = torch.clamp(final_sample, 0, 1)
        final_sample_cfg, _ = final_sample.chunk(2, dim=0)
        return final_sample_cfg


    def generate_fid_samples(self, score_net, total_samples=5000, batch_size=256):
        """
        专门用于生成 FID 评测所需图片的批量生产工厂。
        """
        score_net.eval()
        
        # 创建专属的存放生成图片的文件夹
        save_dir = os.path.join(self.args.image_folder, self.args.doc, 'fid_samples')
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting to generate {total_samples} samples for FID evaluation...")
        print(f"Images will be saved to: {save_dir}")
        
        num_batches = int(np.ceil(total_samples / batch_size))
        samples_generated = 0
        
        with tqdm.tqdm(total=total_samples, desc="Generating Samples") as pbar:
            for i in range(num_batches):
                # 确定当前批次的实际大小 (最后一次可能不够一个满 Batch)
                current_batch_size = min(batch_size, total_samples - samples_generated)
                
                # --- 构建 2B 结构的输入张量 ---
                # 1. 随机类别标签 (或者你可以写死均匀分布的标签)
                # 这里我们假设随机抽取 10 个类
                labels = torch.randint(0, self.config.data.num_classes, (current_batch_size,), device=self.config.device)
                
                # 2. 初始噪声 z
                z = torch.randn(current_batch_size, 3, 32, 32, device=self.config.device) * self.sigmas[0]
                
                # 3. 翻倍成 2B
                z_2b = torch.cat([z, z], 0)
                y_null = torch.tensor([self.config.data.num_classes] * current_batch_size, device=self.config.device)
                y_2b = torch.cat([labels, y_null], 0)
                
                samples_tensor = self._pure_langevin_sampling(score_net, z_2b, y_2b, num_steps_each_scale=200, epsilon=4e-5)
                
                im_data = samples_tensor.mul(255).add_(0.5).clamp(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                
                for j in range(current_batch_size):
                    global_idx = samples_generated + j
                    img = Image.fromarray(im_data[j])
                    
                    # 命名格式: 00001_class_3.png
                    filename = f"{global_idx:05d}_class_{labels[j].item()}.png"
                    img.save(os.path.join(save_dir, filename), format='PNG')
                
                samples_generated += current_batch_size
                pbar.update(current_batch_size)

        print(f"\n[Success] {samples_generated} images have been saved to {save_dir}.")