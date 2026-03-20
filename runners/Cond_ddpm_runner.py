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
from models.Cond_unet_attention import EDM_UNet_attention_wrapped
from losses.dsm import cond_score_anneal, cond_dsm_anneal, EDMLoss
from Sampler.EDM import edm_sampler, edm_sampler_GIF
import tqdm
from torchvision.utils import save_image, make_grid
from PIL import Image
from .abstruct_runner import model_runner
from utils.EMA import EMA

class Cond_DDPM_Runner(model_runner):
    def __init__(self, args, config):
        self.config = config
        self.args = args
        
    def train(self):
        '''
        step1: create two function for data preprocessing (self.config.data.random_flip:true or false)
        step2: get dataset
        step3: create dataloader, testloader, tg_logger, score(xstart_net to train), optimizer
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
        
        num_workers = getattr(self.config.training, 'num_workers', 8)
        train_loader = DataLoader(train_dataset, self.config.training.batch_size, shuffle = True, num_workers= num_workers, drop_last = True)
        test_loader = DataLoader(test_dataset, self.config.training.batch_size, shuffle = True, num_workers= num_workers, drop_last = True)
        test_iter = iter(test_loader)
        
        
        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)
        tb_logger = tensorboardX.SummaryWriter(log_dir = tb_path)
        
        if self.config.model.model_name == 'EDM_AdaGN_Unet_attention':
            xstart_net = EDM_UNet_attention_wrapped(self.config).to(self.config.device)
        else:
            raise NotImplementedError('model {} not understood.'.format(self.config.model.model_name))

        optimizer = self.get_optimizer(xstart_net.parameters())
        loss_caculator = EDMLoss(sigma_data = self.config.data.sigma_data)

        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'))
            xstart_net.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
            
        if torch.cuda.device_count() > 1:
            logging.info(f"Training on {torch.cuda.device_count()} GPUs!")
            xstart_net = torch.nn.DataParallel(xstart_net)
            
        base_model = xstart_net.module if hasattr(xstart_net, 'module') else xstart_net
        ema = EMA(base_model, decay = 0.9999)
        # ema = EMA(xstart_net, decay = 0.9999)
        step = 0
               
        for epoch in range(self.config.training.n_epochs):
            for i, (X,y) in enumerate(train_loader):
                step += 1
                xstart_net.train()
                X = X.to(self.config.device)
                y = y.to(self.config.device)
                X = X * (255. / 256.) + (torch.rand_like(X) * 2) / 256.
                if self.config.data.logit_transform is True:
                    X = self.logit_transform(X)
                
                loss = loss_caculator(xstart_net, X, y)
                loss = loss.mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                ema.update()
                
                tb_logger.add_scalar('loss', loss.item(), global_step=step)
                if i % 10 == 0:
                    logging.info("step: {}, loss: {}".format(step, loss.item()))
                
                if step >= self.config.training.n_iters:
                    return 0
                
                if step % 100 == 0:
                    xstart_net.eval()
                    try:
                        test_X, test_y = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X, test_y = next(test_iter)
                        
                    test_X = test_X.to(self.config.device)
                    test_y = test_y.to(self.config.device)
                         
                    with torch.no_grad():
                        test_loss = loss_caculator(xstart_net, test_X, test_y)                        
                        test_loss = test_loss.mean()                        
                    tb_logger.add_scalar('test_loss', test_loss, global_step=step)
                    
                if step % self.config.training.snapshot_freq == 0:
                    # save model checkpoint
                    ema.apply_shadow()
                    model_to_save = xstart_net.module if hasattr(xstart_net, 'module') else xstart_net
                    states = [
                        model_to_save.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))
                    ema.restore()
    
    def test(self):
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        if self.config.model.model_name == 'EDM_AdaGN_Unet_attention':
            xstart_net = EDM_UNet_attention_wrapped(self.config).to(self.config.device)
        else:
            raise NotImplementedError('model {} not understood.'.format(self.config.model.model_name))
        xstart_net.load_state_dict(states[0])

        if not os.path.exists(self.args.image_folder):
            os.makedirs(self.args.image_folder)

        xstart_net.eval()
        
        if self.config.data.dataset == 'CIFAR10':
            if getattr(self.args, 'fid_mode', False):
                self.generate_fid_samples(xstart_net, total_samples=50000, batch_size=128)
            else:
                class_labels = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
                n = len(class_labels)
                z = torch.randn(n, 3, 32, 32, device = self.config.device)
                y = torch.tensor(class_labels, device = self.config.device)
            
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([self.config.data.num_classes] * n, device = self.config.device)
                y = torch.cat([y, y_null], 0)
            
                edm_sampler_GIF(xstart_net, z, y, self.args, self.config,
                                S_churn = 40,
                                S_min = 0.05,
                                S_max = 50,
                                S_noise = 1.003
                                )    
            

    def generate_fid_samples(self, xstart_net, sampler = 'EDM_sampler', total_samples=5000, batch_size=256):
        """
        专门用于生成 FID 评测所需图片的批量生产工厂。
        """
        xstart_net.eval()
        
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
                
                y = torch.randint(0, self.config.data.num_classes, (current_batch_size,), device=self.config.device)
                z = torch.randn(current_batch_size, 3, 32, 32, device=self.config.device) * self.sigmas[0]
                z_2b = torch.cat([z, z], 0)
                y_null = torch.tensor([self.config.data.num_classes] * current_batch_size, device=self.config.device)
                y_2b = torch.cat([y, y_null], 0)
                
                if sampler == 'EDM_sampler':
                    with torch.no_grad():
                        samples_tensor = edm_sampler(xstart_net, z_2b, y_2b,
                                                     cfg_scale = self.config.sampling.cfg_scale,
                                                     S_churn = 40,
                                                     S_min = 0.05,
                                                     S_max = 50,
                                                     S_noise = 1.003)
                else:
                    raise NotImplementedError('sampler {} not understood.'.format(self.config.sampling.sample_way))

                if self.config.data.logit_transform:
                    samples_tensor = torch.sigmoid(samples_tensor)
                else:
                    samples_tensor = (samples_tensor + 1) / 2.0
            
                samples_tensor = torch.clamp(samples_tensor, 0, 1)
                final_sample_cfg, _ = samples_tensor.chunk(2, dim=0)
                im_data = final_sample_cfg.mul(255).add_(0.5).clamp(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
                
                for j in range(current_batch_size):
                    global_idx = samples_generated + j
                    img = Image.fromarray(im_data[j])
                    
                    # 命名格式: 00001_class_3.png
                    filename = f"{global_idx:05d}_class_{y[j].item()}.png"
                    img.save(os.path.join(save_dir, filename), format='PNG')
                
                samples_generated += current_batch_size
                pbar.update(current_batch_size)

        print(f"\n[Success] {samples_generated} images have been saved to {save_dir}.")