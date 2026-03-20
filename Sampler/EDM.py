import torch
import numpy as np 
import os
import tqdm
from torchvision.utils import make_grid
from PIL import Image

def edm_sampler(
    xstart_net, latents, y, 
    cfg_scale = 0.3, num_steps=18, sigma_min=0.002, sigma_max=80,
    rho=7, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    #t_steps = torch.cat([xstart_net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        # t_hat = xstart_net.round_sigma(t_cur + gamma * t_cur)
        t_hat = t_cur + gamma * t_cur
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = xstart_net.forward_with_cfg(x_hat, t_hat, y, cfg_scale).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = xstart_net.forward_with_cfg(x_next, t_next, y, cfg_scale).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def edm_sampler_GIF(
    xstart_net, latents, y, args, config, 
    cfg_scale=0.3, num_steps=18, sigma_min=0.002, sigma_max=80,
    rho=7, S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):

    gif_frames = []
    grid_size = int((latents.shape[0] / 2) ** 0.5) 
    save_dir = os.path.join(args.image_folder, args.doc)
    os.makedirs(save_dir, exist_ok=True)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    x_next = latents.to(torch.float64) * t_steps[0]
    
    for i, (t_cur, t_next) in enumerate(tqdm.tqdm(zip(t_steps[:-1], t_steps[1:]), total=num_steps, desc='EDM GIF Sampling')): 
        x_cur = x_next

        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur
        
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)
        denoised = xstart_net.forward_with_cfg(x_hat, t_hat, y, cfg_scale).to(torch.float64)
        
        # --- GIF 帧收集逻辑 (抓取 denoised) ---
        # 把预测的干净图转回 float32 用于可视化
        sample_to_show = denoised.clone().float() 
        if config.data.logit_transform:
            sample_to_show = torch.sigmoid(sample_to_show)
        sample_to_show = torch.clamp((sample_to_show + 1)/2, 0, 1)
        
        # 切片，只保留带有条件引导 (CFG) 的前半部分图像来画网格
        sample_to_show_cfg, _ = sample_to_show.chunk(2, dim=0)
        grid_cfg = make_grid(sample_to_show_cfg, nrow=grid_size)
        im_data_cfg = grid_cfg.mul(255).add_(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        gif_frames.append(Image.fromarray(im_data_cfg))
        # --------------------------------------

        # 继续计算 ODE 步进
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = xstart_net.forward_with_cfg(x_next, t_next, y, cfg_scale).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    if gif_frames:
        gif_path = os.path.join(save_dir, 'edm_diffusion_process_cfg.gif')
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=150,
            loop=0
        )
        print(f"\n[Saved] EDM GIF saved to {gif_path}")
        

    return x_next