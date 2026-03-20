**VAE**:

### Training

We can train an VAE by running

```bash
python main.py  --config VAE.yml --runner VAERunner --doc VAE_cifar10
```

if we need to continue training before

```bash
python main.py  --config VAE.yml --runner VAERunner --resume_training --doc VAE_cifar10 
```

### Sampling

You can have a competition with gobang model by running

```bash
python main.py --runner VAERunner --test -o samples --doc VAE_cifar10
```

**NCSN**

(based on ncsn by Yang Song)
### Training

The usage of `main.py` is quite self-evident. For example, we can train an NCSN by running

```bash
python main.py --runner AnnealRunner --config anneal.yml --doc NCSN_cifar10
```

if we need to continue training before

```bash
python main.py  --config anneal.yml --runner AnnealRunner --resume_training --doc NCSN_cifar10 
```

Then the model will be trained according to the configuration files in `configs/anneal.yml`. The log files will be stored in `run/logs/cifar10`, and the tensorboard logs are in `run/tensorboard/cifar10`.

### Sampling

Suppose the log files are stored in `run/logs/cifar10`. We can produce samples to folder `samples` by running

```bash
python main.py --runner AnnealRunner --test -o NCSN_samples --doc NCSN_cifar10
```

**DiT**

We can train an VAE by running

```bash
python main.py  --config DiT.yml --runner DiTRunner --doc DiT_cifar10
```

if we need to continue training before

```bash
python main.py  --config DiT.yml --runner DiTRunner --resume_training --doc DiT_cifar10 
```

### Sampling

You can have a competition with gobang model by running

```bash
python main.py --runner DiTRunner --test -o DiT_samples --doc DiT_cifar10
```

**DiT_pixel**

We can train an VAE by running

```bash
python main.py  --config DiT_pixel.yml --runner DiT_pixel_Runner --doc DiT_pixel_cifar10 
```

if we need to continue training before

```bash
python main.py  --config DiT_pixel.yml --runner DiT_pixel_Runner --resume_training --doc DiT_pixel_cifar10 
```

### Sampling

You can have a competition with gobang model by running

```bash
python main.py --runner DiT_pixel_Runner --test -o DiT_pixel_samples --doc DiT_pixel_cifar10 
```

**AdaGN_Unet**
We can train an VAE by running

```bash
python main.py  --config AdaGN_Unet_baseline.yml --runner AdaGN_Unet_Runner --doc AdaGN_Unet_baseline_cifar10 
```

if we need to continue training before

```bash
python main.py  --config AdaGN_Unet_baseline.yml --runner AdaGN_Unet_Runner --resume_training --doc AdaGN_Unet_baseline_cifar10
```

### Sampling

You can have a competition with gobang model by running

```bash
python main.py --runner AdaGN_Unet_Runner --test -o AdaGN_Unet_baseline_samples --doc AdaGN_Unet_baseline_cifar10
```
### fid
```bash
python main.py --runner AdaGN_Unet_Runner --test --fid_mode -o AdaGN_Unet_baseline_samples --doc AdaGN_Unet_baseline_cifar10
```
```bash
python -m pytorch_fid cifar10_real_images_5000 AdaGN_Unet_baseline_samples/AdaGN_Unet_baseline_cifar10_2058_2026-Mar-11-13-28-20/fid_samples --device cuda:0
```

**EDM_UNet_DDPM**
We can train an VAE by running

```bash
python main.py  --config AdaGN_Unet_atte_ddpm.yml --runner Cond_DDPM_Runner --doc AdaGN_Unet_atte_EDM_cifar10 
```

if we need to continue training before

```bash
python main.py  --config AdaGN_Unet_atte_ddpm.yml --runner Cond_DDPM_Runner --resume_training --doc AdaGN_Unet_atte_EDM_cifar10
```

### Sampling

You can have a competition with gobang model by running

```bash
python main.py --runner Cond_DDPM_Runner --test -o AdaGN_Unet_atte_ddpm_samples --doc AdaGN_Unet_atte_EDM_cifar10
```
### fid
```bash
python main.py --runner AdaGN_Unet_Runner --test --fid_mode -o AdaGN_Unet_atte_ddpm_samples --doc AdaGN_Unet_atte_EDM_cifar10
```
```bash
python -m pytorch_fid cifar10_real_images_5000 AdaGN_Unet_atte_ddpm_samples/AdaGN_Unet_atte_EDM_cifar10_2058_2026-Mar-11-13-28-20/fid_samples --device cuda:0
```

