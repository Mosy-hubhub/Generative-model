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
python main.py --runner AnnealRunner --config anneal.yml --doc cifar10
```

Then the model will be trained according to the configuration files in `configs/anneal.yml`. The log files will be stored in `run/logs/cifar10`, and the tensorboard logs are in `run/tensorboard/cifar10`.

### Sampling

Suppose the log files are stored in `run/logs/cifar10`. We can produce samples to folder `samples` by running

```bash
python main.py --runner AnnealRunner --test -o samples
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
python main.py --runner DiTRunner --test -o samples --doc DiT_cifar10
```

**DiT**

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