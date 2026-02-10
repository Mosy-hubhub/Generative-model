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