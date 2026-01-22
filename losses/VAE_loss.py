import torch
import math

def ELBO(X, VAE_model):
    '''
    loss = E_{x~p(x)}[E_{z~p(z|x)}[-logq(x|z)] + KL(p(z|x)||q(z))]
    '''
    N = X.shape[0]
    latent_dim = VAE_model.encoder.latent_dim
    image_dim = X[0].numel()
    
    epsilon = torch.randn((N, latent_dim), device = X.device)
    output, mean, log_var = VAE_model(X, epsilon)
    KL_divergence = (1 / (2 * N)) * (torch.sum(mean ** 2) 
                                     + torch.sum(torch.exp(log_var)) 
                                     - torch.sum(log_var)) - latent_dim / 2
    loss = (1 / (2 * N)) * torch.sum((X - output) ** 2) 
    + 0.5 * image_dim * torch.log(2 * math.pi)
    + KL_divergence
    
    return loss, KL_divergence