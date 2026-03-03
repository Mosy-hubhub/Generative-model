import torch

def dsm_baseline(scorenet, X, sigma = 1):
    '''
    this is denoiseing score matching with single noise scale
    '''
    X_perturb = X + torch.randn_like(X) * sigma
    score_pred = scorenet(X_perturb)
    score_target = - (X_perturb - X) / (sigma ** 2)
    loss = 1/2. * ((score_pred - score_target) ** 2).sum(dim = -1).mean(dim = 0)
    return loss

def dsm_anneal(scorenet, X, sigmas, labels, anneal_power=2.):
    '''
    this is denoising score matching with muti-scale noise
    labels: the order of the noise be used in sigmas
    sigmas: a list of noise level
    '''
    used_sigmas = sigmas[labels].view(X.shape[0], 1, 1, 1)
    X_perturbed = X + torch.randn_like(X) * used_sigmas
    score_pred = scorenet(X_perturbed, labels)
    score_target = - (X_perturbed - X) / (used_sigmas ** 2)
    loss = ((score_pred - score_target) ** 2).view(X.shape[0], -1).sum(dim=-1)
    loss =  1/2. * loss * (used_sigmas.squeeze() ** anneal_power)
    return loss.mean(dim=0)
    