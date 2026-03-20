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
    loss = ((score_pred - score_target) ** 2).view(X.shape[0], -1).mean(dim=-1)
    loss =  1/2. * loss * (used_sigmas.squeeze() ** anneal_power)
    return loss.mean(dim=0)

def cond_score_anneal(scorenet, X, y, t, sigmas, anneal_power=2.):
    '''
    this is denoising score matching with muti-scale noise
    labels: the order of the noise be used in sigmas
    sigmas: a list of noise level
    '''
    used_sigmas = sigmas[t].view(X.shape[0], 1, 1, 1)
    z = torch.randn_like(X)
    X_perturbed = X + z * used_sigmas
    score_pred = scorenet(X_perturbed, t, y)
    score_target = - z / used_sigmas 
    loss = ((score_pred - score_target) ** 2).view(X.shape[0], -1).mean(dim=-1)
    loss =  1/2. * loss * (used_sigmas.squeeze() ** anneal_power)
    return loss.mean(dim=0)



def cond_dsm_anneal(noisenet, X, y, t, sigmas):
    used_sigmas = sigmas[t].view(X.shape[0], 1, 1, 1)
    z = torch.randn_like(X)
    noise = z * used_sigmas
    X_perturbed = (X + noise) / torch.sqrt(1 + used_sigmas ** 2)
    noise_pred = noisenet(X_perturbed, t, y)
    loss = ((noise_pred - z) ** 2).view(X.shape[0], -1).mean(dim=-1)
    return loss.mean(dim=0)
    
    
def cond_vpred_anneal(v_net, X, y, t, sigmas):
    used_sigmas = sigmas[t].view(X.shape[0], 1, 1, 1)
    z = torch.randn_like(X)
    noise = z * used_sigmas
    
    # 1. 依然是安全的输入预处理
    c_in = 1.0 / torch.sqrt(1.0 + used_sigmas ** 2)
    X_input = (X + noise) * c_in
    
    # 2. 让 DiT 吐出原始特征 F (不要逼它直接输出 z 了)
    v_pred = v_net(X_input, t, y)
        
    # 3. 【EDM 魔法全局跳跃连接】
    # 用公式算出两个权重系数
    c_skip_noise = used_sigmas / torch.sqrt(1.0 + used_sigmas ** 2)
    c_out_noise = 1.0 / torch.sqrt(1.0 + used_sigmas ** 2)
    
    # 强行用输入数据 X_input 减去网络的输出 F，得到最终预测的噪声！
    target_v = c_skip_noise * X_input - c_out_noise * z
    
    # 4. 算 Loss (目标依然是 z)
    loss = ((v_pred - target_v) ** 2).view(X.shape[0], -1).mean(dim=-1)
    return loss.mean(dim=0)


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, xstart_net, X, y):
        rnd_normal = torch.randn([X.shape[0], 1, 1, 1], device=X.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(X) * sigma
        D_yn = xstart_net(X + noise, sigma, y)
        loss = weight * ((D_yn - X) ** 2)
        return loss
