import torch
import torch.autograd as autograd

def ssm_baseline(X, scorenet, num_Projection = 1, Random_Projection = 'Gaussian'):
    '''
    slice score matching
    loss fuction:
    L = mean(v.T nabla(s(x;theta) v) + 1/2 . * (v.T s(x;theta))**2
    '''
    X_dup = X.unsqueeze(0).expand(num_Projection, *X.shape).contiguous().view(-1, *X.shape[1:])
    X_dup.requires_grad_(True)
    
    grad = scorenet(X_dup)
    
    Projection_vector = torch.randn(grad.shape)
    Projection_vector /= torch.norm(Projection_vector, dim = (1, 2, 3), keepdim = True)
    
    # score_multi_vec.shape = (num_Projection * batch_size, proj, proj, proj)
    score_multi_vec = grad * Projection_vector
    loss1 = torch.mean(torch.sum(score_multi_vec, dim = (1, 2, 3)) ** 2) * 0.5
    grad_proj = torch.sum(score_multi_vec)
    grad2 = autograd.grad(grad_proj, X_dup, create_graph=True)[0]
    loss2 = torch.mean(grad2 * Projection_vector)
    
    
    return loss1 + loss2
    
    
    
    
    
def ssm_vr():
    '''
    slice score matching with variation reduction
    loss function:
    L = mean(v.T nabla(s(x;theta) v) + 1/2 . * ||s(x;theta)||_2 ** 2
    
    
    
    '''
    

    
    
def ssm_anneal_vr():
    '''
    anneal score matching with variation reduction
    '''