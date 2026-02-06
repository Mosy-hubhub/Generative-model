# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py


import math

import numpy as np
import torch as th
import enum

from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.
    Original ported from this codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type
    ):
        # for example: self.model_mean_type = ModelMeanType.START_X
        # Which type of output the model predicts
        self.model_mean_type = model_mean_type
        
        # What is used as the model's output variance
        self.model_var_type = model_var_type
        
        # MSE, KL, ...
        self.loss_type = loss_type

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        # forward SDE for DDPM: x_t = x_{t-1} + alpha s(x_{t-1}) + sqrt(2 * alpha) W
        alphas = 1.0 - betas
        # self.alphas_cumprod.shape = (self.num_timesteps,)
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = x_start * _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)   
        variance = _extract_into_tensor(1 - self.alphas_cumprod, t, x_start.shape)   
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return (mean, variance, log_variance)
        

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.rand_like(x_start)
        assert noise.shape == x_start.shape
        noise_version = (x_start * _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
                         + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
            
        return noise_version

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
            
            by Bayesian: q(x_{t-1} | x_t, x_0) = q(x_{t-1} | x_0)q(x_t | x_{t-1}, x_0) / q(x_t | x_0)
                                               = q(x_{t-1} | x_0)q(x_t | x_{t-1}) / q(x_t | x_0)
            Gaussian plus Gaussian is still Gaussian
        """
        assert x_start.shape == x_t.shape
        mean = (x_t * _extract_into_tensor(self.posterior_mean_coef2, t, x_start.shape) +
                x_start * _extract_into_tensor(self.posterior_mean_coef2, t, x_start.shape))
        
        variance = _extract_into_tensor(self.posterior_variance, t, x_start.shape)
        
        log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_start.shape)
        
        return (mean, variance, log_variance_clipped)
        
        
    def p_mean_variance(self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        # B batch_size, C num_channel
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, t, **model_kwargs)
        if isinstance(model_output, tuple):
            model_output, extra = model_output
        else:
            extra = None
        
        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised is True:
                x = th.clip(x, -1, 1)
        
        if self.model_mean_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_mean, model_log_var = th.split(model_output, C, dim=1)
            if self.model_mean_type == ModelVarType.LEARNED:
                log_variance = model_log_var
            else:
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                frac = (1 + model_log_var) / 2
                log_variance = frac * min_log + (1 - frac) * max_log
            variance = th.exp(log_variance)
        else:
            log_variance, variance = {ModelVarType.FIXED_SMALL:(self.posterior_log_variance_clipped,
                                                                       np.exp(self.posterior_log_variance_clipped)),
                                     ModelVarType.FIXED_LARGE:(np.append(self.posterior_log_variance_clipped[:1],
                                                                         self.betas[1:]), self.betas)}[self.model_mean_type]
            variance = _extract_into_tensor(variance, t, x.shape)
            log_variance = _extract_into_tensor(log_variance, t, x.shape)
        
        
        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = process_xstart(model_output)
        else:
            pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
            pred_xstart = process_xstart(pred_xstart)
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)


        assert model_mean.shape == log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": variance,
            "log_variance": log_variance,
            "pred_xstart": pred_xstart,
            "extra": extra,
        }
        
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        pred_xstart = (x_t * _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) 
                       - eps * _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))
        return pred_xstart

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        assert x_t.shape == pred_xstart.shape
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return eps
    
    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, model_kwargs)
        mean = p_mean_var['mean'].float() + p_mean_var['var'] * gradient.float()
        return mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        gradient = cond_fn(x, t, model_kwargs)
        eps = self._predict_eps_from_xstart(x, t, p_mean_var['pred_xstart'])
        eps = eps - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * gradient
        
        pred_xstart = self._predict_xstart_from_eps(x, t, eps)
        
        out = p_mean_var.copy()
        out['pred_xstart'] = pred_xstart
        out['mean'], _, _ = self.q_posterior_mean_variance(out['pred_xstart'], x, t)
        
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = {}
        model_output = self.p_mean_variance(model, x, t,
                                            clip_denoised = clip_denoised,
                                            denoised_fn = denoised_fn,
                                            model_kwargs = model_kwargs)
        out['pred_xstart'] = model_output['pred_xstart']
        eps = th.randn_like(x)
        assert isinstance(t, (np.ndarray, th.Tensor))
        if isinstance(t, np.ndarray):
            t = th.from_numpy(t)
        mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))                           
        if cond_fn is not None:
            model_output['mean'] = self.condition_mean(cond_fn, model_output, x, t, model_kwargs)
        out['sample'] = eps * th.exp(0.5 * model_output['log_variance']) * mask + model_output['mean']
        return out


    
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.
        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for out in self.p_sample_loop_progressive(model, shape, noise,
                                                  clip_denoised = clip_denoised,
                                                  denoised_fn = denoised_fn,
                                                  cond_fn = cond_fn,
                                                  model_kwargs = model_kwargs,
                                                  device = device,
                                                  progress = progress):
            final = out
        return final['sample']

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        assert isinstance(shape, (tuple, list))
        if noise is None:
            img = th.randn_like(shape)
        else:
            img = noise
        assert img.shape == shape
        if device is None:
            device = next(model.parameters()).device
            
        img = img.to(device = device)
            
        time_list = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            time_list = tqdm(time_list)
            
        for t in time_list:
            with th.no_grad():
                time = th.ones((shape[0],), device = device) * t
                out = self.p_sample(model, img, time, 
                                    clip_denoised = clip_denoised,
                                    denoised_fn = denoised_fn,
                                    cond_fn = cond_fn,
                                    model_kwargs = model_kwargs)
                yield out
                img = out['sample']

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        
        sigma_t = eta * sqrt[(1 - alpha_{t-1} / 1 - alpha_t) * (1 - alpha_t/alpha_{t-1})]
        when eta = 0, deterministic procedure, pure DDIM
        when eta = 1, DDPM, markov
        """
        # get x_0
        out = {}
        model_output = self.p_mean_variance(model, x, t,
                                            clip_denoised = clip_denoised,
                                            denoised_fn = denoised_fn,
                                            model_kwargs = model_kwargs)
        if cond_fn is not None:
            model_output = self.condition_score(cond_fn, model_output, x, t, model_kwargs=model_kwargs)
        out['pred_xstart'] = model_output['pred_xstart']
        
        # get x_{t-1}(sample)
        eps = th.randn_like(x)
        alpha_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        alpha = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        sigma = (eta * th.sqrt(_extract_into_tensor(self.betas, t, x.shape)) 
                 * th.sqrt((1 - alpha_prev) / (1 - alpha))
                 )
        
        mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))     
        out['sample'] = (alpha_prev * out['pred_xstart'] 
                         + th.sqrt(1 - alpha_prev - sigma ** 2) * self._predict_eps_from_xstart(x, t, out['pred_xstart'])
                         + sigma * eps * mask)
        return out
    
    
    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        
        for DDIM, x_{t-1} = sqrt(alpha_{t-1}) x_0 + sqrt(1 - alpha_{t-1}) epsilon_theta(x_t, t)
        
        then x_t = sqrt(alpha_t) x_0 + sqrt(1 - alpha_t) epsilon_theta(x_{t-1}, t-1)
        x_0 here is which model predicts because we want keep the same as reverse process
        """
        assert eta == 0, "Reverse ODE only for deterministic path"
        output = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            output = self.condition_score(cond_fn, output, x, t, model_kwargs = model_kwargs)
            
        out = {}
        out['pred_xstart'] = output['pred_xstart']
        
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)
        out['sample'] = alpha_bar_next * out['pred_xstart'] + th.sqrt(1 - alpha_bar_next) * self._predict_eps_from_xstart(x, t)
        
        return out
    
    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.
        Same usage as p_sample_loop().
        """
        final = None
        for out in self.ddim_sample_loop_progressive(model, shape, 
                                                     noise = noise,
                                                     clip_denoised = clip_denoised,
                                                     denoised_fn = denoised_fn,
                                                     cond_fn = cond_fn,
                                                     model_kwargs = model_kwargs,
                                                     device = device,
                                                     progress = progress,
                                                     eta = eta):
            final = out
        return final['sample']

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.
        Same usage as p_sample_loop_progressive().
        """
        assert isinstance(shape, (tuple, list))
        if noise is None:
            img = th.randn_like(shape)
        else:
            img = noise
        assert img.shape == shape
        if device is None:
            device = next(model.parameters()).device
            
        img = img.to(device = device)
            
        time_list = list(range(self.num_timesteps))[::-1]
        if progress:
            from tqdm.auto import tqdm
            time_list = tqdm(time_list)
            
        for t in time_list:
            with th.no_grad():
                time = th.ones((shape[0],), device = device) * t
                out = self.ddim_sample(model, img, time, 
                                       clip_denoised = clip_denoised,
                                       denoised_fn = denoised_fn,
                                       cond_fn = cond_fn,
                                       model_kwargs = model_kwargs,
                                       eta = eta)
                yield out
                img = out['sample']

    

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        out = self.p_mean_variance(model, x_t, t,
                                   clip_denoised = clip_denoised,
                                   model_kwargs = model_kwargs)
        
        true_mean, _, true_log_var = self.q_mean_variance(x_start, t)
        
        kl = normal_kl(true_mean, true_log_var, out['mean'], out['log_variance'])
        kl = mean_flat(kl) / np.log(2.0)
        
        decoder_nll = discretized_gaussian_log_likelihood(x_t, means = out['mean'],
                                                          log_scales = 0.5 * out["log_variance"])
        
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
        
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}
    
    

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs ={}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise = noise)
        
        terms = {}
        
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                model_output = model(x_t, t, **model_kwargs)
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, 2 * C, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                        model=lambda *args, r=frozen_out: r,
                        x_start=x_start,
                        x_t=x_t,
                        t=t,
                        clip_denoised=False,
                    )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # In official version he uses:
                    # terms["vb"] *= self.num_timesteps / 1000.0
                    # but it will detach the computation graph. Is there a bug?
                    terms['vb'] = terms['vb'] * self.num_timesteps / 1000.0
             
            target = {ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start, x_t, t)[0],
                      ModelMeanType.EPSILON: noise,
                      ModelMeanType.START_X: x_start
                      }[self.model_mean_type]
            
            assert model_output.shape == target.shape == x_start.shape
            terms['mse'] = mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms['loss'] = terms['mse'] + terms['vb']
            else:
                terms['loss'] = terms['mse']
            
        else:
            raise NotImplementedError(self.loss_type)
        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        N = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * N, device = x_start.device)
        qt_mean, _, qt_log_var = self.q_mean_variance(x_start, t)
        KL = normal_kl(qt_mean, qt_log_var, 0.0, 0.0)
        return mean_flat(KL) / np.log(2.0)
    
    
    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        N = x_start.shape[0]
        prior_bpd = self._prior_bpd(x_start)
        vb = []
        xstart_mse = []
        mse = []
        
        time_list = list(range(self.num_timesteps))[::-1]
        for t in time_list:
            t_list = [t] * N
            noise = th.rand_like(x_start)
            x_t = self.q_sample(x_start, t_list, noise)
            with th.no_grad():
                out = self._vb_terms_bpd(model, x_start, x_t, t, clip_denoised, model_kwargs)
            vb.append(out['output'])
            xstart_mse.append(mean_flat((out['pred_xstart'] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t, out['pred_xstart'])
            mse.append(mean_flat((eps - noise) ** 2))
            
        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)
            
        return {'total_bpd': prior_bpd + vb.sum(dim = 1),
                'prior_bpd': prior_bpd,
                'vb': vb,
                'xstart_mse': xstart_mse,
                'mse': mse
                }

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)
