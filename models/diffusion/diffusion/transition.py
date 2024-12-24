import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_transition(_type, num_steps, opt):
    if _type == 'continuous':
        return PositionTransition(num_steps, opt)
    elif _type == 'categorical':
        return AminoacidCategoricalTransition(num_steps, opt)
    else:
        raise NotImplementedError(f'transition type {_type} not implemented')

def clampped_one_hot(x, num_classes):
    mask = (x >= 0) & (x < num_classes) # (N, L)
    x = x.clamp(min=0, max=num_classes-1)
    y = F.one_hot(x, num_classes) * mask[...,None]  # (N, L, C)
    return y


class VarianceSchedule(nn.Module):

    def __init__(self, num_steps=100, s=0.01):
        super().__init__()
        T = num_steps
        t = torch.arange(0, num_steps+1, dtype=torch.float)
        f_t = torch.cos( (np.pi / 2) * ((t/T) + s) / (1 + s) ) ** 2
        alpha_bars = f_t / f_t[0]

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.cat([torch.zeros([1]), betas], dim=0)
        betas = betas.clamp_max(0.999)

        sigmas = torch.zeros_like(betas)
        for i in range(1, betas.size(0)):
            sigmas[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas = torch.sqrt(sigmas)

        self.register_buffer('betas', betas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alphas', 1 - betas)
        self.register_buffer('sigmas', sigmas)


class PositionTransition(nn.Module):

    def __init__(self, num_steps, var_sched_opt={}):
        super().__init__()
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    def add_noise(self, p_0, mask_generate, t):
        """
        Args:
            p_0: [N, ...]
            mask_generate: [N]
            batch_ids: [N]
            t: [batch_size]
        """
        expand_shape = [p_0.shape[0]] + [1 for _ in p_0.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        alpha_bar = self.var_sched.alpha_bars[t] # [N]
        # alpha_bar = alpha_bar[batch_ids]  # [N]

        c0 = torch.sqrt(alpha_bar).view(*expand_shape)
        c1 = torch.sqrt(1 - alpha_bar).view(*expand_shape)

        e_rand = torch.randn_like(p_0)  # [N, 14, 3]
        supervise_e_rand = e_rand.clone()
        p_noisy = c0*p_0 + c1*e_rand
        p_noisy = torch.where(mask_generate.expand_as(p_0), p_noisy, p_0)

        return p_noisy, supervise_e_rand

    def denoise(self, p_t, eps_p, mask_generate, t):
        # IMPORTANT:
        #   clampping alpha is to fix the instability issue at the first step (t=T)
        #   it seems like a problem with the ``improved ddpm''.
        expand_shape = [p_t.shape[0]] + [1 for _ in p_t.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        alpha = self.var_sched.alphas[t].clamp_min(
            self.var_sched.alphas[-2]
        )
        alpha_bar = self.var_sched.alpha_bars[t]
        sigma = self.var_sched.sigmas[t].view(*expand_shape)

        c0 = ( 1.0 / torch.sqrt(alpha + 1e-8) ).view(*expand_shape)
        c1 = ( (1 - alpha) / torch.sqrt(1 - alpha_bar + 1e-8) ).view(*expand_shape)

        z = torch.where(
            (t > 1).view(*expand_shape).expand_as(p_t),
            torch.randn_like(p_t),
            torch.zeros_like(p_t),
        )

        p_next = c0 * (p_t - c1 * eps_p) + sigma * z
        p_next = torch.where(mask_generate.expand_as(p_t), p_next, p_t)
        return p_next



class AminoacidCategoricalTransition(nn.Module):
    
    def __init__(self, num_steps, num_classes=20, var_sched_opt={}):
        super().__init__()
        self.num_classes = num_classes
        self.var_sched = VarianceSchedule(num_steps, **var_sched_opt)

    @staticmethod
    def _sample(c):
        """
        Args:
            c:    (N, K).
        Returns:
            x:    (N).
        """
        return torch.multinomial(c, 1)


    def add_noise(self, x_0, mask_generate, t):
        """
        Args:
            x_0:    (N)
            mask_generate:    (N).
            t:  (N,).
        Returns:
            c_t:    Probability, (N, K).
            x_t:    Sample, LongTensor, (N).
        """
        N = x_0.size()
        K = self.num_classes
        c_0 = clampped_one_hot(x_0, num_classes=K).float() # (N, K).
        # alpha_bar = self.var_sched.alpha_bars[t][:, None] # (N, 1)
        alpha_bar = self.var_sched.alpha_bars[t]
        c_noisy = (alpha_bar*c_0) + ( (1-alpha_bar)/K )
        c_t = torch.where(mask_generate[..., None].expand(N, K), c_noisy, c_0)
        x_t = self._sample(c_t)
        return c_t, x_t

    def posterior(self, x_t, x_0, t):
        """
        Args:
            x_t:    Category LongTensor (N) or Probability FloatTensor (N, K).
            x_0:    Category LongTensor (N) or Probability FloatTensor (N, K).
            t:  (N,).
        Returns:
            theta:  Posterior probability at (t-1)-th step, (N, L, K).
        """
        K = self.num_classes

        if x_t.dim() == 2:
            c_t = x_t   # When x_t is probability distribution.
        else:
            c_t = clampped_one_hot(x_t, num_classes=K).float() # (N, K)

        if x_0.dim() == 2:
            c_0 = x_0   # When x_0 is probability distribution.
        else:
            c_0 = clampped_one_hot(x_0, num_classes=K).float() # (N, K)

        alpha = self.var_sched.alpha_bars[t][:, None]     # (N, 1)
        alpha_bar = self.var_sched.alpha_bars[t][:, None] # (N, 1)

        theta = ((alpha*c_t) + (1-alpha)/K) * ((alpha_bar*c_0) + (1-alpha_bar)/K)   # (N, K)
        theta = theta / (theta.sum(dim=-1, keepdim=True) + 1e-8)
        return theta
    
    def denoise(self, x_t, c_0_pred, mask_generate, t):
        """
        Args:
            x_t:        (N).
            c_0_pred:   Normalized probability predicted by networks, (N, K).
            mask_generate:    (N).
            t:  (N,).
        Returns:
            post:   Posterior probability at (t-1)-th step, (N, L, K).
            x_next: Sample at (t-1)-th step, LongTensor, (N, L).
        """
        c_t = clampped_one_hot(x_t, num_classes=self.num_classes).float()  # (N, K)
        post = self.posterior(c_t, c_0_pred, t=t)   # (N, K)
        post = torch.where(mask_generate[..., None].expand(post.size()), post, c_t)
        x_next = self._sample(post)
        return post, x_next
    

class FlowMatchingTransition(nn.Module):

    def __init__(self, num_steps, opt={}):
        super().__init__()
        self.num_steps = num_steps
        # TODO: number of steps T or T + 1
        c1 = torch.arange(0, num_steps + 1).float() / num_steps
        c0 = 1 - c1
        self.register_buffer('c0', c0)
        self.register_buffer('c1', c1)
        self.var_sched = VarianceSchedule(num_steps) # just as relative positional encoding

    def get_timestamp(self, t):
        # use c1 as timestamp
        return self.c1[t]
    
    def add_noise(self, p_0, mask_generate, t):
        """
        Args:
            p_0: [N, ...]
            mask_generate: [N]
            batch_ids: [N]
            t: [batch_size]
        """
        expand_shape = [p_0.shape[0]] + [1 for _ in p_0.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        c0 = self.c0[t].view(*expand_shape)
        c1 = self.c1[t].view(*expand_shape)

        e_rand = torch.randn_like(p_0)  # [N, 14, 3]
        p_noisy = c0*p_0 + c1*e_rand
        p_noisy = torch.where(mask_generate.expand_as(p_0), p_noisy, p_0)

        return p_noisy, (e_rand - p_0)

    def denoise(self, p_t, eps_p, mask_generate, t):
        # IMPORTANT:
        #   clampping alpha is to fix the instability issue at the first step (t=T)
        #   it seems like a problem with the ``improved ddpm''.
        expand_shape = [p_t.shape[0]] + [1 for _ in p_t.shape[1:]]
        mask_generate = mask_generate.view(*expand_shape)

        p_next = p_t - eps_p / self.num_steps
        p_next = torch.where(mask_generate.expand_as(p_t), p_next, p_t)
        return p_next