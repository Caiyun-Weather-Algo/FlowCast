import torch
from einops import rearrange
from src.utils.istropic_noise import generate_isotropic_noise


class ConditionalFlow:
    """Conditional Flow base class."""
    def get_conditional_flow(self, x_0, x_1, t):
        raise NotImplementedError

    def get_conditional_vector_field(self, x_0, x_1):
        raise NotImplementedError


class OTConditionalFlow(ConditionalFlow):
    """Optimal Transport """
    def __init__(self, sigma_min) -> None:
        self.sigma_min = sigma_min

    def get_conditional_flow(self, x_0, x_1, t):
        t = rearrange(t, "b -> b 1 1 1")
        mu_t = t * x_1
        sigma_t = 1 - (1 - self.sigma_min) * t
        x_t = mu_t + sigma_t * x_0
        return x_t

    def get_conditional_vector_field(self, x_0, x_1):
        u_t = x_1 - (1 - self.sigma_min) * x_0
        return u_t
    
    
class DyConditionalFlow(ConditionalFlow):
    """Flow for dynamic system """
    def __init__(self, sigma, sigma_min) -> None:
        self.sigma = sigma
        self.sigma_min = sigma_min

    def get_mu_t(self, x_0, x_1, t):
        t = rearrange(t, "b -> b 1 1 1")
        return (1 - t) * x_0 + t * x_1
    
    def get_sigma_t(self, t):
        t = rearrange(t, "b -> b 1 1 1")
        return torch.sqrt(self.sigma_min ** 2 + self.sigma ** 2 * t * (1 - t))
    
    def get_conditional_flow(self, mu_t, sigma_t):
        noise = torch.randn_like(mu_t)
        x_t = mu_t + sigma_t * noise
        return x_t

    def get_conditional_vector_field(self, x_t, mu_t, mu_dt, t):
        t = rearrange(t, "b -> b 1 1 1")
        part1 = self.sigma ** 2 / 2 * ( 1 - 2 * t)
        part2 = self.sigma_min ** 2 + self.sigma ** 2 * t * (1 - t)
        if (part2 == 0).any():
            u_t = mu_dt
        else:
            u_t = part1 / part2 * (x_t - mu_t) + mu_dt
        return u_t


class StochasticInterpolants:
    """Flow for dynamic system """
    def __init__(self, sigma_coef, path="linear") -> None:
        self.sigma_coef = sigma_coef
        self.path = path

    # def make_path(self, x0, x1, t):
    #     t = rearrange(t, "b -> b 1 1 1")
    #     if self.path == "linear":
    #         alpha_t = 1 -  t
    #         beta_t = t
    #         alpha_t_dot = -1
    #         beta_t_dot = 1
    #     pass
    
    # def make_gamma(self, x0, x1, t):
    #     pass

    def get_It(self, x0, x1, t, isotropic=False):
        t = rearrange(t, "b -> b 1 1 1")
        b, c, h, w = x0.shape
        noise = generate_isotropic_noise(b, c, h, w, isotropic).to(x0.device)
        if self.path == "linear":
            alpha_t = 1 -  t
            beta_t = t
            sigma_t = self.sigma_coef * (1 - t)
            w_t = torch.sqrt(t) * noise

            alpha_t_dot = -1
            beta_t_dot = 1
            sigma_t_dot = -1 * self.sigma_coef

            It = alpha_t * x0 + beta_t * x1 + sigma_t * w_t 
            It_dot = alpha_t_dot * x0 + beta_t_dot * x1 + sigma_t_dot * w_t
            
        elif self.path == "linear_beta2":
            alpha_t = 1 -  t
            beta_t = t ** 2
            sigma_t = self.sigma_coef * (1 - t)
            w_t = torch.sqrt(t) * noise

            alpha_t_dot = -1
            beta_t_dot = 2 * t
            sigma_t_dot = -1 * self.sigma_coef

            It = alpha_t * x0 + beta_t * x1 + sigma_t * w_t 
            It_dot = alpha_t_dot * x0 + beta_t_dot * x1 + sigma_t_dot * w_t

        elif self.path == "linear_bsquared":
            alpha_t = 1 -  t
            beta_t = t
            sigma_t = self.sigma_coef * t * (1 - t)
            w_t = torch.sqrt(t) * noise

            alpha_t_dot = -1
            beta_t_dot = 1
            sigma_t_dot = (1 - 2*t) * self.sigma_coef

            It = alpha_t * x0 + beta_t * x1 + sigma_t * w_t 
            It_dot = alpha_t_dot * x0 + beta_t_dot * x1 + sigma_t_dot * w_t
            del alpha_t, beta_t, sigma_t, w_t, alpha_t_dot, beta_t_dot, sigma_t_dot

        elif self.path == "tri_bsquared":
            alpha_t = torch.cos(torch.pi * t / 2)
            beta_t = torch.sin(torch.pi * t / 2)
            sigma_t = self.sigma_coef * t * (1 - t)
            w_t = torch.sqrt(t) * noise

            alpha_t_dot = - (torch.pi / 2) * torch.sin(torch.pi / 2 * t)
            beta_t_dot = torch.pi / 2 * torch.cos(torch.pi / 2 * t)
            sigma_t_dot = (1 - 2*t) * self.sigma_coef

            It = alpha_t * x0 + beta_t * x1 + sigma_t * w_t 
            It_dot = alpha_t_dot * x0 + beta_t_dot * x1 + sigma_t_dot * w_t

        else:
            raise ValueError(f"Path {self.path} not supported")
        return It, It_dot