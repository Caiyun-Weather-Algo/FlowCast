import torch
from contextlib import contextmanager, nullcontext
import pandas as pd
from src.utils.istropic_noise import generate_isotropic_noise


class PartialSolver:
    def __init__(self, model, ema=None, sigma_var=0.1, noise_scale=1.0):
        self.model = model
        self.ema = ema
        self.sigma_t = lambda t: (1. - t) * sigma_var
        self.noise_scale = noise_scale
        
    @contextmanager
    def maybe_ema(self):
        ema = self.ema
        if ema is None:
            with nullcontext():
                yield
        else:
            with ema.average_parameters():
                yield

    @torch.no_grad()
    def sampling(self, x0=None, N=1, s=None, cond=None, sampler="euler", dataset=None, autoregressive_step=6, previous_input=None):
        """
        Sample from the ODE or SDE depending on the selected sampler.
        Supports Euler, RK45, and SDE methods.
        """
        with self.maybe_ema():
            if sampler == "euler":            
                x = self.euler_ode(x0, N, s, dataset=dataset)
            
            elif sampler == "euler_nonorm":            
                x = self.euler_ode_nonorm(x0, N, s, dataset=dataset)
            
            elif sampler == "euler_nonorm_addinput":            
                x = self.euler_ode_nonorm_addinput(x0, N, s, dataset=dataset, previous_input=previous_input)
            
            elif sampler == "euler_nonorm_cond":            
                x = self.euler_ode_nonorm_cond(x0, N, s, dataset=dataset)
          
            elif sampler == "euler_nonorm_6hr":            
                x = self.euler_ode_nonorm_6hr(x0, N, s, dataset=dataset, autoregressive_step=autoregressive_step)  

            elif sampler == "euler_sde_nonorm":
                x = self.euler_sde_nonorm(x0, N, s, dataset=dataset)
            
            elif sampler == "euler_sde_nonorm_cond":
                x = self.euler_sde_nonorm_cond(x0, N, s, dataset=dataset)

            elif sampler == "euler_raw":            
                x = self.euler_ode_raw(x0, N, s, dataset=dataset)

            elif sampler == "euler_scale":            
                x = self.euler_ode_scale(x0, N, s, dataset=dataset)

            elif sampler == "euler_cond":            
                x = self.euler_ode_cond(x0, N, s, cond=cond)
                
            elif sampler == 'heun_2order':
                x = self.heun_2order_ode(x0, N, s, dataset=dataset)
            
            elif sampler == 'heun_2order_nonorm':
                x = self.heun_2order_nonorm(x0, N, s, dataset=dataset)

            elif sampler == "rk45":
                # TODO: Implement the RK45 solver (currently placeholder)
                x = self.rk45_ode(x0, N)

            elif sampler == "sde":
                # Use Euler-Maruyama method for SDE
                x = self.sde_solver(x0, N, s, dataset=dataset)

            else:
                x = x0
        return x

    @torch.no_grad()
    def euler_ode(self, x0, N, s=None, dataset=None):
        """
         Use Euler method to sample from the learned flow
        """  
        dt = 1. / N
        x = x0.detach().clone()
        B = x.shape[0]
        for i in range(N):
            input = dataset.normalize(x)
            t = torch.ones((B), device=x.device) * i / N * 1000
            if s is not None:
                pred = self.model.forward(input, t, s)
            else:
                pred = self.model.forward(input, t)

            pred = dataset.normalize_diff(pred, reverse=True)
            x = x.detach().clone() + pred * dt
        return x 

    @torch.no_grad()
    def euler_ode_raw(self, x0, N, s=None, dataset=None):
        """
         Use Euler method to sample from the learned flow
        """   
        dt = 1. / N
        x = x0.detach().clone()
        B = x.shape[0]
        for i in range(N):
            t = torch.ones((B), device=x.device) * i / N
            if s is not None:
                pred = self.model.forward(x, t*1000, s)
            else:
                pred = self.model.forward(x, t*1000) 
            x = x.detach().clone() + (pred - x0.detach()) * dt
        return x 

    @torch.no_grad()
    def euler_ode_nonorm(self, x0, N, s=None, dataset=None):
        """
         Use Euler method to sample from the learned flow
        """   
        dt = 1. / N
        x = x0.detach().clone()
        x = dataset.normalize(x)
        B = x.shape[0]
        for i in range(N):
            t = torch.ones((B), device=x.device) * i / N
            if s is not None:
                pred = self.model.forward(x, t*1000, s)
            else:
                pred = self.model.ç(x, t*1000)
            x = x.detach().clone() + pred * dt
        x = dataset.normalize(x, reverse=True, data_pack=True)
        return x 

    @torch.no_grad()
    def euler_ode_nonorm_addinput(self, x0, N, s=None, dataset=None, previous_input=None):
        """
         Use Euler method to sample from the learned flow
        """   
        dt = 1. / N
        x = x0.detach().clone()
        x = dataset.normalize(x)
        previous_input = dataset.normalize(previous_input)
        B = x.shape[0]
        for i in range(N):
            t = torch.ones((B), device=x.device) * i / N
            if s is not None:
                pred = self.model.forward(x, t*1000, s, previous_input)
            else:
                pred = self.model.ç(x, t*1000, previous_input)
            x = x.detach().clone() + pred * dt
        x = dataset.normalize(x, reverse=True, data_pack=True)
        return x 

    @torch.no_grad()
    def euler_ode_nonorm_cond(self, x0, N, s=None, dataset=None):
        """
         Use Euler method to sample from the learned flow
        """   
        dt = 1. / N
        x = x0.detach().clone()
        x = dataset.normalize(x)
        cond = x.clone()
        B = x.shape[0]
        for i in range(N):
            t = torch.ones((B), device=x.device) * i / N
            if s is not None:
                pred = self.model.forward(x, t*1000, s, cond)
            else:
                pred = self.model(x, t*1000, cond)
            x = x.detach().clone() + pred * dt
        x = dataset.normalize(x, reverse=True, data_pack=True)
        return x 
    
    @torch.no_grad()
    def euler_ode_nonorm_6hr(self, x0, N, s=None, dataset=None, autoregressive_step=6):
        """
         Use Euler method to sample from the learned flow
        """  
        total_autoregressive_step = 6 
        dt = 1. / (N * total_autoregressive_step)
        x = x0.detach().clone()
        x = dataset.normalize(x)
        B = x.shape[0]
        for i in range(N):
            t = torch.ones((B), device=x.device) * (i + autoregressive_step * N) / (N * total_autoregressive_step)
            if s is not None:
                pred = self.model.forward(x, t*1000, s)
            else:
                pred = self.model.forward(x, t*1000)
            x = x.detach().clone() + pred * dt
        x = dataset.normalize(x, reverse=True, data_pack=True)
        return x 
    
    @torch.no_grad()
    def euler_ode_scale(self, x0, N, s=None, dataset=None):
        """
         Use Euler method to sample from the learned flow
        """   
        flow_coefs = pd.read_csv('./data/ERA5_GLOBAL/flow_coefs.csv', index_col=0).values.reshape(-1)
        flow_coefs = torch.tensor(flow_coefs)[None, :, None, None]

        dt = 1. / N
        x = x0.detach().clone()
        x = dataset.normalize(x)
        B = x.shape[0]
        for i in range(N):
            t = torch.ones((B), device=x.device) * i / N
            pred = self.model.forward(x, t*1000, s)
            pred = pred * flow_coefs.to(x)
            x = x.detach().clone() + pred * dt
        x = dataset.normalize(x, reverse=True, data_pack=True)
        return x 
    
    @torch.no_grad()
    def euler_ode_cond(self, x0, N, s=None, cond=None):
        """
         Use Euler method to sample from the learned flow
        """   
        dt = 1. / N
        x = x0.detach().clone()
        B = x.shape[0]
        for i in range(N):
            t = torch.ones((B), device=x.device) * i / N
            pred = self.model.forward(x, t, s, cond)            
            x = x.detach().clone() + pred * dt
        return x 
                
    @torch.no_grad()
    def rk45_ode(self, x0, N):
        """
        Placeholder for RK45 solver.
        Implement RK45 or other adaptive step-size ODE solver here.
        """
        x = x0  # TODO: Implement RK45 method
        return x
    
    @torch.no_grad()
    def heun_2order_ode(self, x0, N, s=None, dataset=None):
        """
         Use Euler method to sample from the learned flow
        """   
        dt = 1. / N
        x = x0.detach().clone()
        B = x.shape[0]
        for i in range(N):
            input = dataset.normalize(x.clone())
            t1 = torch.ones((B), device=x.device) * i / N
            if s is not None:
                pred1 = self.model.forward(input, t1*1000, s)
            else:
                pred1 = self.model.forward(input, t1*1000)
            
            pred1 = dataset.normalize_diff(pred1, reverse=True)
            x1 = x.detach().clone() + pred1 * dt
            
            # correction step 
            input = dataset.normalize(x1.clone())
            t2 = torch.ones((B), device=x1.device) * (i + 1) / N
            if s is not None:
                pred2 = self.model.forward(input, t2*1000, s)
            else:
                pred2 = self.model.forward(input, t2*1000)
            
            pred2 = dataset.normalize_diff(pred2, reverse=True)
            x = x.detach().clone() + (pred1  + pred2) / 2 * dt
        return x 
    
    @torch.no_grad()
    def heun_2order_nonorm(self, x0, N, s=None, dataset=None):
        """
         Use Euler method to sample from the learned flow
        """   
        dt = 1. / N
        x = x0.detach().clone()
        x = dataset.normalize(x)
        B = x.shape[0]
        for i in range(N):
            t1 = torch.ones((B), device=x.device) * i / N
            pred1 = self.model.forward(x, t1*1000, s)
            x1 = x + pred1 * dt
            
            # correction step 
            t2 = torch.ones((B), device=x1.device) * (i + 1) / N
            pred2 = self.model.forward(x1, t2*1000, s)
            x = x + (pred1  + pred2) / 2 * dt

        x = dataset.normalize(x, reverse=True, data_pack=True)
        return x 
    
    @torch.no_grad()
    def sde_solver(self, x0, N, s=None, dataset=None, correct_term=False):
        """
        Euler-Maruyama method for solving SDEs.
        Adds noise term to simulate the stochastic differential process.
        """
        dt = 1. / N
        x = x0.detach().clone()
        B, *shape = x0.shape
        for i in range(N):
            t = torch.ones((B,), device=x.device) * i / N
            num_t = torch.tensor((i / N), device=x.device)
            sigma_t = self.sigma_t(num_t).to(x.device) 
            
            # prediction
            input = dataset.normalize(x.clone())
            if s is not None:
                pred = self.model.forward(input, t, s)
            else:
                pred = self.model.forward(input, t)
            pred = dataset.normalize_diff(pred, reverse=True)

            # perserving the marginal probability
            if correct_term:
                correction_term = (sigma_t**2) / (2 * (self.noise_scale**2) * ((1. - num_t)**2)) \
                * (0.5 * num_t * (1. - num_t) * pred - 0.5 * (2. - num_t) * x.detach().clone())
                pred = pred + correction_term
        
            # Adding stochastic noise
            channel_std = torch.std(pred, dim=(0, 2, 3), keepdim=True)
            noise = torch.randn_like(x) * (dt ** 0.5) * channel_std
            x = x.detach().clone() + pred * dt + sigma_t * noise
        return x

    @torch.no_grad()
    def euler_sde_nonorm(self, x0, N, s=None, dataset=None):
        """
         Use Euler method to sample from the learned flow
        """   
        dt = 1. / N
        x = x0.detach().clone()
        x = dataset.normalize(x)
        B = x.shape[0]
        for i in range(N):
            t = torch.ones((B), device=x.device) * i / N
            if s is not None:
                pred = self.model.forward(x, t*1000, s)
            else:
                pred = self.model.forward(x, t*1000)
            noise = torch.randn_like(x)
          #  sigma_t = 1 - t
            sigma_t = t * (1 - t)  #todo 
            sigma_t = sigma_t.view(B,1,1,1)
            x = x.detach().clone() + pred * dt + dt ** 0.5 * sigma_t * noise
        x = dataset.normalize(x, reverse=True, data_pack=True)
        return x 

    @torch.no_grad()
    def euler_sde_nonorm_cond(self, x0, N, s=None, dataset=None):
        """
         Use Euler method to sample from the learned flow
        """   
        dt = 1. / N
        x = x0.detach().clone()
        x = dataset.normalize(x)
        cond = x
        B, C, H, W = x.shape
        for i in range(N):
            t = torch.ones((B), device=x.device) * i / N
            if s is not None:
                pred = self.model.forward(x, t*1000, s, cond)
            else:
                pred = self.model.forward(x, t*1000, cond)
            noise = generate_isotropic_noise(B, C, H, W, isotropic=True).to(x0.device)

          #  sigma_t = 1 - t
            sigma_t = t * (1 - t)  #todo 
            sigma_t = sigma_t.view(B,1,1,1)
            x = x.detach().clone() + pred * dt + dt ** 0.5 * sigma_t * noise
        x = dataset.normalize(x, reverse=True, data_pack=True)
        return x 