import torch
from typing import Callable, Tuple, Optional
from torch import Tensor


class GAMP:
    def __init__(self,
                 prior: Callable,
                 likelihood: Callable,
                 max_iter: int = 1):
        """
        GAMP class
        """
        self.prior = prior
        self.likelihood = likelihood
        self.max_iter = max_iter
        self.buffer_initialized = False
        self.x_hat_buf = None
        self.tau_x_buf = None
        self.s_buf = None

    def forward(self, y, H_funcs, noise_sigma, x_t,
                a_t,
                b2_t,
                x_0_hat,
                x_prev,
                x_init,
                tau_x_init,
                s_init):

        device = y.device
        dtype = y.dtype

        bs = H_funcs.block_num
        M = H_funcs.N
        N = H_funcs.M

        if not self.buffer_initialized or self.x_hat_buf is None or self.x_hat_buf.shape != (bs, N):
            self.x_hat_buf = torch.zeros(bs, N, device=device, dtype=dtype)
            self.tau_x_buf = torch.ones(bs, N, device=device, dtype=dtype)
            self.s_buf = torch.zeros(bs, M, device=device, dtype=dtype)
            self.buffer_initialized = True

        if x_init is None:
            x_hat = self.x_hat_buf.zero_()
        else:
            x_hat = self.x_hat_buf.copy_(x_init.detach())

        if tau_x_init is None:
            tau_x = self.tau_x_buf.fill_(1.0)
        else:
            tau_x = self.tau_x_buf.copy_(tau_x_init.detach())

        if s_init is None:
            s = self.s_buf.zero_()
        else:
            s = self.s_buf.copy_(s_init.detach())

        if not torch.is_tensor(noise_sigma):
            noise_sigma = torch.tensor(noise_sigma, dtype=dtype, device=device)
        else:
            noise_sigma = noise_sigma.to(device)

        if noise_sigma.dim() == 0:
            noise_sigma = noise_sigma.view(1).expand(bs)
        if noise_sigma.dim() == 1:
            noise_sigma = noise_sigma.view(bs, 1)

        delta0 = noise_sigma ** 2

        with torch.no_grad():
            for iter in range(self.max_iter):
                if iter == 0:
                    tau_p = (H_funcs.H_squared(tau_x.view(1, -1))).view(bs, M)

                p = H_funcs.H(x_hat.view(1, -1)).view(bs, M) - s * tau_p

                z_hat, tau_z = self.likelihood(p, tau_p, y, delta0)
                tau_z = torch.real(tau_z)

                tau_p_clamped = torch.clamp(tau_p, min=1e-25)
                s = (z_hat - p) / tau_p_clamped

                tau_s = (1.0 - tau_z / tau_p_clamped) / tau_p_clamped
                tau_s = torch.real(tau_s)
                tau_s = torch.clamp(tau_s, min=1e-25)

                A2_tau_s = H_funcs.Ht_squared(tau_s.view(1, -1)).view(bs, N)
                tau_r = 1.0 / torch.clamp(A2_tau_s, min=1e-25)

                At_s = H_funcs.Ht(s.view(1, -1)).view(bs, N)
                r = x_hat + tau_r * At_s

                # DMPS
                # x_t_flat = x_t.view(bs, N)
                # diff = r - x_t_flat / a_t
                # nabla_r_xt = diff / (1 * a_t * tau_r + b2_t / a_t)

                # DPS/GDM/MM
                with torch.enable_grad():
                  x_0_hat_flat = x_0_hat.view(bs, N)
                  diff = r.detach() - x_0_hat_flat
                  norm = 0.5 * torch.sum(diff ** 2)
                  # norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, create_graph=False)[0]
                  norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev, create_graph=False, retain_graph=False)[0]
                del norm, diff
                nabla_r_xt = - norm_grad.view(bs, N) / (tau_r + b2_t / (a_t ** 2)) # tau_r + b2_t / (a_t ** 2) 
                del norm_grad

                # call prior
                x_hat_new, tau_x_new, x_hat1 = self.prior(a_t, b2_t, tau_r, x_t, x_0_hat, nabla_r_xt)

                x_hat = x_hat_new.detach() 
                tau_x = tau_x_new.detach()
                s = s.detach()

                tau_p = (H_funcs.H_squared(tau_x.view(1, -1))).view(bs, M)

                del p, z_hat, tau_p_clamped, tau_s, A2_tau_s, At_s, r, nabla_r_xt

        return x_hat, tau_x, s, x_hat1

    @staticmethod
    def awgn_likelihood_cs(p: Tensor, tau_p: Tensor,
                           y: Tensor, noise_power: float = 1.0) -> Tuple[Tensor, Tensor]:
        sigma_w_sq = noise_power
        tau_z = 1.0 / (1.0 / torch.clamp(tau_p, min=1e-25) + 1.0 / sigma_w_sq)
        z_hat = (p / torch.clamp(tau_p, min=1e-25) + y / sigma_w_sq) * tau_z
        return z_hat, tau_z

    @staticmethod
    def sparse_prior3(a_t: Tensor, b2_t: Tensor, tau_r: Tensor, x_t: Tensor, x_0_hat: Tensor, nabla_r_xt: Tensor):
        # 
        b, block_num, M, _ = x_0_hat.shape
        nabla_r_xt1 = nabla_r_xt.view(1, -1)

        # 
        nabla_xt_r = 1 * (a_t * x_0_hat - x_t) / (b2_t) + 1 * nabla_r_xt1.view(1, block_num, M, M) #
        x_hat1 = (x_t + b2_t * 1 * nabla_xt_r) / (a_t)

        x_hat = x_hat1.view(1, -1).view(tau_r.shape)
        tau_x = (b2_t / a_t ** 2) - (b2_t ** 2 / a_t ** 2) / (1 * (a_t ** 2) * tau_r  + b2_t) # 1 * (a_t ** 2) * tau_r + b2_t

        # inflation_factor = 1.0 + 0.5 * b2_t.item()
        # tau_x = tau_x * inflation_factor
        
        tau_x = torch.clamp(tau_x, min=1e-15, max=1e8)

        return x_hat, tau_x, x_hat1
