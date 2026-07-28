import math
import os
from functools import partial
from xml.parsers.expat import model
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from util.img_utils import clear_color
from .posterior_mean_variance import get_mean_processor, get_var_processor

from GAMP import GAMP

__SAMPLER__ = {}


def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __SAMPLER__[name] = cls
        return cls

    return wrapper


def get_sampler(name: str):
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def create_sampler(sampler,
                   steps,
                   noise_schedule,
                   model_mean_type,
                   model_var_type,
                   dynamic_threshold,
                   clip_denoised,
                   rescale_timesteps,
                   timestep_respacing=""):
    sampler = get_sampler(name=sampler)

    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]

    return sampler(use_timesteps=space_timesteps(steps, timestep_respacing),
                   betas=betas,
                   model_mean_type=model_mean_type,
                   model_var_type=model_var_type,
                   dynamic_threshold=dynamic_threshold,
                   clip_denoised=clip_denoised,
                   rescale_timesteps=rescale_timesteps)


class GaussianDiffusion:
    def __init__(self,
                 betas,
                 model_mean_type,
                 model_var_type,
                 dynamic_threshold,
                 clip_denoised,
                 rescale_timesteps
                 ):

        # use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (self.betas <= 1).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0])
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
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
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

        self.mean_processor = get_mean_processor(model_mean_type,
                                                 betas=betas,
                                                 dynamic_threshold=dynamic_threshold,
                                                 clip_denoised=clip_denoised)

        self.var_processor = get_var_processor(model_var_type,
                                               betas=betas)
        ###
        self.lambda_t = 0.5 * np.log(self.alphas_cumprod / (1.0 - self.alphas_cumprod))
        denominator = np.clip(1.0 - self.alphas_cumprod_prev, 1e-8, 1.0)
        self.lambda_next = 0.5 * np.log(self.alphas_cumprod_prev / denominator)
        self.old_x_0_list = []
        self.old_x_0_listhat = []

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """

        mean = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start) * x_start
        variance = extract_and_expand(1.0 - self.alphas_cumprod, t, x_start)
        log_variance = extract_and_expand(self.log_one_minus_alphas_cumprod, t, x_start)

        return mean, variance, log_variance

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return coef1 * x_start + coef2 * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        coef2 = extract_and_expand(self.posterior_mean_coef2, t, x_t)
        posterior_mean = coef1 * x_start + coef2 * x_t
        posterior_variance = extract_and_expand(self.posterior_variance, t, x_t)
        posterior_log_variance_clipped = extract_and_expand(self.posterior_log_variance_clipped, t, x_t)

        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_sample_loop(self,
                      model,
                      x_start,
                      measurement,
                      measurement_cond_fn,
                      record,
                      save_root):
        """
        The function used for sampling from noise.
        """
        img = x_start
        device = x_start.device

        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=device)

            img = img.requires_grad_()
            out = self.p_sample(x=img, t=time, model=model)

            # Give condition.
            noisy_measurement = self.q_sample(measurement, t=time)

            # TODO: how can we handle argument for different condition method?
            img, distance = measurement_cond_fn(x_t=out['sample'],
                                                measurement=measurement,
                                                noisy_measurement=noisy_measurement,
                                                x_prev=img,
                                                x_0_hat=out['pred_xstart'],
                                                a_t=self.sqrt_alphas_cumprod[time],
                                                b2_t=1 - self.alphas_cumprod[time])
            img = img.detach_()

            pbar.set_postfix({'distance': distance.item()}, refresh=False)
            if record:
                if idx % 10 == 0:
                    file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))

        return img

    def _step_mmps(self,
                   model,
                   x_start,
                   measurement,
                   H_funcs,
                   noise_std,
                   record,
                   save_root,
                   alg_name,
                   diffusion_sampler,
                   obs_module=None):
        """
        The function used for sampling from noise.
        """
        if obs_module is not None:
            raise ValueError(
                "MMPS/PGDM/DPS require a differentiable measurement model, "
                "but a non-differentiable observation (obs_module) was provided. "
                "Use a GAMP-based algorithm instead (e.g., gamp_mm, gamp_ga, gamp-ga)."
            )
        img = x_start
        device = x_start.device

        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=device)

            img = img.requires_grad_()
            out = self.p_sample(x=img, t=time, model=model)
            x_0_hat = out['pred_xstart']
            x_t = out['sample']

            # TODO: how can we handle argument for different condition method?
            noise_sigma2 = noise_std ** 2
            b2_t = (1 - self.alphas_cumprod[time])
            a_t = self.sqrt_alphas_cumprod[time]
            if alg_name == 'mmps':
                sigma_t = b2_t / (a_t ** 1)  # MMPS

            else:
                sigma_t = b2_t  # PGDM
            n_iters = 1  # GDPM

            pbar.set_postfix({'sigma_t': sigma_t.item()}, refresh=False)

            bs = 1  # H_funcs.block_num
            N = H_funcs.M
            M = -1  # H_funcs.N
            y = measurement.view(bs, M)

            if alg_name == 'dps':
                # DPS like
                # norm_grad = score_y_given_x
                # img = x_t + norm_grad * 0.003
                ##  GDMP noise 0.05 best 0.03 noise 0.5 best 0.01   MMPS noise 0.05 best 0.003

                # DPS
                value = torch.tensor(H_funcs.block_num * int(H_funcs.N // 3), dtype=torch.float32)
                input_size = int(torch.sqrt(value).item())
                difference = measurement.view(1, 3, input_size, input_size) - H_funcs.forward(x_0_hat.view(1, -1))
                norm = torch.linalg.norm(difference)
                norm_grad = torch.autograd.grad(outputs=norm, inputs=img)[0]
                img = x_t - norm_grad * 2  # noise 0.05 best 2

            else:
                # PGDM/MMPS
                # Compute b = y - A * E[x|x_t]
                Ax0 = H_funcs.H(x_0_hat.view(1, -1))  # (bs, M)
                b = y - Ax0.view(bs, M)  # (bs, M)

                def M_product(v):
                    # Compute A^T * v
                    At_v = H_funcs.Ht(v.view(1, -1))

                    # Compute VJP: (grad_outputs^T * J)^T -> J^T * grad_outputs
                    vjp_input = (sigma_t) * At_v  # (1,-1)  # 0.1

                    if alg_name == 'mmps':
                        # MMPS  asymmetric
                        vjp_input_for_grad = vjp_input.view_as(img)
                        vjp = torch.autograd.grad(
                            outputs=x_0_hat,
                            inputs=img,
                            grad_outputs=vjp_input_for_grad,
                            retain_graph=True  #
                        )[0]
                    else:
                        # PGDM
                        vjp = vjp_input

                    # Obtain M*v = \Sigma_y * v + A * vjp
                    Avjp = H_funcs.H(vjp.view(1, -1)).view(bs, M)
                    return noise_sigma2 * v + Avjp

                # v = torch.zeros_like(b)
                # r = b # - M_product(v)
                # p = r.clone()
                # # CG iteration (bs, N) processing
                # for _ in range(n_iters):
                #     Mp = M_product(p)
                #     # Compute alpha = (r^T * r) / (p^T * M * p)
                #     r_dot_r = torch.sum(r * r, dim=1, keepdim=True)
                #     p_dot_Mp = torch.sum(p * Mp, dim=1, keepdim=True)
                #     alpha = r_dot_r / (p_dot_Mp + 1e-8)
                #     v = v + alpha * p
                #     r_new = r - alpha * Mp
                #     # Compute beta = (r_new^T * r_new) / (r^T * r)
                #     r_new_dot_r_new = torch.sum(r_new * r_new, dim=1, keepdim=True)
                #     beta = r_new_dot_r_new / (r_dot_r + 1e-8)
                #     p = r_new + beta * p
                #     r = r_new

                # ==========================================
                # GMRES ( Batched )
                # ==========================================
                eps = 1e-8
                vv = torch.zeros_like(b)  #
                r0 = b  #
                beta = torch.norm(r0, dim=1, keepdim=True)  # (bs, 1)
                # Store Arnoldi V Hessenberg H
                V = [r0 / (beta + eps)]
                H = torch.zeros(bs, n_iters + 1, n_iters, device=b.device, dtype=b.dtype)
                # Batched Arnoldi (Modified Gram-Schmidt)
                for k in range(n_iters):
                    v_k = V[k]
                    w = M_product(v_k)  # (bs, N)
                    for i in range(k + 1):
                        # h_{i,k} = v_i^T * w
                        h_ik = torch.sum(V[i] * w, dim=1, keepdim=True)  # (bs, 1)
                        H[:, i, k] = h_ik.squeeze(1)  # Hessenberg
                        w = w - h_ik * V[i]  #
                    # h_{k+1, k} = ||w||
                    h_next = torch.norm(w, dim=1, keepdim=True)  # (bs, 1)
                    H[:, k + 1, k] = h_next.squeeze(1)
                    V.append(w / (h_next + eps))
                # g = \beta * e_1
                g = torch.zeros(bs, n_iters + 1, 1, device=b.device, dtype=b.dtype)
                g[:, 0, :] = beta
                # Batched Least Squares: min_y || H y - g ||_2
                # H: (bs, n_iters+1, n_iters), g: (bs, n_iters+1, 1)
                yy = torch.linalg.lstsq(H, g).solution  #: (bs, n_iters, 1)
                # vv = V_k * y
                # V_tensor: (bs, N, n_iters)
                V_tensor = torch.stack(V[:-1], dim=2)
                vv = torch.bmm(V_tensor, yy).squeeze(2)  # (bs, N)
                v = vv
                # ==========================================

                final_vjp = H_funcs.Ht(v.view(1, -1))  # (1 -1)
                final_vjp_input = final_vjp.view_as(img)

                score_y_given_x = torch.autograd.grad(
                    outputs=x_0_hat,
                    inputs=img,
                    grad_outputs=final_vjp_input,
                    retain_graph=False
                )[0]  #

                # DDPM
                if diffusion_sampler == 'ddpm':
                    norm_grad = ((1 - self.alphas_cumprod[time]) * score_y_given_x) / self.sqrt_alphas_cumprod[time]
                    coef1 = extract_and_expand(self.posterior_mean_coef1, time, img)
                    img = x_t + norm_grad * coef1
                else:
                    if alg_name == 'pgdm':
                        # DDIM PGDM
                        scale = torch.sqrt(out["alpha_prev"] * out["alpha"])
                        norm_grad = sigma_t * score_y_given_x
                        img = x_t + norm_grad * scale
                    elif alg_name == 'mmps':
                        # DDIM MMPS
                        scale = out["scale"]
                        norm_grad = sigma_t * score_y_given_x
                        img = x_t + norm_grad * scale

            img = img.detach_()

            if record:
                if idx % 10 == 0:
                    file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))

        return img

    def _step_cm_mmps(self,
                      model,
                      x_start,
                      measurement,
                      H_funcs,
                      noise_std,
                      record,
                      save_root,
                      alg_name,
                      obs_module=None):
        """
        CM-prior version of MMPS.

        This mirrors _step_mmps(...) for the measurement correction, but uses
        the differentiable CM endpoint as x_0_hat and advances the state with
        CM-style VP re-noising instead of DDPM/DDIM transitions.
        """
        if obs_module is not None:
            raise ValueError(
                "CM-MMPS requires a differentiable measurement model, "
                "but a non-differentiable observation (obs_module) was provided. "
                "Use a GAMP-based algorithm instead (e.g., gamp_mm)."
            )
        if alg_name != 'cm_mmps':
            raise NotImplementedError("CM-MMPS is implemented only for algorithm name 'cm_mmps'.")

        img = x_start
        device = x_start.device
        num_cm_steps = len(model.sigmas)
        pbar = tqdm(list(range(num_cm_steps - 1)))

        noise_sigma2 = noise_std ** 2
        bs = 1
        M = -1
        y = measurement.view(bs, M)

        for loop_idx in pbar:
            sigma = model.get_sigma(loop_idx, device=img.device, dtype=img.dtype)
            a_t, b2_t = model.vp_coeffs(sigma)
            sigma_t = b2_t / (a_t ** 1)

            img = img.requires_grad_()
            x_0_hat = model.endpoint_from_vp(img, sigma)

            n_iters = 5
            pbar.set_postfix({'sigma': sigma.item()}, refresh=False)

            # Compute b = y - A * E[x|x_t]
            Ax0 = H_funcs.H(x_0_hat.view(1, -1))
            b = y - Ax0.view(bs, M)

            def M_product(v):
                # Compute A^T * v.
                At_v = H_funcs.Ht(v.view(1, -1))

                # Compute VJP: (grad_outputs^T * J)^T -> J^T * grad_outputs.
                vjp_input = sigma_t * At_v
                vjp_input_for_grad = vjp_input.view_as(img)
                vjp = torch.autograd.grad(
                    outputs=x_0_hat,
                    inputs=img,
                    grad_outputs=vjp_input_for_grad,
                    retain_graph=True
                )[0]

                # Obtain M*v = Sigma_y * v + A * vjp.
                Avjp = H_funcs.H(vjp.view(1, -1)).view(bs, M)
                return noise_sigma2 * v + Avjp

            # GMRES (batched), copied from _step_mmps.
            eps = 1e-8
            r0 = b
            beta = torch.norm(r0, dim=1, keepdim=True)
            V = [r0 / (beta + eps)]
            H = torch.zeros(bs, n_iters + 1, n_iters, device=b.device, dtype=b.dtype)
            for k in range(n_iters):
                v_k = V[k]
                w = M_product(v_k)
                for i in range(k + 1):
                    h_ik = torch.sum(V[i] * w, dim=1, keepdim=True)
                    H[:, i, k] = h_ik.squeeze(1)
                    w = w - h_ik * V[i]
                h_next = torch.norm(w, dim=1, keepdim=True)
                H[:, k + 1, k] = h_next.squeeze(1)
                V.append(w / (h_next + eps))
            g = torch.zeros(bs, n_iters + 1, 1, device=b.device, dtype=b.dtype)
            g[:, 0, :] = beta
            yy = torch.linalg.lstsq(H, g).solution
            V_tensor = torch.stack(V[:-1], dim=2)
            v = torch.bmm(V_tensor, yy).squeeze(2)

            final_vjp = H_funcs.Ht(v.view(1, -1))
            final_vjp_input = final_vjp.view_as(img)
            score_y_given_x = torch.autograd.grad(
                outputs=x_0_hat,
                inputs=img,
                grad_outputs=final_vjp_input,
                retain_graph=False
            )[0]

            # Convert the CM-MMPS state correction into a clean estimate.
            x_hat1_raw = x_0_hat + (b2_t / a_t) * score_y_given_x
            correction_damping = getattr(model, "correction_damping", 1.0)
            x_hat1 = x_0_hat + correction_damping * (x_hat1_raw - x_0_hat)

            if loop_idx < num_cm_steps - 2:
                sigma_next = model.get_sigma(loop_idx + 1, device=img.device, dtype=img.dtype)
                a_next, _ = model.vp_coeffs(sigma_next)
                sigma_min = getattr(model.diffusion, "sigma_min", 0.002)
                sigma_min = torch.as_tensor(sigma_min, device=img.device, dtype=img.dtype)
                noise_scale = torch.sqrt(torch.clamp(sigma_next ** 2 - sigma_min ** 2, min=0.0))
                img = a_next * x_hat1.detach() + a_next * noise_scale * torch.randn_like(x_hat1)
            else:
                img = x_hat1.detach()

            img = img.detach_()

            if record:
                file_path = os.path.join(save_root, f"progress/x_cm_mmps_{str(loop_idx).zfill(4)}.png")
                plt.imsave(file_path, clear_color(img))

            torch.cuda.empty_cache()

        return img

    def _step_gamp(self,
                   model,
                   x_start,
                   measurement,
                   H_funcs,
                   noise_std,
                   record,
                   save_root,
                   alg_name,
                   diffusion_sampler,
                   obs_module=None):
        """
        The function used for sampling from noise.
        """
        img = x_start
        device = x_start.device

        self.old_x_0_listhat = []

        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        x_hat_temp = torch.zeros(H_funcs.block_num, H_funcs.M, device=device)
        tau_x_temp = torch.ones(H_funcs.block_num, H_funcs.M, device=device)
        s_temp = torch.zeros(H_funcs.block_num, H_funcs.N, device=device)
        y = measurement.view(H_funcs.block_num, H_funcs.N)

        # gamp = GAMP(
        #     prior=lambda a_t, b2_t, tau_r, x_t, x_0_hat, nabla_r_xt: GAMP.sparse_prior3(a_t, b2_t, tau_r, x_t, x_0_hat,
        #                                                                                 nabla_r_xt),
        #     likelihood=lambda p, tau_p, y, noise_sigma: GAMP.awgn_likelihood_cs(p, tau_p, y, noise_sigma),
        #     max_iter = 10)   # 10

        noise_sigma = noise_std
        delta0 = noise_sigma ** 2
        bs = H_funcs.block_num
        M = H_funcs.N
        N = H_funcs.M
        for loop_idx, idx in enumerate(pbar):
            time = torch.tensor([idx] * img.shape[0], device=device)
            # print(time)
            img = img.requires_grad_()
            out = self.p_sample(x=img, t=time, model=model)
            x_0_hat = out['pred_xstart']
            x_t = img
            b2_t = (1 - self.alphas_cumprod[time])
            a_t = self.sqrt_alphas_cumprod[time]
            # sigma_t = b2_t / (a_t ** 2) #
            sigma_t = b2_t / (a_t ** 2)  # MMPS 1
            x_hat = x_hat_temp
            tau_x = tau_x_temp
            s = s_temp

            if idx < 50:
                progress = 1 - (idx / 50)  #
                max_iter = int(6 + 5 * (progress ** 2))  #
            elif idx < 30:
                max_iter = 3
            else:
                max_iter = 3

            for iter in range(max_iter):
                with torch.no_grad():
                    # --- output Step ---
                    if iter == 0:
                        tau_p = (H_funcs.H_squared(tau_x.view(1, -1))).view(bs, M)

                    p = H_funcs.H(x_hat.view(1, -1)).view(bs, M) - s * tau_p

                    if obs_module is not None:
                        z_hat, tau_z = obs_module.gamp_likelihood(p, tau_p, y, noise_sigma)
                    else:
                        tau_z = 1.0 / (1.0 / torch.clamp(tau_p, min=1e-15) + 1.0 / delta0)
                        z_hat = (p / torch.clamp(tau_p, min=1e-15) + y / delta0) * tau_z

                    # tau_z = torch.real(tau_z)
                    tau_p_clamped = torch.clamp(tau_p, min=1e-10)
                    s = (z_hat - p) / tau_p_clamped
                    tau_s = (1.0 - tau_z / tau_p_clamped) / tau_p_clamped
                    # tau_s = torch.real(tau_s)
                    tau_s = torch.clamp(tau_s, min=1e-10)

                    # --- input Step ---
                    A2_tau_s = H_funcs.Ht_squared(tau_s.view(1, -1)).view(bs, N)
                    tau_r = 1.0 / torch.clamp(A2_tau_s, min=1e-10)

                    At_s = H_funcs.Ht(s.view(1, -1)).view(bs, N)
                    r = x_hat + tau_r * At_s

                    if alg_name == 'gamp_dmps':
                        # DMPS
                        x_t_flat = x_t.view(bs, N)
                        diff = r - x_t_flat / a_t
                        nabla_r_xt = diff / (1 * a_t * tau_r + b2_t / a_t)

                    elif alg_name == 'gamp-ga':
                        x_0_hat_flat = x_0_hat.view(bs, N)
                        diff = r.detach() - x_0_hat_flat
                        norm = 0.5 * torch.sum(diff ** 2)
                        is_last_grad = (iter == max_iter - 1)
                        norm_grad = torch.autograd.grad(outputs=norm, inputs=img, retain_graph=is_last_grad)[0]
                        nabla_r_xt = - norm_grad.view(bs, N) / (tau_r + b2_t / (1))
                        del norm_grad, norm, diff
                    else:
                        # MMPS
                        ## Compute b = r - E[x|x_t]
                        bb = r.detach() - x_0_hat.view(1, -1).view(bs, N)  # (bs, N)
                        n_iters = 1

                        def M_product(vv):
                            # Compute VJP: (grad_outputs^T * J)^T -> J^T * grad_outputs
                            vjp_input = (sigma_t) * vv.view(1, -1)  # (1,-1)  # 0.1

                            if alg_name == 'gamp_ga':
                                # PGDM
                                vjp = vjp_input

                            elif alg_name == 'gamp_mm':
                                # MMPS  asymmetric
                                vjp_input_for_grad = vjp_input.view_as(img)
                                vjp = torch.autograd.grad(
                                    outputs=x_0_hat,
                                    inputs=img,
                                    grad_outputs=vjp_input_for_grad,
                                    retain_graph=True
                                )[0]
                            # Obtain M*v = \Sigma_y * v + A * vjp
                            Avjp = vjp.view(1, -1).view(bs, N)
                            return tau_r * vv + Avjp

                        # CG
                        # vv = torch.zeros_like(bb)
                        # rr = bb #  - M_product(vv) = 0
                        # pp = rr.clone()
                        # # CG iteration (bs, N) processing
                        # for _ in range(n_iters):
                        #     Mp = M_product(pp)
                        #     # Compute alpha = (r^T * r) / (p^T * M * p)
                        #     r_dot_r = torch.sum(rr * rr, dim=1, keepdim=True)
                        #     p_dot_Mp = torch.sum(pp * Mp, dim=1, keepdim=True)
                        #     alpha = r_dot_r / (p_dot_Mp + 1e-8)
                        #     vv = vv + alpha * pp
                        #     r_new = rr - alpha * Mp
                        #     # Compute beta = (r_new^T * r_new) / (r^T * r)
                        #     r_new_dot_r_new = torch.sum(r_new * r_new, dim=1, keepdim=True)
                        #     beta = r_new_dot_r_new / (r_dot_r + 1e-8)
                        #     pp = r_new + beta * pp
                        #     rr = r_new

                        # ==========================================
                        # GMRES ( Batched )
                        # ==========================================
                        # eps = 1e-8
                        # vv = torch.zeros_like(bb)  #
                        # r0 = bb  #
                        # beta = torch.norm(r0, dim=1, keepdim=True)  # (bs, 1)
                        # # Store Arnoldi V Hessenberg H
                        # V = [r0 / (beta + eps)]
                        # H = torch.zeros(bs, n_iters + 1, n_iters, device=bb.device, dtype=bb.dtype)
                        # # Batched Arnoldi (Modified Gram-Schmidt)
                        # for k in range(n_iters):
                        #     v_k = V[k]
                        #     w = M_product(v_k)  # (bs, N)
                        #     for i in range(k + 1):
                        #         # h_{i,k} = v_i^T * w
                        #         h_ik = torch.sum(V[i] * w, dim=1, keepdim=True)  # (bs, 1)
                        #         H[:, i, k] = h_ik.squeeze(1)  # Hessenberg
                        #         w = w - h_ik * V[i]  #
                        #     # h_{k+1, k} = ||w||
                        #     h_next = torch.norm(w, dim=1, keepdim=True)  # (bs, 1)
                        #     H[:, k + 1, k] = h_next.squeeze(1)
                        #     V.append(w / (h_next + eps))
                        # # g = \beta * e_1
                        # g = torch.zeros(bs, n_iters + 1, 1, device=bb.device, dtype=bb.dtype)
                        # g[:, 0, :] = beta
                        # # Batched Least Squares: min_y || H y - g ||_2
                        # # H: (bs, n_iters+1, n_iters), g: (bs, n_iters+1, 1)
                        # yy = torch.linalg.lstsq(H, g).solution  #: (bs, n_iters, 1)
                        # # vv = V_k * y
                        # # V_tensor: (bs, N, n_iters)
                        # V_tensor = torch.stack(V[:-1], dim=2)
                        # vv = torch.bmm(V_tensor, yy).squeeze(2)  # (bs, N)
                        # # ==========================================
                        # ==========================================
                        # Optimization 2: GMRES (for n_iters = 1)
                        # ==========================================
                        eps_gmres = 1e-8
                        r0 = bb
                        beta = torch.norm(r0, dim=1, keepdim=True)  # (bs, 1)
                        v_0 = r0 / (beta + eps_gmres)  # (bs, N)
                        w = M_product(v_0)  # (bs, N)
                        # h_{0,0} = v_0^T * w
                        h_00 = torch.sum(v_0 * w, dim=1, keepdim=True)  # (bs, 1)
                        # w_perp = w - h_{0,0} * v_0
                        w_perp = w - h_00 * v_0  # (bs, N)
                        # h_{1,0} = ||w_perp||
                        h_10 = torch.norm(w_perp, dim=1, keepdim=True)  # (bs, 1)
                        # y = (h_00 * beta) / (h_00^2 + h_10^2 + eps)
                        y_opt = (h_00 * beta) / (h_00 ** 2 + h_10 ** 2 + eps_gmres)  # (bs, 1)
                        vv = v_0 * y_opt  # (bs, N)
                        # ==========================================

                        final_vjp_input = vv.view(1, -1).view_as(img)  # (1 -1)

                        is_last_grad = (iter == max_iter - 1)
                        score_y_given_x = torch.autograd.grad(
                            outputs=x_0_hat,
                            inputs=img,
                            grad_outputs=final_vjp_input,
                            retain_graph=not is_last_grad
                        )[0]  #

                        nabla_r_xt = score_y_given_x
                        del final_vjp_input

                # prior
                # x_hat, tau_x, x_hat1 = gamp.prior(a_t, b2_t, tau_r, x_t, x_0_hat, nabla_r_xt)
                nabla_xt_r = (a_t * x_0_hat - x_t) / (b2_t) + nabla_r_xt.view_as(img)
                x_hat1 = (x_t + b2_t * nabla_xt_r) / (a_t)
                x_hat = x_hat1.view(tau_r.shape)

                # if iter < max_iter - 1:   # < max_iter - 1:
                #     vx = torch.randn_like(x_t)
                #     hvp = torch.autograd.grad(
                #     outputs=nabla_xt_r,
                #     inputs=x_t,
                #     grad_outputs=vx,
                #     retain_graph=True
                #     )[0]
                #     trace_H = vx * hvp

                # === Optimization 1: Only when iter == 0 ===
                if iter <= 1 and max_iter > 1:
                    vx = torch.randn_like(x_t)
                    hvp = torch.autograd.grad(
                        outputs=nabla_xt_r,
                        inputs=x_t,
                        grad_outputs=vx,
                        retain_graph=True
                    )[0]
                    # Use detach()
                    trace_H = (vx * hvp)

                if iter < max_iter - 1:  # < max_iter - 1:
                    # \tau_x = (b^2/a^2) + (b^4/a^2) * trace_H
                    tau_x = (b2_t / a_t ** 2) + (b2_t ** 2 / a_t ** 2) * trace_H
                else:
                    tau_x = (b2_t / a_t ** 2) - (b2_t ** 2 / a_t ** 2) / ((a_t ** 2) * tau_r + b2_t)  #
                # inflation_factor = 1.0 + 0.5 * b2_t.item()
                # tau_x = tau_x * inflation_factor
                tau_x = torch.clamp(tau_x, min=1e-15, max=1e8)

                tau_p = (H_funcs.H_squared(tau_x.view(1, -1))).view(bs, M)

                del p, z_hat, tau_p_clamped, tau_s, A2_tau_s, At_s, r, nabla_r_xt

            x_hat_temp = x_hat
            tau_x_temp = tau_x
            s_temp = s

            if diffusion_sampler == 'ddpm':
                # DDPM
                img = self.mean_processor.q_posterior_mean(x_hat1, img, time)
                noise = torch.randn_like(img)
                if idx != 0:  # no noise when t == 0
                    img += torch.exp(0.5 * out['log_variance']) * noise  # 0.5

            elif diffusion_sampler == 'ddim':
                # DDIM
                eta = 1.0
                alpha_bar = extract_and_expand(self.alphas_cumprod, time, x_t)
                alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, time, x_t)
                sigma = (
                        eta
                        * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                        * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
                )
                # Equation 12.
                noise = torch.randn_like(img)
                coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, time, x_t)
                coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, time, x_t)
                eps = (coef1 * x_t - x_hat1) / coef2
                mean_pred = (
                        x_hat1 * torch.sqrt(alpha_bar_prev)
                        + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
                )
                img = mean_pred
                if idx != 0:
                    img += sigma * noise

            # DPM-solver++
            # x_0_cur = x_hat1

            # #
            # lambda_cur = self.lambda_t[idx]
            # lambda_next = self.lambda_next[idx]
            # alpha_next = np.sqrt(self.alphas_cumprod_prev[idx])
            # sigma_cur = np.sqrt(1.0 - self.alphas_cumprod[idx])
            # sigma_next = np.sqrt(1.0 - self.alphas_cumprod_prev[idx])
            # # h
            # h = lambda_next - lambda_cur

            # # --- Algorithm 2 ---
            # if loop_idx == 0 or len(self.old_x_0_listhat) == 0:
            #     # Step 4:
            #     # x_next = (sigma_next / sigma_cur) * x - alpha_next * (exp(-h) - 1) * x_theta
            #     img = (sigma_next / sigma_cur) * img - alpha_next * torch.expm1(-torch.tensor(h, device=device)) * x_0_cur

            #     # Step 5:
            #     self.old_x_0_listhat.append((lambda_cur, x_0_cur))

            # else:
            #     # Step 7: r = h_{i-1} / h_i
            #     lambda_prev, x_0_prev = self.old_x_0_listhat[-1]
            #     h_prev = lambda_cur - lambda_prev
            #     r = h_prev / h

            #     # Step 8: D_i
            #     # D_i = (1 + 1/2r) * x_theta_cur - (1/2r) * x_theta_prev
            #     D_i = (1.0 + 1.0 / (2.0 * r)) * x_0_cur - (1.0 / (2.0 * r)) * x_0_prev

            #     # Step 9:
            #     # x_next = (sigma_next / sigma_cur) * x - alpha_next * (exp(-h) - 1) * D_i
            #     img = (sigma_next / sigma_cur) * img - alpha_next * torch.expm1(-torch.tensor(h, device=device)) * D_i

            #     # Step 10: 2M
            #     self.old_x_0_listhat.append((lambda_cur, x_0_cur))
            #     if len(self.old_x_0_listhat) > 1:
            #         self.old_x_0_listhat.pop(0)

            img = img.detach_()
            pbar.set_postfix({'maxiter': max_iter}, refresh=False)
            if record:
                if idx % 10 == 0:
                    file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))
            # img = img.clip_(-1,1)
            ###
            torch.cuda.empty_cache()

        return img

    def _step_gamp_cm(self,
                      model,
                      x_start,
                      measurement,
                      H_funcs,
                      noise_std,
                      record,
                      save_root,
                      alg_name,
                      obs_module=None):
        """
        CM-prior version of GAMP-MM.

        This keeps the GAMP output/input-step structure from _step_gamp(...),
        but replaces the DDPM/DDIM prior endpoint with the differentiable
        OpenAI Consistency Model endpoint and uses VP re-noising between CM
        sigma levels.
        """
        img = x_start
        device = x_start.device

        if alg_name != 'gamp_mm':
            raise NotImplementedError("CM prior is first implemented only for gamp_mm.")

        self.old_x_0_listhat = []

        num_cm_steps = len(model.sigmas)
        pbar = tqdm(list(range(num_cm_steps)))
        x_hat_temp = torch.zeros(H_funcs.block_num, H_funcs.M, device=device)
        tau_x_temp = torch.ones(H_funcs.block_num, H_funcs.M, device=device)
        s_temp = torch.zeros(H_funcs.block_num, H_funcs.N, device=device)
        y = measurement.view(H_funcs.block_num, H_funcs.N)

        noise_sigma = noise_std
        delta0 = noise_sigma ** 2
        bs = H_funcs.block_num
        M = H_funcs.N
        N = H_funcs.M
        rho_cm = 0.8

        for loop_idx in pbar:
            sigma = model.get_sigma(loop_idx, device=img.device, dtype=img.dtype)
            a_t, b2_t = model.vp_coeffs(sigma)
            sigma_t = b2_t / (a_t ** 2)

            img = img.requires_grad_()
            x_0_hat = model.endpoint_from_vp(img, sigma)
            x_t = img

            x_hat = x_hat_temp
            tau_x = tau_x_temp
            tau_x_new = tau_x_temp
            s = s_temp
            # if loop_idx < 10:
            #     progress = (loop_idx / 40) #
            #     max_iter = 3 # int(2 + 3 * (progress ** 2)) #
            # else:
            #     max_iter = 3

            max_iter = 3

            for iter in range(max_iter):
                with torch.no_grad():
                    # --- output Step ---
                    if iter == 0:
                        tau_p = (H_funcs.H_squared(tau_x.view(1, -1))).view(bs, M)

                    p = H_funcs.H(x_hat.view(1, -1)).view(bs, M) - s * tau_p

                    if obs_module is not None:
                        z_hat, tau_z = obs_module.gamp_likelihood(p, tau_p, y, noise_sigma)
                    else:
                        tau_z = 1.0 / (1.0 / torch.clamp(tau_p, min=1e-15) + 1.0 / delta0)
                        z_hat = (p / torch.clamp(tau_p, min=1e-15) + y / delta0) * tau_z

                    tau_p_clamped = torch.clamp(tau_p, min=1e-10)
                    s = (z_hat - p) / tau_p_clamped
                    tau_s = (1.0 - tau_z / tau_p_clamped) / tau_p_clamped
                    tau_s = torch.clamp(tau_s, min=1e-10)

                    # --- input Step ---
                    A2_tau_s = H_funcs.Ht_squared(tau_s.view(1, -1)).view(bs, N)
                    tau_r = 1.0 / torch.clamp(A2_tau_s, min=1e-10)

                    At_s = H_funcs.Ht(s.view(1, -1)).view(bs, N)
                    r = x_hat + tau_r * At_s

                    # GAMP-MM: solve the local moment-matching correction with
                    # the same one-step GMRES approximation used in _step_gamp.
                    bb = r.detach() - x_0_hat.view(1, -1).view(bs, N)

                    def M_product(vv):
                        vjp_input = sigma_t * vv.view(1, -1)
                        vjp = vjp_input  # GA
                        # vjp_input_for_grad = vjp_input.view_as(img)
                        # vjp = torch.autograd.grad(
                        #     outputs=x_0_hat,
                        #     inputs=img,
                        #     grad_outputs=vjp_input.view_as(img),
                        #     retain_graph=True
                        # )[0]
                        Avjp = vjp.view(1, -1).view(bs, N)
                        return tau_r * vv + Avjp

                    eps_gmres = 1e-8
                    r0 = bb
                    beta = torch.norm(r0, dim=1, keepdim=True)
                    v_0 = r0 / (beta + eps_gmres)
                    w = M_product(v_0)
                    h_00 = torch.sum(v_0 * w, dim=1, keepdim=True)
                    w_perp = w - h_00 * v_0
                    h_10 = torch.norm(w_perp, dim=1, keepdim=True)
                    y_opt = (h_00 * beta) / (h_00 ** 2 + h_10 ** 2 + eps_gmres)
                    vv = v_0 * y_opt

                    final_vjp_input = vv.view(1, -1).view_as(img)
                    is_last_grad = (iter == max_iter)  # - 1
                    score_y_given_x = torch.autograd.grad(
                        outputs=x_0_hat,
                        inputs=img,
                        grad_outputs=final_vjp_input,
                        retain_graph=not is_last_grad
                    )[0]

                    nabla_r_xt = score_y_given_x
                    del final_vjp_input

                # prior correction in VP coordinates
                nabla_xt_r = (a_t * x_0_hat - x_t) / b2_t + nabla_r_xt.view_as(img)
                # Old undamped GAMP correction:
                # x_hat1 = (x_t + b2_t * nabla_xt_r) / a_t
                x_hat1_raw = (x_t + b2_t * nabla_xt_r) / a_t
                correction_damping = getattr(model, "correction_damping", 1.0)
                x_hat1 = x_0_hat + correction_damping * (x_hat1_raw - x_0_hat)
                x_hat_new = x_hat1.view(tau_r.shape)

                if iter < 1 and max_iter > 1:
                    vx = torch.randn_like(x_t)
                    hvp = torch.autograd.grad(
                        outputs=nabla_xt_r,
                        inputs=x_t,
                        grad_outputs=vx,
                        retain_graph=True
                    )[0]
                    trace_H = vx * hvp
                    trace_H = trace_H.view(bs, -1)

                if iter < max_iter:
                    tau_x_new = (b2_t / a_t ** 2) + (b2_t ** 2 / a_t ** 2) * trace_H
                else:
                    tau_x_new = (b2_t / a_t ** 2) - (b2_t ** 2 / a_t ** 2) / ((a_t ** 2) * tau_r + b2_t)
                    tau_x_new = tau_x_new * getattr(model, "tau_inflation", 1.0)

                with torch.no_grad():
                    x_hat = rho_cm * x_hat_new.detach() + (1.0 - rho_cm) * x_hat.detach()
                    tau_x = rho_cm * tau_x_new.detach() + (1.0 - rho_cm) * tau_x.detach()
                    tau_x = torch.clamp(tau_x, min=1e-15, max=1e8)

                tau_p = (H_funcs.H_squared(tau_x.view(1, -1))).view(bs, M)

                del p, z_hat, tau_p_clamped, tau_s, A2_tau_s, At_s, r, nabla_r_xt

            x_hat_temp = x_hat
            tau_x_temp = tau_x
            s_temp = s

            # x_hat1 = x_0_hat ### pure generation
            if loop_idx < num_cm_steps - 1:
                sigma_next = model.get_sigma(loop_idx + 1, device=img.device, dtype=img.dtype)
                a_next, b2_next = model.vp_coeffs(sigma_next)
                # b_next = torch.sqrt(b2_next)
                noise_scale = torch.sqrt(torch.clamp(sigma_next ** 2 - 0.002 ** 2, min=0.0))
                img = a_next * x_hat1.detach() + a_next * noise_scale * torch.randn_like(x_hat1)
            else:
                img = x_hat1.detach()

            img = img.detach_()
            pbar.set_postfix({'sigma': sigma.item(), 'maxiter': max_iter}, refresh=False)
            if record:
                file_path = os.path.join(save_root, f"progress/x_cm_{str(loop_idx).zfill(4)}.png")
                plt.imsave(file_path, clear_color(img))

            torch.cuda.empty_cache()

        return img

    def _step_vamp(self,
                   model,
                   x_start,
                   measurement,
                   H_funcs,
                   noise_std,
                   record,
                   save_root,
                   alg_name,
                   diffusion_sampler,
                   obs_module=None):
        """
        The function used for sampling from noise.
        """
        if obs_module is not None:
            raise NotImplementedError(
                "VAMP with non-differentiable observation is not yet implemented. "
                "Use a GAMP-based algorithm instead (e.g., gamp_mm, gamp_ga, gamp-ga)."
            )
        img = x_start
        device = x_start.device

        pbar = tqdm(list(range(self.num_timesteps))[::-1])

        delta2_0 = noise_std ** 2
        s2 = H_funcs._S_small ** 2  #
        delta = 1.0
        variance_min = 1e-12
        variance_max = 1e8
        precision_min = 1.0 / variance_max

        def require_finite(name, value):
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"VAMP produced a non-finite value in {name} at timestep {idx}.")

        # BlockCS_H selects rows from an orthonormal DCT matrix, so A A^T = I.
        if not torch.allclose(s2, torch.ones_like(s2), rtol=1e-4, atol=1e-5):
            raise ValueError("VAMP closed-form Module A requires a row-orthonormal matrix (A A^T = I).")

        v_A_pri_temp = 10 * x_start.new_ones(1)
        x_A_pri_temp = x_start.new_zeros(H_funcs.block_num, H_funcs.M)
        y = measurement.view(H_funcs.block_num, H_funcs.N)
        bs = H_funcs.block_num
        M = H_funcs.N
        N = H_funcs.M
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=device)
            # print(time)
            img = img.requires_grad_()
            out = self.p_sample(x=img, t=time, model=model)
            x_0_hat = out['pred_xstart']
            x_t = img
            b2_t = (1 - self.alphas_cumprod[time])
            a_t = self.sqrt_alphas_cumprod[time]
            sigma_t = b2_t / (a_t ** 2)  #
            v_A_pri = v_A_pri_temp
            x_A_pri = x_A_pri_temp
            Turbo = 1
            v_A_ext_old = x_start.new_ones(1)
            x_A_ext_old = x_start.new_zeros(H_funcs.block_num, H_funcs.M)
            with (torch.no_grad()):
                for i in range(Turbo):
                    # ---------------------------------------------------------
                    # 1. LMMSE Estimator (Module A)
                    # ---------------------------------------------------------
                    # --- Line 5 x_A_post ---
                    # (v_A_pri * A * A^T + delta2_0 * I)^-1 * (y - A * x_A_pri)
                    v_A_pri = torch.clamp(v_A_pri, min=variance_min, max=variance_max)
                    require_finite("v_A_pri", v_A_pri)
                    b = y - H_funcs.H(x_A_pri.view(1, -1)).view(bs, M)
                    require_finite("Module-A residual", b)

                    # For the current DCT operator A A^T = I, hence the LMMSE
                    # system has this exact closed-form solution.
                    lmmse_denom = v_A_pri + delta2_0
                    require_finite("Module-A denominator", lmmse_denom)
                    if torch.any(lmmse_denom <= 0):
                        raise FloatingPointError("VAMP Module-A denominator must be positive.")
                    u = b / lmmse_denom

                    # --- Line 5 x_A_post ---
                    x_A_post = x_A_pri + v_A_pri * H_funcs.Ht(u.view(1, -1)).view(bs, N)
                    # --- Line 6   v_A_post ---
                    # tr(A.T (v_A*AA.T + delta2*I)^-1 A) = sum( lambda / (v_A * lambda + delta2) )
                    tr_term = torch.sum(s2 / (v_A_pri * s2 + delta2_0))
                    v_A_post = v_A_pri - (v_A_pri ** 2 / N) * tr_term
                    # --- Line 7-8 (Extrinsic) ---
                    v_A_post = torch.clamp(v_A_post, min=variance_min, max=variance_max)
                    require_finite("x_A_post", x_A_post)
                    require_finite("v_A_post", v_A_post)
                    denom = 1.0 / v_A_post - 1.0 / v_A_pri
                    require_finite("Module-A extrinsic precision", denom)
                    denom = torch.clamp(denom, min=1e-8)
                    v_A_ext = 1.0 / denom
                    # v_A_ext = 1.0 / (1.0 / v_A_post - 1.0 / v_A_pri)
                    x_A_ext = v_A_ext * (x_A_post / v_A_post - x_A_pri / v_A_pri)
                    require_finite("x_A_ext", x_A_ext)
                    require_finite("v_A_ext", v_A_ext)
                    # Line 9
                    x_B_pri = x_A_ext * delta + x_A_ext_old * (1 - delta)
                    v_B_pri = v_A_ext * delta + v_A_ext_old * (1 - delta)
                    require_finite("x_B_pri", x_B_pri)
                    require_finite("v_B_pri", v_B_pri)
                    x_A_ext_old, v_A_ext_old = x_A_ext, v_A_ext
                    # MMPS
                    #   # Compute b = r - E[x|x_t]
                    bb = x_B_pri.detach() - x_0_hat.view(1, -1).view(bs, N)  # (bs, N)

                    def M_product2(vv):
                        # Compute VJP: (grad_outputs^T * J)^T -> J^T * grad_outputs
                        vjp_input = (sigma_t) * vv.view(1, -1)  # (1,-1)  # 0.1
                        # PGDM
                        # vjp = vjp_input
                        # MMPS  asymmetric
                        vjp_input_for_grad = vjp_input.view_as(img)
                        vjp = torch.autograd.grad(
                            outputs=x_0_hat,
                            inputs=img,
                            grad_outputs=vjp_input_for_grad,
                            retain_graph=True
                        )[0]
                        # Obtain M*v = \Sigma_y * v + A * vjp
                        Avjp = vjp.view(1, -1).view(bs, N)
                        return v_B_pri * vv + 1 * Avjp

                    # ==========================================
                    # Optimization 2: GMRES (for n_iters = 1)
                    # ==========================================
                    eps_gmres = 1e-8
                    r0 = bb
                    beta = torch.norm(r0, dim=1, keepdim=True)  # (bs, 1)
                    require_finite("GMRES initial residual", beta)
                    v_0 = r0 / (beta + eps_gmres)  # (bs, N)
                    w = M_product2(v_0)  # (bs, N)
                    require_finite("GMRES matrix-vector product", w)
                    # h_{0,0} = v_0^T * w
                    h_00 = torch.sum(v_0 * w, dim=1, keepdim=True)  # (bs, 1)
                    # w_perp = w - h_{0,0} * v_0
                    w_perp = w - h_00 * v_0  # (bs, N)
                    # h_{1,0} = ||w_perp||
                    h_10 = torch.norm(w_perp, dim=1, keepdim=True)  # (bs, 1)
                    # y = (h_00 * beta) / (h_00^2 + h_10^2 + eps)
                    y_opt = (h_00 * beta) / (h_00 ** 2 + h_10 ** 2 + eps_gmres)  # (bs, 1)
                    vv = v_0 * y_opt  # (bs, N)
                    require_finite("GMRES solution", vv)
                    # ==========================================

                    final_vjp_input = vv.view(1, -1).view_as(img)  # (1 -1)
                    score_y_given_x = torch.autograd.grad(
                        outputs=x_0_hat,
                        inputs=img,
                        grad_outputs=final_vjp_input,
                        retain_graph=True
                    )[0]  #

                nabla_r_xt = score_y_given_x
                del final_vjp_input

                tau_r = v_B_pri * torch.ones_like(x_B_pri)
                # Diffusion prior, expanded explicitly from sparse_prior3.
                nabla_xt_r = (a_t * x_0_hat - x_t) / b2_t + nabla_r_xt.view_as(img)
                x_hat1 = (x_t + b2_t * nabla_xt_r) / a_t
                x_B_post = x_hat1.view(tau_r.shape)
                v_B_post_tensor = (
                        (b2_t / a_t ** 2)
                        - (b2_t ** 2 / a_t ** 2) / ((a_t ** 2) * tau_r + b2_t)
                )
                v_B_post_tensor = torch.clamp(
                    v_B_post_tensor, min=variance_min, max=variance_max
                )
                require_finite("x_B_post", x_B_post)
                require_finite("v_B_post_tensor", v_B_post_tensor)

                del nabla_r_xt

                # Line 13-14
                v_B_post = torch.mean(v_B_post_tensor)
                v_B_post = torch.clamp(v_B_post, min=variance_min, max=variance_max)
                v_B_pri = torch.clamp(v_B_pri, min=variance_min, max=variance_max)
                require_finite("v_B_post", v_B_post)
                require_finite("v_B_pri before extrinsic update", v_B_pri)

                precision_B_ext = 1.0 / v_B_post - 1.0 / v_B_pri
                valid_B_ext = torch.isfinite(precision_B_ext) & (precision_B_ext > precision_min)
                safe_precision_B_ext = torch.where(
                    valid_B_ext, precision_B_ext, 1.0 / v_B_pri
                )
                v_B_ext = torch.clamp(
                    1.0 / safe_precision_B_ext, min=variance_min, max=variance_max
                )
                x_B_ext_candidate = v_B_ext * (
                        x_B_post / v_B_post - x_B_pri / v_B_pri
                )
                valid_B_mean = valid_B_ext & torch.isfinite(x_B_ext_candidate).all()
                x_B_ext = torch.where(valid_B_mean, x_B_ext_candidate, x_B_post)
                require_finite("x_B_ext", x_B_ext)
                require_finite("v_B_ext", v_B_ext)
                # Line 15
                x_A_pri = x_B_ext
                v_A_pri = v_B_ext
                # x_A_pri, v_A_pri = x_B_ext, v_B_ext

            v_A_pri_temp = v_A_pri
            x_A_pri_temp = x_A_pri

            if diffusion_sampler == 'ddpm':
                # DDPM
                img = self.mean_processor.q_posterior_mean(x_hat1, img, time)
                noise = torch.randn_like(img)
                if idx != 0:  # no noise when t == 0
                    img += torch.exp(0.5 * out['log_variance']) * noise  # 0.5

            elif diffusion_sampler == 'ddim':
                # DDIM
                eta = 1.0
                alpha_bar = extract_and_expand(self.alphas_cumprod, time, x_t)
                alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, time, x_t)
                sigma = (
                        eta
                        * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                        * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
                )
                # Equation 12.
                noise = torch.randn_like(img)
                coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, time, x_t)
                coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, time, x_t)
                eps = (coef1 * x_t - x_hat1) / coef2
                mean_pred = (
                        x_hat1 * torch.sqrt(alpha_bar_prev)
                        + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
                )
                img = mean_pred
                if idx != 0:
                    img += sigma * noise

            img = img.detach_()
            # pbar.set_postfix({'distance': distance.item()}, refresh=False)
            if record:
                if idx % 10 == 0:
                    file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))
            # img = img.clip_(-1,1)

        return img

    def _step_vamp_cm(self,
                      model,
                      x_start,
                      measurement,
                      H_funcs,
                      noise_std,
                      record,
                      save_root,
                      alg_name,
                      obs_module=None):
        """
        CM-prior version of VAMP.

        This keeps the Module-A/Module-B structure from _step_vamp(...),
        but obtains the differentiable endpoint from the consistency model
        and advances between CM sigma levels with VP re-noising.
        """
        if obs_module is not None:
            raise NotImplementedError(
                "CM-VAMP with non-differentiable observation is not yet implemented. "
                "Use a GAMP-based algorithm instead (e.g., gamp_mm)."
            )
        if alg_name != 'vamp':
            raise NotImplementedError("CM-VAMP is implemented only for algorithm name 'vamp'.")

        img = x_start
        num_cm_steps = len(model.sigmas)
        pbar = tqdm(list(range(num_cm_steps - 1)))

        delta2_0 = noise_std ** 2
        s2 = H_funcs._S_small ** 2
        delta = 1
        rho_A = 1
        variance_min = 1e-15
        variance_max = 1e15
        precision_min = 1.0 / variance_max

        def require_finite(name, value):
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"CM-VAMP produced a non-finite value in {name} at CM step {loop_idx}."
                )

        # # BlockCS_H selects rows from an orthonormal DCT matrix, so A A^T = I.
        # if not torch.allclose(s2, torch.ones_like(s2), rtol=1e-4, atol=1e-5):
        #     print("s2:", s2)
        #     raise ValueError("CM-VAMP closed-form Module A requires a row-orthonormal matrix (A A^T = I).")

        v_A_pri_temp = 100 * x_start.new_ones(1)
        x_A_pri_temp = x_start.new_zeros(H_funcs.block_num, H_funcs.M)
        y = measurement.view(H_funcs.block_num, H_funcs.N)
        bs = H_funcs.block_num
        M = H_funcs.N
        N = H_funcs.M

        v_A_ext_old = 100 * x_start.new_ones(1)
        x_A_ext_old = x_start.new_zeros(H_funcs.block_num, H_funcs.M)
        x_B_ext_old = x_A_pri_temp.detach().clone()
        v_B_ext_old = torch.clamp(
            v_A_pri_temp.detach().clone(),
            min=variance_min,
            max=variance_max,
        )

        for loop_idx in pbar:
            sigma = model.get_sigma(loop_idx, device=img.device, dtype=img.dtype)
            a_t, b2_t = model.vp_coeffs(sigma)
            sigma_t = b2_t / (a_t ** 2)

            img = img.requires_grad_()
            x_0_hat = model.endpoint_from_vp(img, sigma)
            x_t = img

            v_A_pri = v_A_pri_temp
            x_A_pri = x_A_pri_temp

            Turbo = 3

            for i in range(Turbo):
                with torch.no_grad():
                    # ---------------------------------------------------------
                    # 1. LMMSE Estimator (Module A)
                    # ---------------------------------------------------------
                    # v_A_pri = torch.clamp(v_A_pri, min=variance_min, max=variance_max)
                    # # require_finite("v_A_pri", v_A_pri)
                    # b = y - H_funcs.H(x_A_pri.view(1, -1)).view(bs, M)
                    # # require_finite("Module-A residual", b)

                    # # For the current DCT operator A A^T = I.
                    # lmmse_denom = v_A_pri + delta2_0
                    # # require_finite("Module-A denominator", lmmse_denom)
                    # # if torch.any(lmmse_denom <= 0):
                    # #     raise FloatingPointError("CM-VAMP Module-A denominator must be positive.")
                    # u = b / lmmse_denom

                    b = y - H_funcs.H(x_A_pri.view(1, -1)).view(bs, M)

                    def M_product(p):
                        # M = v_A_pri * A * A^T + delta2_0 * I
                        temp = H_funcs.H(H_funcs.Ht(p.view(1, -1))).view(bs, M)
                        return v_A_pri * temp + delta2_0 * p
                        # CG M * u = b

                    u = torch.zeros_like(b)
                    r = b  # - M_product(u) = 0
                    p = r.clone()
                    cg_iters = 1
                    for _ in range(cg_iters):
                        Mp = M_product(p)
                        r_dot_r = torch.sum(r * r, dim=1, keepdim=True)
                        alpha = r_dot_r / (torch.sum(p * Mp, dim=1, keepdim=True) + 1e-12)
                        u = u + alpha * p
                        r_new = r - alpha * Mp
                        beta = torch.sum(r_new * r_new, dim=1, keepdim=True) / (r_dot_r + 1e-12)
                        p = r_new + beta * p
                        r = r_new

                    x_A_post = x_A_pri + v_A_pri * H_funcs.Ht(u.view(1, -1)).view(bs, N)
                    tr_term = torch.sum(s2 / (v_A_pri * s2 + delta2_0))
                    v_A_post = v_A_pri - (v_A_pri ** 2 / N) * tr_term
                    v_A_post = torch.clamp(v_A_post, min=variance_min, max=variance_max)
                    require_finite("x_A_post", x_A_post)
                    require_finite("v_A_post", v_A_post)

                    denom = 1.0 / v_A_post - 1.0 / v_A_pri
                    require_finite("Module-A extrinsic precision", denom)
                    denom = torch.clamp(denom, min=1e-8)
                    v_A_ext = 1.0 / denom
                    x_A_ext = v_A_ext * (
                            x_A_post / v_A_post - x_A_pri / v_A_pri
                    )
                    require_finite("x_A_ext", x_A_ext)
                    require_finite("v_A_ext", v_A_ext)

                    x_B_pri = x_A_ext * delta + x_A_ext_old * (1 - delta)
                    v_B_pri = v_A_ext * delta + v_A_ext_old * (1 - delta)
                    require_finite("x_B_pri", x_B_pri)
                    require_finite("v_B_pri", v_B_pri)
                    x_A_ext_old, v_A_ext_old = x_A_ext, v_A_ext
                    tau_r = v_B_pri * torch.ones_like(x_B_pri)

                    # ---------------------------------------------------------
                    # 2. Diffusion-prior estimator (Module B)
                    # ---------------------------------------------------------
                    bb = x_B_pri.detach() - x_0_hat.view(1, -1).view(bs, N)

                    def M_product2(vv):
                        vjp_input = (sigma_t) * vv.view(1, -1)  # (1,-1)  # 0.1
                        # PGDM
                        # vjp = vjp_input
                        # MMPS  asymmetric
                        vjp = torch.autograd.grad(
                            outputs=x_0_hat,
                            inputs=img,
                            grad_outputs=vjp_input.view_as(img),
                            retain_graph=True
                        )[0]
                        return v_B_pri * vv + vjp.view(1, -1).view(bs, N)

                    # Same one-step GMRES approximation as _step_vamp.
                    eps_gmres = 1e-8
                    r0 = bb
                    beta = torch.norm(r0, dim=1, keepdim=True)
                    require_finite("GMRES initial residual", beta)
                    v_0 = r0 / (beta + eps_gmres)
                    w = M_product2(v_0)
                    require_finite("GMRES matrix-vector product", w)
                    h_00 = torch.sum(v_0 * w, dim=1, keepdim=True)
                    w_perp = w - h_00 * v_0
                    h_10 = torch.norm(w_perp, dim=1, keepdim=True)
                    y_opt = (
                            (h_00 * beta)
                            / (h_00 ** 2 + h_10 ** 2 + eps_gmres)
                    )
                    vv = v_0 * y_opt
                    require_finite("GMRES solution", vv)

                    final_vjp_input = vv.view(1, -1).view_as(img)
                    score_y_given_x = torch.autograd.grad(
                        outputs=x_0_hat,
                        inputs=img,
                        grad_outputs=final_vjp_input,
                        retain_graph=True
                    )[0]

                nabla_r_xt = score_y_given_x
                del final_vjp_input

                nabla_xt_r = (a_t * x_0_hat - x_t) / b2_t + nabla_r_xt.view_as(img)
                x_hat1_raw = (x_t + b2_t * nabla_xt_r) / a_t
                correction_damping = getattr(model, "correction_damping", 1.0)
                x_hat1 = x_0_hat + correction_damping * (x_hat1_raw - x_0_hat)
                x_B_post = x_hat1.view(tau_r.shape)

                if i < 1 and Turbo >= 1:  # if i < 1 and Turbo >= 1:
                    vx = torch.randn_like(x_t)
                    hvp = torch.autograd.grad(
                        outputs=nabla_xt_r,
                        inputs=x_t,
                        grad_outputs=vx,
                        retain_graph=True
                    )[0]
                    trace_H = vx * hvp

                if i < Turbo:  # i < Turbo:
                    v_B_post_tensor = (b2_t / a_t ** 2) + (b2_t ** 2 / a_t ** 2) * trace_H
                else:
                    v_B_post_tensor = (b2_t / a_t ** 2) - (b2_t ** 2 / a_t ** 2) / ((a_t ** 2) * tau_r + b2_t)
                    v_B_post_tensor = v_B_post_tensor * getattr(model, "tau_inflation", 1.0)

                v_B_post_tensor = torch.clamp(
                    v_B_post_tensor, min=variance_min, max=variance_max
                )
                require_finite("x_B_post", x_B_post)
                require_finite("v_B_post_tensor", v_B_post_tensor)

                del nabla_r_xt

                v_B_post = torch.mean(v_B_post_tensor)
                # v_B_post = torch.clamp(
                #     v_B_post, min=variance_min, max=variance_max
                # )
                # v_B_pri = torch.clamp(
                #     v_B_pri, min=variance_min, max=variance_max
                # )
                require_finite("v_B_post", v_B_post)
                require_finite("v_B_pri before extrinsic update", v_B_pri)

                precision_B_ext = 1.0 / v_B_post - 1.0 / v_B_pri
                valid_B_ext = (
                        torch.isfinite(precision_B_ext)
                        & (precision_B_ext > precision_min)
                )
                safe_precision_B_ext = torch.where(
                    valid_B_ext, precision_B_ext, 1.0 / v_B_ext_old
                )
                v_B_ext = torch.clamp(
                    1.0 / safe_precision_B_ext,
                    min=variance_min,
                    max=variance_max
                )
                x_B_ext_candidate = v_B_ext * (
                        x_B_post / v_B_post - x_B_pri / v_B_pri
                )
                valid_B_mean = (
                        valid_B_ext & torch.isfinite(x_B_ext_candidate).all()
                )
                x_B_ext = torch.where(
                    valid_B_mean, x_B_ext_candidate, x_B_ext_old
                )
                require_finite("x_B_ext", x_B_ext)
                require_finite("v_B_ext", v_B_ext)

                x_B_ext_old = x_B_ext.detach()
                v_B_ext_old = v_B_ext.detach()

                with torch.no_grad():
                    x_A_pri = rho_A * x_B_ext.detach() + (1 - rho_A) * x_A_pri.detach()
                    v_A_pri = rho_A * v_B_ext.detach() + (1 - rho_A) * v_A_pri.detach()

            x_A_pri_temp = x_A_pri.detach()
            v_A_pri_temp = v_A_pri.detach()

            if loop_idx < num_cm_steps - 2:
                sigma_next = model.get_sigma(
                    loop_idx + 1, device=img.device, dtype=img.dtype
                )
                a_next, _ = model.vp_coeffs(sigma_next)
                noise_scale = torch.sqrt(
                    torch.clamp(sigma_next ** 2 - 0.002 ** 2, min=0.0)
                )
                img = (
                        a_next * x_hat1.detach()
                        + a_next * noise_scale * torch.randn_like(x_hat1)
                )
            else:
                img = x_hat1.detach()

            img = img.detach_()
            pbar.set_postfix({'sigma': sigma.item()}, refresh=False)
            if record:
                file_path = os.path.join(
                    save_root,
                    f"progress/x_cm_vamp_{str(loop_idx).zfill(4)}.png"
                )
                plt.imsave(file_path, clear_color(img))

            torch.cuda.empty_cache()

        return img

    def _step_Tvamp_cm(self,
                       model,
                       x_start,
                       measurement,
                       H_funcs,
                       noise_std,
                       record,
                       save_root,
                       alg_name,
                       obs_module=None):
        """
        CM-prior version of VAMP.

        This keeps the Module-A/Module-B structure from _step_vamp(...),
        but obtains the differentiable endpoint from the consistency model
        and advances between CM sigma levels with VP re-noising.
        """
        if obs_module is not None:
            raise NotImplementedError(
                "CM-VAMP with non-differentiable observation is not yet implemented. "
                "Use a GAMP-based algorithm instead (e.g., gamp_mm)."
            )
        # if alg_name != 'vamp':
        #     raise NotImplementedError("CM-VAMP is implemented only for algorithm name 'vamp'.")

        img = x_start
        num_cm_steps = len(model.sigmas)
        pbar = tqdm(list(range(num_cm_steps - 1)))
        model.model.requires_grad_(False)  #####

        delta2_0 = noise_std ** 2
        s2 = H_funcs._S_small ** 2
        delta = 0.75
        rho_A = 0.8
        variance_min = 1e-15
        variance_max = 1e15
        precision_min = 1.0 / variance_max

        def require_finite(name, value):
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f"CM-VAMP produced a non-finite value in {name} at CM step {loop_idx}."
                )

        # # BlockCS_H selects rows from an orthonormal DCT matrix, so A A^T = I.
        # if not torch.allclose(s2, torch.ones_like(s2), rtol=1e-4, atol=1e-5):
        #     print("s2:", s2)
        #     raise ValueError("CM-VAMP closed-form Module A requires a row-orthonormal matrix (A A^T = I).")

        v_A_pri_temp = 1 * x_start.new_ones(1)
        x_A_pri_temp = x_start.new_zeros(H_funcs.block_num, H_funcs.M)
        y = measurement.view(H_funcs.block_num, H_funcs.N)
        bs = H_funcs.block_num
        M = H_funcs.N
        N = H_funcs.M

        v_A_ext_old = x_start.new_ones(1)
        x_A_ext_old = x_start.new_zeros(H_funcs.block_num, H_funcs.M)
        x_B_ext_old = x_A_pri_temp.detach().clone()
        v_B_ext_old = torch.clamp(
            v_A_pri_temp.detach().clone(),
            min=variance_min,
            max=variance_max,
        )

        for loop_idx in pbar:
            sigma = model.get_sigma(loop_idx, device=img.device, dtype=img.dtype)

            img = img.requires_grad_()

            v_A_pri = v_A_pri_temp
            x_A_pri = x_A_pri_temp

            Turbo = 10

            for i in range(Turbo):
                with torch.no_grad():

                    b = y - H_funcs.H(x_A_pri.view(1, -1)).view(bs, M)

                    def M_product(p):
                        # M = v_A_pri * A * A^T + delta2_0 * I
                        temp = H_funcs.H(H_funcs.Ht(p.view(1, -1))).view(bs, M)
                        return v_A_pri * temp + delta2_0 * p
                        # CG M * u = b

                    u = torch.zeros_like(b)
                    r = b  # - M_product(u) = 0
                    p = r.clone()
                    cg_iters = 1
                    for _ in range(cg_iters):
                        Mp = M_product(p)
                        r_dot_r = torch.sum(r * r, dim=1, keepdim=True)
                        alpha = r_dot_r / (torch.sum(p * Mp, dim=1, keepdim=True) + 1e-12)
                        u = u + alpha * p
                        r_new = r - alpha * Mp
                        beta = torch.sum(r_new * r_new, dim=1, keepdim=True) / (r_dot_r + 1e-12)
                        p = r_new + beta * p
                        r = r_new

                    x_A_post = x_A_pri + v_A_pri * H_funcs.Ht(u.view(1, -1)).view(bs, N)
                    tr_term = torch.sum(s2 / (v_A_pri * s2 + delta2_0))
                    v_A_post = v_A_pri - (v_A_pri ** 2 / N) * tr_term
                    v_A_post = torch.clamp(v_A_post, min=variance_min, max=variance_max)
                    require_finite("x_A_post", x_A_post)
                    require_finite("v_A_post", v_A_post)

                    denom = 1.0 / v_A_post - 1.0 / v_A_pri
                    require_finite("Module-A extrinsic precision", denom)
                    denom = torch.clamp(denom, min=1e-8)
                    v_A_ext = 1.0 / denom
                    x_A_ext = v_A_ext * (
                            x_A_post / v_A_post - x_A_pri / v_A_pri
                    )
                    require_finite("x_A_ext", x_A_ext)
                    require_finite("v_A_ext", v_A_ext)

                    x_B_pri = x_A_ext * delta + x_A_ext_old * (1 - delta)
                    v_B_pri = v_A_ext * delta + v_A_ext_old * (1 - delta)
                    require_finite("x_B_pri", x_B_pri)
                    require_finite("v_B_pri", v_B_pri)
                    x_A_ext_old, v_A_ext_old = x_A_ext, v_A_ext
                    tau_r = v_B_pri * torch.ones_like(x_B_pri)

                    # Treat the Module-B input as r_B = x + N(0, tau_B I),
                    # and evaluate the CM endpoint at its matched noise level.
                    # tau_B = torch.clamp(
                    #     v_B_pri.detach(),
                    #     min=model.diffusion.sigma_min ** 2,
                    #     max=model.diffusion.sigma_max ** 2,
                    # )
                    tau_B = v_B_pri.detach()
                    sigma_B = torch.sqrt(tau_B)
                    a_B, _ = model.vp_coeffs(sigma_B)

                    # A local input graph is required only for the denoiser
                    # VJP; CM parameters remain frozen and sigma_B is fixed.
                    with torch.enable_grad():
                        r_B = (
                            x_B_pri.detach()
                            .view_as(img)
                            .clone()
                            .requires_grad_(True)
                        )
                        x_B_vp = a_B * r_B
                        x_hat1_graph = model.endpoint_from_vp(x_B_vp, sigma_B)

                        # Rademacher Hutchinson estimate:
                        # E[z * (J_D^T z)] = diag(J_D).
                        probe_B = torch.empty_like(r_B).bernoulli_(0.5)
                        probe_B = probe_B.mul_(2.0).sub_(1.0)
                        vjp_B = torch.autograd.grad(
                            outputs=x_hat1_graph,
                            inputs=r_B,
                            grad_outputs=probe_B,
                            retain_graph=False,
                            create_graph=False,
                            only_inputs=True,
                        )[0]
                        v_B_post_tensor = (
                                tau_B * probe_B * vjp_B
                        ).view_as(tau_r)

                    x_hat1 = x_hat1_graph.detach()

                x_B_post = x_hat1.view(tau_r.shape)

                # v_B_post_tensor = torch.clamp(
                #     v_B_post_tensor, min=variance_min, max=variance_max
                # )
                require_finite("x_B_post", x_B_post)
                require_finite("v_B_post_tensor", v_B_post_tensor)

                v_B_post = torch.mean(v_B_post_tensor)
                ####### Important: Clamp v_B_post and v_B_pri to avoid numerical issues
                v_B_post = torch.clamp(
                    v_B_post, min=variance_min, max=variance_max
                )
                v_B_pri = torch.clamp(
                    v_B_pri, min=variance_min, max=variance_max
                )
                require_finite("v_B_post", v_B_post)
                require_finite("v_B_pri before extrinsic update", v_B_pri)

                precision_B_ext = 1.0 / v_B_post - 1.0 / v_B_pri
                valid_B_ext = (
                        torch.isfinite(precision_B_ext)
                        & (precision_B_ext > precision_min)
                )
                safe_precision_B_ext = torch.where(
                    valid_B_ext, precision_B_ext, 1.0 / v_B_ext_old
                )
                v_B_ext = torch.clamp(
                    1.0 / safe_precision_B_ext,
                    min=variance_min,
                    max=variance_max
                )
                x_B_ext_candidate = v_B_ext * (
                        x_B_post / v_B_post - x_B_pri / v_B_pri
                )
                valid_B_mean = (
                        valid_B_ext & torch.isfinite(x_B_ext_candidate).all()
                )
                x_B_ext = torch.where(
                    valid_B_mean, x_B_ext_candidate, x_B_ext_old
                )
                require_finite("x_B_ext", x_B_ext)
                require_finite("v_B_ext", v_B_ext)

                x_B_ext_old = x_B_ext.detach()
                v_B_ext_old = v_B_ext.detach()

                with torch.no_grad():
                    x_A_pri = rho_A * x_B_ext.detach() + (1 - rho_A) * x_A_pri.detach()
                    v_A_pri = rho_A * v_B_ext.detach() + (1 - rho_A) * v_A_pri.detach()

            x_A_pri_temp = x_A_pri.detach()
            v_A_pri_temp = v_A_pri.detach()

            img = x_hat1.detach()

            pbar.set_postfix({'sigma': sigma.item()}, refresh=False)
            if record:
                file_path = os.path.join(
                    save_root,
                    f"progress/x_cm_vamp_{str(loop_idx).zfill(4)}.png"
                )
                plt.imsave(file_path, clear_color(img))

            torch.cuda.empty_cache()

        return img

    def _step_Tgamp_cm(self,
                       model,
                       x_start,
                       measurement,
                       H_funcs,
                       noise_std,
                       record,
                       save_root,
                       alg_name,
                       obs_module=None):
        """
        Turbo GAMP with a consistency-model denoiser.

        This keeps the original GAMP output/input-step structure, but directly
        treats the GAMP effective observation r as an AWGN-corrupted observation
        of x_0 and uses the consistency model as the input denoiser.

        There is no external diffusion/consistency sampling or VP re-noising.
        """

        img = x_start
        device = x_start.device

        self.old_x_0_listhat = []

        # model.sigmas is used only to retain the original number of outer rounds.
        # It is not used as an external consistency-sampling schedule.
        num_iterations = len(model.sigmas)
        pbar = tqdm(list(range(num_iterations - 1)))

        x_hat_temp = torch.zeros(H_funcs.block_num, H_funcs.M, device=device, )
        tau_x_temp = torch.ones(H_funcs.block_num, H_funcs.M, device=device, )
        s_temp = torch.zeros(H_funcs.block_num, H_funcs.N, device=device, )
        y = measurement.view(H_funcs.block_num, H_funcs.N, )

        noise_sigma = noise_std
        delta0 = noise_sigma ** 2
        bs = H_funcs.block_num
        M = H_funcs.N
        N = H_funcs.M
        rho_cm = 0.5  #
        # CM parameters remain frozen. Gradients are required only
        # with respect to the local denoiser input r_B.
        model.model.requires_grad_(False)

        for loop_idx in pbar:

            # Carry the GAMP states over between outer rounds.
            x_hat = x_hat_temp
            tau_x = tau_x_temp
            s = s_temp

            max_iter = 12

            for iter in range(max_iter):
                with torch.no_grad():
                    # ----------------------------------------------------------
                    # GAMP output step
                    # ----------------------------------------------------------
                    if iter == 0:
                        tau_p = H_funcs.H_squared(tau_x.view(1, -1)).view(bs, M)

                    p = (H_funcs.H(x_hat.view(1, -1)).view(bs, M) - s * tau_p)

                    if obs_module is not None:
                        z_hat, tau_z = obs_module.gamp_likelihood(p, tau_p, y, noise_sigma, )
                    else:
                        tau_p_safe = torch.clamp(tau_p, min=1e-15, )
                        tau_z = 1.0 / (1.0 / tau_p_safe + 1.0 / delta0)
                        z_hat = (p / tau_p_safe + y / delta0) * tau_z

                    tau_p_clamped = torch.clamp(tau_p, min=1e-10, )
                    s = (z_hat - p) / tau_p_clamped
                    tau_s = (1.0 - tau_z / tau_p_clamped) / tau_p_clamped
                    tau_s = torch.clamp(tau_s, min=1e-10, )

                    # ----------------------------------------------------------
                    # GAMP input effective observation
                    # ----------------------------------------------------------
                    A2_tau_s = H_funcs.Ht_squared(tau_s.view(1, -1)).view(bs, N)
                    tau_r = 1.0 / torch.clamp(A2_tau_s, min=1e-10, )
                    At_s = H_funcs.Ht(s.view(1, -1)).view(bs, N)
                    r = x_hat + tau_r * At_s

                # --------------------------------------------------------------
                # CM input denoiser
                # --------------------------------------------------------------
                # The GAMP effective variance is coordinate-wise, while the CM
                # accepts one scalar noise level. Use its spatial average.
                tau_B = torch.mean(tau_r).detach()
                tau_B = torch.clamp(tau_B, min=1e-15, max=1e8, )
                sigma_B = torch.sqrt(tau_B)
                a_B, _ = model.vp_coeffs(sigma_B)

                with torch.enable_grad():

                    r_B = (r.detach().view_as(img).clone().requires_grad_(True))
                    x_B_vp = a_B * r_B
                    x_hat_graph = model.endpoint_from_vp(x_B_vp, sigma_B, )
                    # Rademacher Hutchinson estimate:
                    #
                    #     diag(J_D) ≈ z ⊙ J_D^T z.
                    #
                    # This provides the coordinate-wise effective variance
                    # required by the GAMP input step.
                    probe_B = torch.empty_like(r_B).bernoulli_(0.5)
                    probe_B = (probe_B.mul_(2.0).sub_(1.0))
                    vjp_B = torch.autograd.grad(
                        outputs=x_hat_graph,
                        inputs=r_B,
                        grad_outputs=probe_B,
                        retain_graph=False,
                        create_graph=False,
                        only_inputs=True,
                    )[0]
                    tau_x_graph = (tau_B * probe_B * vjp_B)
                    tau_x_scalar = torch.clamp(torch.mean(tau_x_graph), min=1e-10, max=1e8, )
                    tau_x_graph = torch.full_like(tau_x_graph, tau_x_scalar, )

                # GAMP input posterior mean.
                # Raw CM input-step estimates.
                x_hat_new = x_hat_graph.detach().view(bs, N)
                tau_x_new = tau_x_graph.detach().view(bs, N)
                tau_x_new = torch.clamp(tau_x_new, min=1e-10, max=1e8, )
                # Joint damping of the input-step mean and variance.
                x_hat = rho_cm * x_hat_new + (1.0 - rho_cm) * x_hat
                tau_x = rho_cm * tau_x_new + (1.0 - rho_cm) * tau_x
                tau_x = torch.clamp(tau_x, min=1e-10, max=1e8, )

                # Prepare tau_p for the next GAMP iteration.
                tau_p = H_funcs.H_squared(tau_x.view(1, -1)).view(bs, M)

                del (p, z_hat, tau_p_clamped, tau_s, A2_tau_s, At_s, r, r_B, x_B_vp, x_hat_graph,
                     probe_B, vjp_B, tau_x_graph,)

            # Carry all original GAMP states to the next outer round.
            x_hat_temp = x_hat.detach()
            tau_x_temp = tau_x.detach()
            s_temp = s.detach()

            # No external consistency sampling or VP re-noising.
            img = x_hat.view_as(img).detach()
            pbar.set_postfix(
                {
                    'sigma_B': sigma_B.item(),
                    'maxiter': max_iter,
                },
                refresh=False,
            )
            if record:
                file_path = os.path.join(
                    save_root,
                    f"progress/x_Tgamp_cm_{str(loop_idx).zfill(4)}.png",
                )
                plt.imsave(
                    file_path,
                    clear_color(img),
                )

            torch.cuda.empty_cache()

        return img

    def _step_cm_gamp_ps(self,
                         model,
                         x_start,
                         measurement,
                         H_funcs,
                         noise_std,
                         record,
                         save_root,
                         alg_name,
                         obs_module=None):
        """
        Gaussian precision-fused CM-GAMP posterior sampling.
        At each consistency-sampling time, GAMP provides an effective AWGN
        observation
            r = x_0 + N(0, tau_r I),
        while the current VP diffusion state provides
            q_t = x_t / a_t = x_0 + N(0, sigma_t^2 I).
        The two Gaussian observations are fused and passed to the consistency
        model as a single equivalent AWGN denoising problem. The outer
        consistency-sampling and VP re-noising procedure is retained.
        """

        img = x_start
        device = x_start.device

        self.old_x_0_listhat = []

        num_cm_steps = len(model.sigmas)
        pbar = tqdm(list(range(num_cm_steps - 1)))

        x_hat_temp = torch.zeros(H_funcs.block_num, H_funcs.M, device=device, )
        tau_x_temp = torch.ones(H_funcs.block_num, H_funcs.M, device=device, )
        s_temp = torch.zeros(H_funcs.block_num, H_funcs.N, device=device, )
        y = measurement.view(H_funcs.block_num, H_funcs.N, )

        noise_sigma = noise_std
        delta0 = noise_sigma ** 2
        bs = H_funcs.block_num
        M = H_funcs.N
        N = H_funcs.M
        rho_cm = 0.7
        # CM parameters are frozen. Gradients are required only with respect to the fused CM input for the Hutchinson VJP.
        model.model.requires_grad_(False)

        for loop_idx in pbar:
            # --------------------------------------------------------------
            # Current diffusion observation
            # --------------------------------------------------------------
            sigma = model.get_sigma(loop_idx, device=img.device, dtype=img.dtype, )
            a_t, b2_t = model.vp_coeffs(sigma)
            # x_t = a_t x_0 + b_t epsilon
            x_t = img.detach()

            # Normalize the VP state into an equivalent VE observation:
            # q_t = x_t / a_t = x_0 + N(0, sigma_t^2 I).
            q_t = x_t / a_t
            sigma_t2 = torch.clamp(b2_t / (a_t ** 2), min=1e-15, max=1e8, )

            # Carry the GAMP states across consistency-sampling steps.
            x_hat = x_hat_temp
            tau_x = tau_x_temp
            s = s_temp

            max_iter = 3

            for iter in range(max_iter):
                with torch.no_grad():
                    # ------------------------------------------------------
                    # GAMP output step
                    # ------------------------------------------------------
                    if iter == 0:
                        tau_p = H_funcs.H_squared(tau_x.view(1, -1)).view(bs, M)

                    p = H_funcs.H(x_hat.view(1, -1)).view(bs, M) - s * tau_p

                    if obs_module is not None:
                        z_hat, tau_z = obs_module.gamp_likelihood(p, tau_p, y, noise_sigma, )
                    else:
                        tau_p_safe = torch.clamp(tau_p, min=1e-15, )
                        tau_z = 1.0 / (1.0 / tau_p_safe + 1.0 / delta0)
                        z_hat = (p / tau_p_safe + y / delta0) * tau_z

                    tau_p_clamped = torch.clamp(tau_p, min=1e-10, )
                    s = (z_hat - p) / tau_p_clamped
                    tau_s = (1.0 - tau_z / tau_p_clamped) / tau_p_clamped
                    tau_s = torch.clamp(tau_s, min=1e-10, )

                    # ------------------------------------------------------
                    # GAMP input effective observation
                    # ------------------------------------------------------
                    A2_tau_s = H_funcs.Ht_squared(tau_s.view(1, -1)).view(bs, N)
                    tau_r = 1.0 / torch.clamp(A2_tau_s, min=1e-10, )
                    At_s = H_funcs.Ht(s.view(1, -1)).view(bs, N)
                    r = x_hat + tau_r * At_s

                # ----------------------------------------------------------
                # Gaussian precision fusion
                # ----------------------------------------------------------
                # The proposition assumes a scalar tau_r. Since GAMP produces
                # coordinate-wise tau_r, use its spatial average before fusion.
                tau_r_scalar = torch.clamp(torch.mean(tau_r).detach(), min=1e-15, max=1e8, )

                # Fused variance: v_c = (tau_r^{-1} + sigma_t^{-2})^{-1}.
                tau_B = 1.0 / (1.0 / tau_r_scalar + 1.0 / sigma_t2)
                tau_B = torch.clamp(tau_B, min=1e-15, max=1e8, )
                # Fused mean: u_c = v_c (r / tau_r + q_t / sigma_t^2).
                r_fused = tau_B * (r.detach() / tau_r_scalar + q_t.view(bs, N) / sigma_t2)

                # ----------------------------------------------------------
                # CM denoiser on the fused AWGN observation
                # ----------------------------------------------------------
                # r_fused is in VE/AWGN coordinates:  r_fused = x_0 + N(0, tau_B I).
                # Convert it to the VP coordinates expected by the CM using
                # the coefficient associated with the fused noise variance.
                sigma_B = torch.sqrt(tau_B)
                a_B, _ = model.vp_coeffs(sigma_B)
                with torch.enable_grad():
                    r_B = (r_fused.view_as(img).clone().requires_grad_(True))
                    # Correct scale conversion:
                    # VE observation r_B -> VP state a_B * r_B.
                    # a_B corresponds to sigma_B = sqrt(tau_B), not to the
                    # outer diffusion coefficient a_t.
                    x_B_vp = a_B * r_B
                    x_hat_graph = model.endpoint_from_vp(x_B_vp, sigma_B, )

                    # Hutchinson estimate of the CM Jacobian response.
                    probe_B = torch.empty_like(r_B).bernoulli_(0.5)
                    probe_B = probe_B.mul_(2.0).sub_(1.0)
                    vjp_B = torch.autograd.grad(
                        outputs=x_hat_graph,
                        inputs=r_B,
                        grad_outputs=probe_B,
                        retain_graph=False,
                        create_graph=False,
                        only_inputs=True,
                    )[0]
                    tau_x_graph = (tau_B * probe_B * vjp_B)
                    # Use the scalar average response, consistent with the
                    # scalar Gaussian fusion approximation.
                    tau_x_scalar = torch.clamp(torch.mean(tau_x_graph), min=1e-10, max=1e8, )
                    tau_x_graph = torch.full_like(tau_x_graph, tau_x_scalar, )

                # Raw CM posterior-moment estimates.
                x_hat1 = x_hat_graph.detach()
                x_hat_new = x_hat1.view(bs, N)
                tau_x_new = tau_x_graph.detach().view(bs, N)
                tau_x_new = torch.clamp(tau_x_new, min=1e-10, max=1e8, )

                # Preserve the original joint mean/variance damping.
                with torch.no_grad():
                    x_hat = rho_cm * x_hat_new.detach() + (1.0 - rho_cm) * x_hat.detach()
                    tau_x = rho_cm * tau_x_new.detach() + (1.0 - rho_cm) * tau_x.detach()
                    tau_x = torch.clamp(tau_x, min=1e-10, max=1e8, )

                # Prepare the next GAMP output step.
                tau_p = H_funcs.H_squared(tau_x.view(1, -1)).view(bs, M)

                del (p, z_hat, tau_p_clamped, tau_s, A2_tau_s, At_s, r, r_fused, r_B, x_B_vp, x_hat_graph, probe_B,
                     vjp_B, tau_x_graph,)

            # Carry the GAMP states to the next diffusion time.
            x_hat_temp = x_hat.detach()
            tau_x_temp = tau_x.detach()
            s_temp = s.detach()

            # --------------------------------------------------------------
            # Outer consistency sampling / VP re-noising
            # --------------------------------------------------------------
            if loop_idx < num_cm_steps - 2:
                sigma_next = model.get_sigma(loop_idx + 1, device=img.device, dtype=img.dtype, )
                a_next, _ = model.vp_coeffs(sigma_next)
                noise_scale = torch.sqrt(torch.clamp(sigma_next ** 2 - 0.002 ** 2, min=0.0, ))
                img = a_next * x_hat1 + a_next * noise_scale * torch.randn_like(x_hat1)
            else:
                img = x_hat1

            img = img.detach()
            pbar.set_postfix(
                {
                    'sigma': sigma.item(),
                    'sigma_B': sigma_B.item(),
                    'maxiter': max_iter,
                },
                refresh=False,
            )
            if record:
                file_path = os.path.join(save_root, f"progress/x_cm_gamp_ps_{str(loop_idx).zfill(4)}.png", )
                plt.imsave(file_path, clear_color(img), )

            torch.cuda.empty_cache()

        return img

    def p_sample_loop_cs(self,
                         model,
                         x_start,
                         measurement,
                         measurement_cond_fn,
                         H_funcs=None,
                         noise_std=0.1,
                         config=None,
                         record=False,
                         save_root=None,
                         diffusion_sampler='mmps',
                         obs_module=None):

        # alg_name
        alg_config = config.get('algorithm', {})
        alg_name = alg_config.get('name', 'mmps')
        prior_type = alg_config.get('prior_type', None)
        # (Algorithm Dispatcher)
        if prior_type == 'openai_cm':
            if alg_name == 'gamp_mm':
                img = self._step_gamp_cm(model, x_start, measurement, H_funcs, noise_std, record, save_root, alg_name,
                                         obs_module=obs_module)
            elif alg_name == 'vamp':
                img = self._step_vamp_cm(model, x_start, measurement, H_funcs, noise_std, record, save_root, alg_name,
                                         obs_module=obs_module)
            elif alg_name == 'Tvamp':
                img = self._step_Tvamp_cm(model, x_start, measurement, H_funcs, noise_std, record, save_root, alg_name,
                                          obs_module=obs_module)
            elif alg_name == 'Tgamp':
                img = self._step_Tgamp_cm(model, x_start, measurement, H_funcs, noise_std, record, save_root, alg_name,
                                          obs_module=obs_module)
            elif alg_name == 'cm_gamp_ps':
                img = self._step_cm_gamp_ps(model, x_start, measurement, H_funcs, noise_std, record, save_root,
                                            alg_name,
                                            obs_module=obs_module)
            elif alg_name == 'cm_mmps':
                img = self._step_cm_mmps(model, x_start, measurement, H_funcs, noise_std, record, save_root, alg_name,
                                         obs_module=obs_module)
            else:
                raise NotImplementedError("CM prior supports only 'gamp_mm', 'vamp', 'Tvamp', 'Tgamp', and 'cm_mmps'.")
        elif 'gamp' in alg_name:
            img = self._step_gamp(model, x_start, measurement, H_funcs, noise_std, record, save_root, alg_name,
                                  diffusion_sampler, obs_module=obs_module)
        elif 'vamp' in alg_name:
            img = self._step_vamp(model, x_start, measurement, H_funcs, noise_std, record, save_root, alg_name,
                                  diffusion_sampler, obs_module=obs_module)
        else:
            img = self._step_mmps(model, x_start, measurement, H_funcs, noise_std, record, save_root, alg_name,
                                  diffusion_sampler, obs_module=obs_module)

        return img

    def p_sample(self, model, x, t):
        raise NotImplementedError

    def p_mean_variance(self, model, x, t):
        model_output = model(x, self._scale_timesteps(t))

        # In the case of "learned" variance, model will give twice channels.
        if model_output.shape[1] == 2 * x.shape[1]:
            model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        else:
            # The name of variable is wrong.
            # This will just provide shape information, and
            # will not be used for calculating something important in variance.
            model_var_values = model_output

        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_variance, model_log_variance = self.var_processor.get_variance(model_var_values, t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {'mean': model_mean,
                'variance': model_variance,
                'log_variance': model_log_variance,
                'pred_xstart': pred_xstart}

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
            self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
            self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


@register_sampler(name='ddpm')
class DDPM(SpacedDiffusion):
    def p_sample(self, model, x, t):
        out = self.p_mean_variance(model, x, t)
        sample = out['mean']

        noise = torch.randn_like(x)
        if t != 0:  # no noise when t == 0
            sample += torch.exp(0.5 * out['log_variance']) * noise

        return {'sample': sample, 'pred_xstart': out['pred_xstart'], 'log_variance': out['log_variance']}


@register_sampler(name="ddim")
class DDIM(SpacedDiffusion):
    def __init__(self, eta=1.0, **kwargs):
        super().__init__(**kwargs)

        self.eta = eta

    def p_sample(self, model, x, t):
        alpha_t = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_s = extract_and_expand(self.alphas_cumprod_prev, t, x)

        if t == 0:
            sigma = 0
        else:
            sigma = (
                    self.eta
                    * torch.sqrt((1 - alpha_s) / (1 - alpha_t))
                    * torch.sqrt(1 - alpha_t / alpha_s)
            )

        out = self.p_mean_variance(model, x, t)

        x_0 = out["pred_xstart"]
        eps = (x - torch.sqrt(alpha_t) * x_0) / torch.sqrt(1 - alpha_t)
        x_s = (
                torch.sqrt(alpha_s) * x_0
                + torch.sqrt(1 - alpha_s - sigma ** 2) * eps
                + sigma * torch.randn_like(x_0)
        )

        scale = torch.sqrt(alpha_s) - torch.sqrt(1 - alpha_s - sigma ** 2) * torch.sqrt(
            alpha_t
        ) / torch.sqrt(1 - alpha_t)

        return {
            "sample": x_s,
            "pred_xstart": x_0,
            "alpha": alpha_s,
            "alpha_prev": alpha_t,
            "scale": scale,
        }


@register_sampler(name='dpmsolver++')
class DPM_Solver_plus(SpacedDiffusion):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def p_sample(self, model, x, t, idx=None, total_steps=None):
        out = self.p_mean_variance(model, x, t)
        x_0 = out["pred_xstart"]
        return {
            "pred_xstart": x_0
        }

    def clear_history(self):
        self.old_x_0_list = []


# =================
# Helper functions
# =================

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
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
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


# ================
# Helper function
# ================

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.float):
        array = torch.tensor([array])

    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
