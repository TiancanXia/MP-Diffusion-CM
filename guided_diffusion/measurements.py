'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

import math
from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m

import numpy as np
# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)

@register_operator(name='CS')
class BlockCS_H(LinearOperator):
    def __init__(self, img_dim=196608, block_num=16, block_dim=12288//1, compressed_dim = 1323//1, device='cuda'):
        # 12288 1200/3072/588/6075/1323
        """
        Args:
            img_dim:  (3*256*256 = 196608)
            block_num:  (64/16)
            block_dim:  M (3072)
            compressed_dim:  N (768)
            device: 
        """
        self.img_dim = img_dim
        self.block_num = block_num
        self.M = block_dim
        self.N = compressed_dim
        self.device = device

        # 1. Construct DCT Matrix W (M x M)
        # DCT-II Definition
        # W_ij = c_i * cos(pi * i * (2j + 1) / (2M))
        # c_0 = sqrt(1/M), c_k = sqrt(2/M)
        print("Constructing DCT matrix...")
        i = torch.arange(self.M, device=device).unsqueeze(1)  # 0..M-1 (M, 1)
        j = torch.arange(self.M, device=device).unsqueeze(0)  # 0..M-1 (1, M)

        # DCT coefficients
        c = torch.ones(self.M, 1, device=device) * np.sqrt(2 / self.M)
        c[0, 0] = np.sqrt(1 / self.M)

        # W[i, j]
        W = c * torch.cos(np.pi * i * (2 * j + 1) / (2 * self.M))

        # 2. Theta (M x M, diagnol)
        print("Constructing Random Sign matrix...")
        signs = torch.sign(torch.randn(self.M, device=device))
        # A = S * W * Theta
        W_Theta = W * signs.unsqueeze(0)  # Broadcast: (M, M) * (1, M)

        # 3. S (N x M)
        print("Constructing Selection matrix...")
        perm = torch.randperm(self.M, device=device)
        selection_indices = perm[:self.N]

        # 4. A (N x M)
        # A = S * (W * Theta) -> Select W_Theta rows
        self.A = W_Theta[selection_indices, :]  # Shape: (768, 3072)
        #
        # torch.manual_seed(333)

        # Gaussian random matrix A ~ N(0, 1/M)
        # M = compressed_dim, variance 1/M
        # print("Construction of A...")
        # A = torch.randn(compressed_dim, block_dim)

        # 5. A  SVD decomposition
        # A = U_small * S_small * V_small.T
        # print("Computing SVD of A...")
        #
        # _, _, self.A = torch.linalg.svd(A, full_matrices = False)

        # Use some=False to obtain V (M x M) # torch.svd(self.A, some=False)
        # self._U_small, self._S_small, self._Vh = torch.svd_lowrank(self.A, q=min(self.A.shape[-2:]))
        # self._V_small = self._Vh.t()
        #
        # # _V_small (M, M), _U_small (N, N), _S_small (N,)
        # # H_functions ^ T
        # self._Vt_small = self._V_small.t()  # (M, M)
        # self._Ut_small = self._U_small.t()  # (N, N)
        #
        # # For stability
        ZERO = 1e-3
        self._S_small = torch.linalg.svdvals(self.A)
        self._S_small[self._S_small < ZERO] = 0

        # 6. A2 = |A|^2  ########################
        print("Computing the square of A...")
        self.A2 = torch.abs(self.A) ** 2
        self.A2_t = self.A2.t()

        # # complete singular vectors (repeat block_num times)
        # self._singulars_full = self._S_small.repeat(self.block_num)

    def _prepare_input(self, vec):
        """ (Batch, img_dim) """
        if vec.dim() == 4:  #  (B, C, H, W) 
            return vec.reshape(vec.shape[0], -1)
        elif vec.dim() == 2:
            return vec
        else:
            raise ValueError(f"Unsupported input shape: {vec.shape}")

    def _block_matmul(self, mat, vec, input_dim, output_dim):
        """
        vec: (Batch, Total_Input_Dim) - reshape
        mat: (Small_Output_Dim, Small_Input_Dim) 
        Returns: (Batch, Total_Output_Dim)
        """
        b, total_input = vec.shape

        # check
        if total_input != self.block_num * input_dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.block_num * input_dim}, got {total_input}")

        # 1. Reshape input to (Batch, Block_Num, Block_Dim)
        vec_reshaped = vec.view(b, self.block_num, input_dim)

        # 2. Matmul: (Batch, Block_Num, In) x (Out, In)^T -> (Batch, Block_Num, Out)
        # functional.linear: y = xA^T
        # out_reshaped = torch.nn.functional.linear(vec_reshaped, mat)
        out_reshaped = torch.matmul(vec_reshaped, mat.transpose(0, 1))

        # 3. Flatten back
        return out_reshaped.view(b, self.block_num * output_dim)

    def forward(self, vec, **kwargs):

        input_size = int(np.sqrt(self.block_num * self.N / 3))
        temp = self.H(vec)
        return temp.reshape(1, 3, input_size, input_size)

    def H(self, vec):
        """
        A
        Input: (Batch, C, H, W) or (Batch, 196608)
        Output: (Batch, 49152)
        """
        temp = self._prepare_input(vec)
        return self._block_matmul(self.A, temp, self.M, self.N)

    def Ht(self, vec):
        """
        A ^ H
        Input: (Batch, 49152) or (Batch, C, H, W) H*W*C = 49152
        Output: (Batch, 196608)
        """
        temp = self._prepare_input(vec)
        return self._block_matmul(self.A.t(), temp, self.N, self.M)

    def transpose(self, data):
        return data

    def ortho_project(self, data):
        return data

    def project(self, data):
        return data

    def H_squared(self, vec):
        """
         A2 = |A|^2
        Input: (Batch, C, H, W) or (Batch, 196608)
        Output: (Batch, 49152)
        """
        temp = self._prepare_input(vec)
        return self._block_matmul(self.A2, temp, self.M, self.N)
    
    def Ht_squared(self, vec):
        """
         A2 ^ H
        Input: (Batch, 49152) or (Batch, C, H, W) H*W*C = 49152
        Output: (Batch, 196608)
        """
        temp = self._prepare_input(vec)
        return self._block_matmul(self.A2_t, temp, self.N, self.M)

@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)


# ==========================================
# Non-differentiable element-wise observation
# ==========================================

class NonDiffObservation(ABC):
    """Base class for non-differentiable element-wise observation y = g(Ax + n).

    GAMP decouples the output step from the input step: the likelihood
    (output step) handles g while the prior (input step) only sees the
    diffusion model. This class encapsulates the output-step logic so
    that _step_gamp can call likelihood() instead of the hardcoded AWGN
    formula.
    """

    @abstractmethod
    def forward(self, z):
        """Apply g(z) element-wise.

        Args:
            z: (bs, M) tensor of noisy linear measurements.

        Returns:
            y: (bs, M) tensor after element-wise non-differentiable g.
        """
        pass

    @abstractmethod
    def gamp_likelihood(self, p, tau_p, y, noise_sigma):
        """GAMP output-step posterior for y = g(z + n), n ~ N(0, sigma^2).

        Given prior  z_i ~ N(p_i, tau_p_i)  and observation y_i = g(z_i + n_i),
        compute the scalar posterior mean and variance element-wise.

        Args:
            p:          (bs, M) prior mean.
            tau_p:      (bs, M) prior variance.
            y:          (bs, M) observed (quantized) values.
            noise_sigma: scalar, std of pre-g noise n.

        Returns:
            z_hat: (bs, M) posterior mean  E[z | y].
            tau_z: (bs, M) posterior variance Var[z | y].
        """
        pass


class QuantizedObservation(NonDiffObservation):
    """Uniform quantizer:  y = delta * round((z + n) / delta).

    The posterior mean / variance are computed by numerical integration
    on a fixed grid in the standard-normal (u) space.
    """

    def __init__(self, step_size, n_integration=51):
        self.delta = step_size
        self.n_grid = n_integration

    def forward(self, z):
        return self.delta * torch.round(z / self.delta)

    def gamp_likelihood(self, p, tau_p, y, noise_sigma):
        """
        Numerical integration via grid in u-space (u ~ N(0,1)):
            z_j = p + sqrt(tau_p) * u_j
        prior     ∝ exp(-u_j^2 / 2)
        likelihood = Phi((y + 0.5*delta - z_j) / sigma)
                   - Phi((y - 0.5*delta - z_j) / sigma)
        """
        bs, M = p.shape
        device = p.device
        delta = self.delta
        sigma_n = noise_sigma

        # -- integration grid in u-space --
        u_max = 5.0
        u = torch.linspace(-u_max, u_max, self.n_grid, device=device, dtype=p.dtype)

        sqrt_tau_p = torch.sqrt(torch.clamp(tau_p, min=1e-15))
        z_grid = p.unsqueeze(-1) + sqrt_tau_p.unsqueeze(-1) * u  # (bs, M, n_grid)

        # -- prior (Gaussian in u-space) --
        log_prior = -0.5 * u.pow(2)  # (n_grid,)

        # -- likelihood: difference of two normal CDFs --
        y_exp = y.unsqueeze(-1)
        upper = (y_exp + 0.5 * delta - z_grid) / sigma_n
        lower = (y_exp - 0.5 * delta - z_grid) / sigma_n

        cdf_upper = 0.5 * (1.0 + torch.erf(upper / math.sqrt(2.0)))
        cdf_lower = 0.5 * (1.0 + torch.erf(lower / math.sqrt(2.0)))

        likelihood = cdf_upper - cdf_lower
        likelihood = torch.clamp(likelihood, min=1e-15)

        # -- unnormalized log-posterior --
        log_likelihood = torch.log(likelihood)
        log_post = log_prior + log_likelihood  # broadcasts: (n_grid,) + (bs, M, n_grid)

        # -- normalize (log-sum-exp trick) --
        log_post_max = torch.max(log_post, dim=-1, keepdim=True)[0]
        post = torch.exp(log_post - log_post_max)
        post_sum = torch.sum(post, dim=-1, keepdim=True)
        post = post / post_sum

        # -- posterior mean --
        z_hat = torch.sum(z_grid * post, dim=-1)

        # -- posterior variance --
        z_diff = z_grid - z_hat.unsqueeze(-1)
        tau_z = torch.sum(z_diff.pow(2) * post, dim=-1)
        tau_z = torch.clamp(tau_z, min=1e-15, max=1e8)

        return z_hat, tau_z