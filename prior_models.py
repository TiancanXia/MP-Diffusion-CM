import os
import sys

import torch


class ConsistencyPrior:
    """Thin differentiable wrapper around OpenAI Consistency Models.

    The GAMP-MM code keeps an internal VP-style state

        x_vp = a(sigma) * x0 + b(sigma) * eps,

    while OpenAI CM expects the EDM/VE-style input

        x_edm = x0 + sigma * eps.

    This wrapper performs only that coordinate conversion and calls
    diffusion.denoise(...). It deliberately does not call karras_sample(),
    because GAMP-MM needs x0_hat to remain differentiable w.r.t. x_vp.
    """

    def __init__(
        self,
        model,
        diffusion,
        sigmas,
        clip_denoised=True,
        tau_inflation=1.0,
        correction_damping=1.0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.sigmas = sigmas
        self.clip_denoised = clip_denoised
        self.tau_inflation = tau_inflation
        self.correction_damping = correction_damping
        self.in_channels = 3

    @property
    def num_steps(self):
        return len(self.sigmas)

    def get_sigma(self, i, device, dtype):
        return self.sigmas[i].to(device=device, dtype=dtype)

    def vp_coeffs(self, sigma):
        a = 1.0 / torch.sqrt(1.0 + sigma**2)
        b2 = sigma**2 / (1.0 + sigma**2)
        return a, b2

    def endpoint_from_vp(self, x_vp, sigma):
        a, _ = self.vp_coeffs(sigma)
        x_edm = x_vp / a
        sigma_batch = sigma.reshape(1).expand(x_vp.shape[0])
        _, x0_hat = self.diffusion.denoise(self.model, x_edm, sigma_batch)
        if self.clip_denoised:
            x0_hat = x0_hat.clamp(-1, 1)
        return x0_hat


def _resolve_repo_root(repo_root):
    if os.path.isabs(repo_root):
        return repo_root

    cwd_path = os.path.abspath(repo_root)
    if os.path.isdir(cwd_path):
        return cwd_path

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    project_path = os.path.join(project_root, repo_root)
    return os.path.abspath(project_path)


def _extract_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model", "ema"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def create_openai_cm_prior(prior_config, device):
    """Create a differentiable OpenAI CM prior for Algorithm 1.

    Expected config fields:

        prior:
          type: openai_cm
          repo_root: consistency_models-main
          checkpoint: consistency_models-main/checkpoints/cd_cat256_l2.pt
          sigma_min: 0.002
          sigma_max: 80.0
          num_steps: 8
          base_num_steps: 40
          timestep_indices: [0, 6, 11, 17, 22, 28, 33, 39]
          tau_inflation: 1.0
          correction_damping: 0.25
          clip_denoised: True
    """

    repo_root = prior_config.get("repo_root", "consistency_models-main")
    repo_root = _resolve_repo_root(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from cm.karras_diffusion import get_sigmas_karras
    from cm.script_util import create_model_and_diffusion

    checkpoint = prior_config["checkpoint"]
    checkpoint = os.path.abspath(checkpoint)

    sigma_min = prior_config.get("sigma_min", 0.002)
    sigma_max = prior_config.get("sigma_max", 80.0)
    num_steps = prior_config.get("num_steps", 8)

    model, diffusion = create_model_and_diffusion(
        image_size=prior_config.get("image_size", 256),
        class_cond=False,
        learn_sigma=False,
        num_channels=256,
        num_res_blocks=2,
        channel_mult="",
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=False,
        resblock_updown=True,
        use_fp16=prior_config.get("use_fp16", False),
        use_new_attention_order=False,
        weight_schedule="uniform",
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        distillation=True,
    )

    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(_extract_state_dict(state))
    model.to(device)
    model.eval()

    if prior_config.get("use_fp16", False):
        model.convert_to_fp16()

    # Old direct n-step Karras schedule:
    # sigmas = get_sigmas_karras(
    #     num_steps,
    #     sigma_min,
    #     sigma_max,
    #     rho=prior_config.get("rho", 7.0),
    #     device=device,
    # )
    # sigmas = sigmas[:-1]

    base_num_steps = prior_config.get("base_num_steps", 40)
    timestep_indices = prior_config.get("timestep_indices", None)
    if timestep_indices is None:
        if num_steps <= 1:
            timestep_indices = [0]
        else:
            timestep_indices = [
                round(i * (base_num_steps - 1) / (num_steps - 1))
                for i in range(num_steps)
            ]

    base_sigmas = get_sigmas_karras(
        base_num_steps,
        sigma_min,
        sigma_max,
        rho=prior_config.get("rho", 7.0),
        device=device,
    )
    base_sigmas = base_sigmas[:-1]
    timestep_indices = torch.tensor(timestep_indices, dtype=torch.long, device=device)
    sigmas = base_sigmas[timestep_indices]

    return ConsistencyPrior(
        model=model,
        diffusion=diffusion,
        sigmas=sigmas,
        clip_denoised=prior_config.get("clip_denoised", True),
        tau_inflation=prior_config.get("tau_inflation", 1.0),
        correction_damping=prior_config.get("correction_damping", 1.0),
    )
