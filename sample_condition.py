from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger

import time
import random
import numpy as np

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

# for debug
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='configs/model_config.yaml', type=str)
    parser.add_argument('--diffusion_config', default='configs/cm_diffusion_config.yaml', type=str)
    parser.add_argument('--task_config', default='configs/CS_cm_config.yaml', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default = 0)
    args = parser.parse_args()
   
    seed_torch(args.seed)
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model / prior
    prior_config = diffusion_config.get("prior", None)
    if prior_config is not None and prior_config.get("type") == "openai_cm":
        from prior_models import create_openai_cm_prior
        model = create_openai_cm_prior(prior_config, device)
        logger.info("Using OpenAI Consistency Model prior.")
    else:
        model = create_model(**model_config)
        model = model.to(device)
        model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']} / Algorithm : {task_config['algorithm']['name']}")
   
    # Load diffusion sampler
    sampler_config = dict(diffusion_config)
    sampler_config.pop("prior", None)
    sampler = create_sampler(**sampler_config)
    sample_fn = partial(sampler.p_sample_loop_cs, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    # Do Inference
    start_time = time.time()
    psnr_results = []
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        # Exception In case of inpainging,
        if measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
         

        if measure_config['operator']['name'] == 'CS':
            from guided_diffusion.measurements import BlockCS_H
            H_funcs = BlockCS_H(block_num=16, device=device)
            y_x = H_funcs.H(ref_img)
            noise = get_noise(**measure_config['noise'])
            y_noisy = y_x + noise.sigma * torch.randn_like(y_x)

            # Non-differentiable element-wise observation (e.g., quantization)
            obs_config = measure_config.get('observation', None)
            if obs_config is not None and obs_config.get('type') == 'quantization':
                from guided_diffusion.measurements import QuantizedObservation
                obs_module = QuantizedObservation(step_size=obs_config['step_size'])
                y_xn = obs_module.forward(y_noisy)
            else:
                obs_module = None
                y_xn = y_noisy
            ratio = 1
        else:
            H_funcs = None
            y_xn = None
            noise = None
            obs_module = None

        # Sampling
        DPS_start_time = time.time()
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        if measure_config['operator']['name'] == 'CS':
            sample = sample_fn(x_start=x_start, measurement=y_xn, H_funcs=H_funcs, noise_std=noise.sigma, config=task_config, record=True, save_root=out_path, diffusion_sampler=diffusion_config['sampler'], obs_module=obs_module)
        else:
            sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)

        DPS_end_time = time.time()
        print('DPS running time: {}'.format(DPS_end_time - DPS_start_time))
        psnr = peak_signal_noise_ratio(ref_img.cpu().numpy(),sample.cpu().numpy())
        psnr_results.append([psnr])
        print('PSNR: {}'.format(psnr))

        if measure_config['operator']['name'] == 'CS':
            value = torch.tensor(H_funcs.block_num * int(H_funcs.N // 1) // 3, dtype=torch.float32)
            input_size = int(torch.sqrt(value).item())
            y_n = y_xn.reshape(1,model.in_channels,input_size,input_size)

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

        if i >= 0:
            break

    end_time = time.time()
    running_time = end_time - start_time
    print('Total # total running Time: {}'.format(end_time - start_time))

if __name__ == '__main__':
    main()
