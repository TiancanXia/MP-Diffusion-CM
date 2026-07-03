from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch

# device = 'cuda:0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

task = 'SR'
factor = 4
sigma = 0.1
scale = 1.0


label_root = Path(f'./results/CS/label')
delta_recon_root = Path(f'./results/CS/recon') # GAMP-DMPS_1323_12288_1000_1 2 2.5 all
# normal_recon_root = Path(f'./results/{task}/ffhq/{factor}/{sigma}/ps+/{scale}/recon')

psnr_delta_list = []
psnr_normal_list = []

lpips_delta_list = []
lpips_normal_list = []

ssim_delta_list =[]
ssim_normal_list = []  # 

with torch.no_grad():  # 
    for idx in tqdm(range(100)):
        fname = str(idx).zfill(5)

        # float32 [0, 1]
        label_np = plt.imread(label_root / f'{fname}.png')[:, :, :3].astype(np.float32)
        recon_np = plt.imread(delta_recon_root / f'{fname}.png')[:, :, :3].astype(np.float32)

        # SSIM
        ssim_delta = structural_similarity(
            label_np, recon_np,
            data_range=1.0,
            channel_axis=2
        )
        ssim_delta_list.append(ssim_delta)

        # PSNR
        psnr_delta = peak_signal_noise_ratio(label_np, recon_np, data_range=1.0)
        psnr_delta_list.append(psnr_delta)

        # LPIPS
        t_recon = torch.from_numpy(recon_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2. - 1.
        t_label = torch.from_numpy(label_np).permute(2, 0, 1).unsqueeze(0).to(device) * 2. - 1.

        # LPIPS
        dist = loss_fn_vgg(t_recon, t_label)
        lpips_delta_list.append(dist.item())

        print(f'DPS PSNR: {psnr_delta:.4f} | SSIM: {ssim_delta:.4f} | LPIPS: {dist.item():.4f}')

# 
psnr_avg = np.mean(psnr_delta_list)
ssim_avg = np.mean(ssim_delta_list)
lpips_avg = np.mean(lpips_delta_list)

print(f'DPS PSNR: {psnr_avg:.4f} | SSIM: {ssim_avg:.4f} | LPIPS: {lpips_avg:.4f}')
