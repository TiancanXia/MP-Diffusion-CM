# MP-Diffusion

This repository implements **MP-Diffusion**, a novel approach that combines Message Passing (MP) with diffusion models for solving inverse problems.

## Key Features

- **GAMP Integration**: Implements GAMP algorithm for efficient posterior sampling in diffusion models
- **Multiple Algorithms**: Supports MMPS, PGDM, DPS, GAMP-MM, GAMP-GA, VAMP, and their Consistency Model (CM) variants
- **Flexible Configuration**: Easy-to-use YAML configuration files for different tasks
- **Non-Differentiable Observations**: Supports element-wise non-differentiable measurement functions $y = g(Ax + n)$ (e.g., quantization), solved via GAMP's decoupled output-step likelihood estimation
- **Consistency Model Prior**: Supports OpenAI Consistency Models as a prior through CM-GAMP-MM, CM-MMPS, and CM-VAMP sampling paths

## Non-Differentiable Observation Support

GAMP-Diffusion extends the standard linear observation model to handle **non-differentiable element-wise measurement functions** $y = g(Ax + n)$, where $g$ can be any element-wise function (e.g., quantization, saturation). This is achieved by replacing only the output-step likelihood in GAMP, leaving the diffusion prior unchanged. Traditional gradient-based methods (DPS, MMPS, PGDM) are inapplicable here - GAMP-based algorithms are required.

### Quantized Compressed Sensing

Uniform quantization $y = Q_\Delta(Ax + n)$ with $Q_\Delta(\cdot) = \Delta \cdot \text{round}(\cdot / \Delta)$.

```bash
python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/quantized_CS_config.yaml \
    --gpu=0 \
    --save_dir=./results
```

See `interface.md` for detailed mathematical derivation and implementation notes.

For the CM prior path, non-differentiable observations are supported by the CM-GAMP route (`algorithm.name: gamp_mm` with `algorithm.prior_type: openai_cm`). CM-VAMP currently targets differentiable linear observations and will raise an error for non-differentiable observation modules.

## Prerequisites

- Python 3.8+
- PyTorch 1.11.0+
- CUDA 11.3+ (GPU recommended)
- NVIDIA-Docker (optional, for containerized deployment)

Lower CUDA versions are supported with appropriate PyTorch versions (e.g., CUDA 10.2 with PyTorch 1.7.0).

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/TiancanXia/GAMP-diffusion.git
cd GAMP-diffusion
```

### 2. Download Pretrained Checkpoints

Download the pretrained checkpoint `ffhq_10m.pt` from [Google Drive](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing) and place it in the `models/` directory:

```bash
mkdir models
mv {DOWNLOAD_DIR}/ffhq_10m.pt ./models/
```

For CM-based experiments, also download an OpenAI Consistency Model checkpoint and place it under `consistency_models-main/checkpoints/`. The default CM configuration expects:

```bash
consistency_models-main/checkpoints/cd_cat256_lpips.pt
```

### 3. Setup Environment

#### Option 1: Local Environment Setup

Create conda environment and install dependencies:

```bash
conda create -n gamp python=3.8
conda activate gamp
pip install -r requirements.txt

pip install blobfile
pip install mpi4py
pip install scikit-image

pip install piq==0.7.0
pip install einops

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

#### Option 2: Docker Container

Build and run the Docker container (requires Docker >= 19.03 with GPU support):

```bash
docker build -t gamp-diffusion:latest .
docker run -it --rm --gpus=all gamp-diffusion
```

### 4. Run Inference

Execute the sampling script with your desired configuration:

```bash
python sample_condition.py

python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/diffusion_config.yaml \
    --task_config=configs/CS_config.yaml
```

Run with the OpenAI Consistency Model prior:

```bash
python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/cm_diffusion_config.yaml \
    --task_config=configs/CS_cm_config.yaml \
    --gpu=0 \
    --save_dir=./results_cm
```

`configs/CS_cm_config.yaml` selects the CM prior with:

```yaml
algorithm:
  name: cm_mmps # gamp_mm cm_mmps vamp
  prior_type: openai_cm
```

Set `algorithm.name: vamp` in this file to run CM-VAMP. For quantized compressed sensing with the CM prior, use:

```bash
python3 sample_condition.py \
    --model_config=configs/model_config.yaml \
    --diffusion_config=configs/cm_diffusion_config.yaml \
    --task_config=configs/quantized_CS_cm_config.yaml \
    --gpu=0 \
    --save_dir=./results_cm_quant
```

.vscode/settings.json:

```json
{
  "python-envs.defaultEnvManager": "ms-python.python:conda",
  "python-envs.defaultPackageManager": "ms-python.python:conda",
  "python.analysis.extraPaths": [
    "./consistency_models-main"
  ]
}
```

## Project Structure

```
GAMP-diffusion/
|-- GAMP.py                         # Core GAMP algorithm implementation
|-- prior_models.py                 # OpenAI CM prior wrapper and sigma schedule setup
|-- sample_condition.py             # Main inference script
|-- compute_metric.py               # Metric computation utilities
|-- guided_diffusion/               # Diffusion model components
|   |-- measurements.py             # Measurement operators
|   `-- gaussian_diffusion.py       # DDPM/CM samplers with GAMP, MMPS, and VAMP integration
|-- consistency_models-main/        # OpenAI Consistency Models code and checkpoints
|-- configs/                        # Configuration files for different tasks
|   |-- diffusion_config.yaml        # Standard diffusion prior config
|   |-- cm_diffusion_config.yaml     # OpenAI CM prior config
|   |-- CS_config.yaml               # Standard compressed sensing config
|   |-- CS_cm_config.yaml            # CM compressed sensing config
|   `-- quantized_CS_cm_config.yaml  # CM quantized compressed sensing config
|-- models/                         # Pretrained DDPM checkpoints
|-- results/                        # Output directory for reconstructed images
`-- util/                           # Utility functions
```

## Algorithm Details

The standard diffusion path uses `configs/diffusion_config.yaml` and dispatches `mmps`, `pgdm`, `dps`, `gamp_mm`, `gamp_ga`, or `vamp` through `guided_diffusion/gaussian_diffusion.py`.

The CM path is enabled by adding a `prior` section in `configs/cm_diffusion_config.yaml`:

```yaml
prior:
  type: openai_cm
  repo_root: consistency_models-main
  checkpoint: consistency_models-main/checkpoints/cd_cat256_lpips.pt
  image_size: 256
  sigma_min: 0.002
  sigma_max: 80.0
  num_steps: 20
  base_num_steps: 40
  timestep_indices: null
  tau_inflation: 1.0
  correction_damping: 1
  clip_denoised: True
```

When `algorithm.prior_type: openai_cm` is set, `sample_condition.py` builds a `ConsistencyPrior` from `prior_models.py` instead of the DDPM model. The sampler then dispatches:

- `algorithm.name: gamp_mm` -> CM-GAMP-MM (`_step_gamp_cm`)
- `algorithm.name: cm_mmps` -> CM-MMPS (`_step_cm_mmps`)
- `algorithm.name: vamp` -> CM-VAMP (`_step_vamp_cm`)

CM-VAMP keeps the original VAMP Module-A/Module-B structure, obtains a differentiable endpoint from the consistency model through `endpoint_from_vp(...)`, and advances between CM sigma levels with VP re-noising.

Example results will be saved in the `results/` directory with the following structure:

- `input/`: Measurement inputs
- `recon/`: Reconstructed images
- `progress/`: Intermediate sampling results
- `label/`: Ground truth images (for comparison)

## Citation

If you find this work useful in your research, please cite the original DPS paper:

```bibtex
@inproceedings{
chung2023diffusion,
title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
author={Hyungjin Chung and Jeongsol Kim and Michael Thompson Mccann and Marc Louis Klasky and Jong Chul Ye},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=OnD9zGAGT0k}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation builds upon the foundation of [Diffusion Posterior Sampling](https://github.com/DPS2022/diffusion-posterior-sampling) and incorporates GAMP algorithms for improved inverse problem solving.