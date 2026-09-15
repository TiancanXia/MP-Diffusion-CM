# MP-Diffusion-CM

MP-Diffusion-CM combines diffusion or consistency-model priors with message-passing methods for image inverse problems. The current executable pipeline is specialized for **block compressed sensing (CS)** and includes support for non-differentiable, element-wise observations such as uniform quantization.

## Highlights

- DDPM/DDIM and Consistency Model (CM) priors.
- MMPS, PGDM, DPS, GAMP, and VAMP reconstruction paths.
- GAMP output steps for Gaussian and quantized measurements.
- YAML-based experiment configuration.
- Automatic saving of measurements, reconstructions, references, and intermediate results.

## Supported algorithms

Algorithms are selected with `algorithm.name` in the task configuration. Names are case-sensitive.

| Prior     | `algorithm.name`                             | Notes                                                   |
| --------- | -------------------------------------------- | ------------------------------------------------------- |
| DDPM/DDIM | `mmps`, `pgdm`, `dps`                        | Gradient-based measurement correction                   |
| DDPM/DDIM | `gamp_mm`, `gamp_ga`, `gamp-ga`, `gamp_dmps` | GAMP variants; support the quantized observation module |
| DDPM/DDIM | `vamp`                                       | VAMP with a row-orthonormal CS operator                 |
| CM        | `gamp_mm`, `cm_mmps`, `vamp`                 | CM versions of GAMP-MM, MMPS, and VAMP                  |
| CM        | `Tvamp`, `Tgamp`, `cm_gamp_ps`               | Experimental turbo/fusion variants                      |

Non-differentiable observations cannot be used with MMPS, PGDM, DPS, or VAMP. The supplied quantized-CS configurations therefore use `gamp_mm`.

## Installation

The original environment uses Python 3.8, PyTorch 1.11, and CUDA 11.3. A CUDA-capable GPU is strongly recommended.

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

### Checkpoints

- DDPM default: `models/ffhq_10m.pt`, configured by `configs/model_config.yaml`.
- CM default: `consistency_models-main/checkpoints/cd_cat256_lpips.pt`, configured by `configs/cm_diffusion_config.yaml`.

The FFHQ checkpoint is available from the original [Google Drive folder](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh?usp=sharing). Place downloaded checkpoints at the paths above or update the corresponding YAML fields.

## Running experiments

### Standard compressed sensing

```bash
python sample_condition.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/CS_config.yaml \
  --gpu 0 \
  --save_dir ./results
```

The default configuration uses DDIM with `gamp_mm`. Change `algorithm.name` in `configs/CS_config.yaml` to select another compatible algorithm.

### Quantized compressed sensing

```bash
python sample_condition.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/diffusion_config.yaml \
  --task_config configs/quantized_CS_config.yaml \
  --gpu 0 \
  --save_dir ./results_quantized
```

Uniform quantization is enabled by:

```yaml
measurement:
  observation:
    type: quantization
    step_size: 0.1
```

### Consistency Model prior

```bash
python sample_condition.py \
  --model_config configs/model_config.yaml \
  --diffusion_config configs/cm_diffusion_config.yaml \
  --task_config configs/CS_cm_config.yaml \
  --gpu 0 \
  --save_dir ./results_cm
```

The CM path requires both the `prior` section in the diffusion configuration and `prior_type` in the task configuration:

```yaml
algorithm:
  name: cm_mmps
  prior_type: openai_cm
```

For quantized CS with the CM prior, use `configs/quantized_CS_cm_config.yaml`; it selects CM-GAMP-MM with `algorithm.name: gamp_mm`.

The shorthand script accepts the task name, GPU index, and output directory:

```bash
bash scripts/run_sampling.sh CS 0 ./results
```

## Configuration reference

| File                                  | Purpose                                                |
| ------------------------------------- | ------------------------------------------------------ |
| `configs/model_config.yaml`           | DDPM U-Net architecture and checkpoint                 |
| `configs/diffusion_config.yaml`       | DDPM/DDIM schedule and timestep respacing              |
| `configs/cm_diffusion_config.yaml`    | CM checkpoint, sigma schedule, and correction controls |
| `configs/CS_config.yaml`              | Standard CS algorithm, data, and noise                 |
| `configs/quantized_CS_config.yaml`    | Quantized CS with a DDPM/DDIM prior                    |
| `configs/CS_cm_config.yaml`           | CS with an OpenAI CM prior                             |
| `configs/quantized_CS_cm_config.yaml` | Quantized CS with an OpenAI CM prior                   |

Important fields:

- `algorithm.name`: reconstruction algorithm.
- `algorithm.prior_type`: set to `openai_cm` for CM paths.
- `sampler`: `ddim` or `ddpm` for the standard diffusion paths.
- `timestep_respacing`: number or spacing of retained diffusion steps.
- `measurement.noise.sigma`: Gaussian noise standard deviation before optional quantization.
- `measurement.observation.step_size`: uniform quantizer interval.
- `prior.num_steps`, `prior.base_num_steps`, `prior.timestep_indices`: CM sigma schedule.
- `prior.correction_damping`, `prior.tau_inflation`: CM correction controls.

## Outputs

Results are written below `<save_dir>/CS/`:

```text
<save_dir>/CS/
|-- input/      # measurements
|-- recon/      # final reconstructions
|-- progress/   # intermediate samples
`-- label/      # reference images
```

The entry point also reports per-image PSNR and sampling time.

## Project structure

```text
.
|-- sample_condition.py                 # experiment entry point
|-- GAMP.py                             # reusable GAMP components
|-- prior_models.py                     # differentiable OpenAI CM wrapper
|-- guided_diffusion/
|   |-- gaussian_diffusion.py           # samplers and algorithm dispatcher
|   |-- measurements.py                 # operators, noise, and quantization
|   `-- condition_methods.py            # gradient conditioning methods
|-- configs/                            # experiment configurations
|-- consistency_models-main/            # vendored OpenAI CM implementation
|-- data/                               # input images
|-- models/                             # DDPM checkpoints
`-- results/                            # default output root
```

## Citation

This code builds on Diffusion Posterior Sampling. If it is useful in your research, please cite:

```bibtex
@software{xia_mp_diffusion_cm,
  author    = {Xia, Tiancan},
  title     = {{MP-Diffusion-CM}},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/TiancanXia/MP-Diffusion-CM},
  note      = {GitHub repository}
}
```

## Acknowledgments

This repository is based on [Diffusion Posterior Sampling](https://github.com/DPS2022/diffusion-posterior-sampling) and incorporates GAMP/VAMP methods together with the OpenAI Consistency Models implementation.
