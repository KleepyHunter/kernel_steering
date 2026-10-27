# KAS: Image Generation Style Control

Code for the image-generation experiment of
[Kernelized Activation Steering](https://arxiv.org/abs/2610.01062) (Section 5.3 and Appendix B.4):
steering a text-to-image diffusion model toward a target visual style with KAS, compared with
Mean-AcT. It is built on [Mean-AcT](https://github.com/apple/ml-act) (Rodriguez et al., ICLR 2025).
The language-model experiments (jailbreaking, CAA) are in [`../Jailbreaking_CAA`](../Jailbreaking_CAA).

**Setup of the experiment.** Prompts are sampled from the COCO Captions training set. Each prompt
is used twice: as is (source) and with style tags appended (target); the style tags, generated with
Llama-3-8B-Instruct, are in `data/style_prompts.json`. KAS is fitted on the FLUX.1-schnell
activations of the source and target prompts and applied sequentially. Images are then generated from
COCO validation prompts at a range of intervention strengths and evaluated with zero-shot CLIP style
classification (style induction) and CLIPScore against the original prompt (content preservation).

## Setup

This folder has its own environment, separate from `Jailbreaking_CAA`:

```bash
cd kernel_steering/Image
pip install -r requirements.txt
```

Download COCO Captions into the data directory:

```bash
export DATA_DIR="$PWD/data" CACHE_DIR="$PWD/cache"
python -m act.scripts.download_external_data   # COCO Captions (also RealToxicityPrompts, Jigsaw)
```

The run scripts use `Image/data` and `Image/cache` unless `DATA_DIR` and `CACHE_DIR` are set.
`Image/data` already contains the style tags (`style_prompts.json`); with another data directory,
copy that file into it. `HF_HUB_CACHE` and `HF_TOKEN` are optional. Run the unit tests with
`python -m pytest tests`.

## Running

| Method | Command |
|---|---|
| KAS | `bash run_kernel_t2i.sh <seed> [gpu_id] [style]` |
| Mean-AcT | `bash run_t2i.sh <seed> [gpu_id] [style]` |

`style` is the target style, one of the keys of `data/style_prompts.json`; pass the same style to
both scripts to compare the methods. Each script runs `act.scripts.pipeline` with `text_to_image_generation.yaml`: it collects
FLUX.1-schnell activations for the source and target prompts, fits the intervention, and generates
images from the validation prompts over two strength sweeps. The second sweep also runs the CLIP
evaluation and writes `calculate_clip_score/clip_score.csv` to the results directory
(`results_<model>_kernel_<kernel>_<style>_eta_<eta>_<seed>` for KAS,
`results_<model>_mean_ot_<style>_<seed>` for Mean-AcT).

## Hyperparameters

The target style is the `style` argument of the run scripts. All other settings are Hydra
overrides, set in the run scripts:

| Setting | Where |
|---|---|
| KAS kernel and bandwidth | `intervention_params.hook_params.kernel_type` and `kernel_params` (`null` uses the median heuristic for the RBF bandwidth) |
| KAS step size and steps | `intervention_params.hook_params.eta` and `step_count` (the `for eta in ...` loop) |
| intervention strengths (x-axis of the figure) | `text_to_image_generation.min_strength`, `max_strength`, `strength_steps`, one line per sweep; the strength scales the KAS update |
| sequential application | `intervention_params.incremental=incr` |
| number of reference prompts | `responses.batch_size` × `responses.max_batches` |
| number of evaluation prompts | `text_to_image_generation.batch_size` × `text_to_image_generation.max_batches` |
| diffusion model | `model=` (configs in `act/configs/model/`: FLUX.1-schnell, FLUX.1-dev, SDXL-Lightning) |

Defaults for every intervention are in `act/configs/intervention_params/` (`kernel.yaml` for KAS,
`mean_ot.yaml` for Mean-AcT). To compare the two methods, run both scripts with the same seed,
style and model.

## Results

- `read_results.ipynb` loads the KAS and Mean-AcT CSVs and plots the zero-shot style score and
  CLIPScore against intervention strength. Set `model`, `style`, `kernel_type`, `eta` and `seed`
  in its second cell to the values used for the runs.
- `visualize.py` builds image grids (rows: prompts, columns: strengths) from the generated images.

## Layout

| Path | Contents |
|---|---|
| `act/` | pipeline (`scripts/`), interventions (`hooks/`; KAS is `hooks/kernel_steering.py`), datasets, evaluations, Hydra configs |
| `data/style_prompts.json` | style tags appended to the target prompts |
| `run_kernel_t2i.sh`, `run_t2i.sh` | KAS and Mean-AcT style-control runs |
| `run_kernel.sh`, `run_mean_ot.sh` | toxicity-mitigation runs from Mean-AcT (not part of the paper) |
| `read_results.ipynb`, `visualize.py` | result plots and image grids |
| `tests/` | unit tests |

## Citation

If you use this code, please cite our paper and Mean-AcT:

```bibtex
@misc{abdullaev2026kernelizedactivationsteering,
      title={Kernelized Activation Steering},
      author={Laziz U. Abdullaev and Minh-Hieu Pham and Bach Do and Khoat Than and Tan M. Nguyen},
      year={2026},
      eprint={2610.01062},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2610.01062},
}

@inproceedings{rodriguez2025act,
  title     = {Controlling Language and Diffusion Models by Transporting Activations},
  author    = {Rodriguez, Pau and Blaas, Arno and Klein, Michal and Zappella, Luca and Apostoloff, Nicholas and Cuturi, Marco and Suau, Xavier},
  booktitle = {International Conference on Learning Representations},
  year      = {2025}
}
```
