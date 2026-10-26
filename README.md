# Kernelized Activation Steering (KAS)

**Content warning**: This repository contains text that is offensive, harmful, or otherwise inappropriate in nature.

Code for the language-model experiments of
[Kernelized Activation Steering](https://arxiv.org/abs/2610.01062): jailbreaking (JailbreakBench),
CAA-style multiple-choice steering, and the KAS ablations. KAS is compared with ActAdd,
Curveball, CHaRS, and (opt-in, Gemma-2-9B only) SAS. The pipeline and data are built on
[refusal_direction](https://github.com/andyrdt/refusal_direction).

## Setup

Requires Linux, an NVIDIA GPU with a CUDA 12.8 driver, and Python 3.11.

```bash
git clone https://github.com/KleepyHunter/kernel_steering.git
cd kernel_steering
conda env create -f environment.yml
conda activate refusal_direction
pip install --no-deps -r constraints.txt   # vLLM and the rest of the pinned environment
huggingface-cli login
```

Without conda, `source setup.sh` creates `./venv` with the same packages.
Accept the licenses of [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it),
[`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and
[`meta-llama/Llama-Guard-3-8B`](https://huggingface.co/meta-llama/Llama-Guard-3-8B) on Hugging Face first.

**Jailbreak judge.** Jailbreak ASR is scored by Llama-Guard-3-8B served with vLLM. Start it
once on its own GPU and leave it running (CAA experiments do not need it):

```bash
LLAMAGUARD_GPU=1 scripts/serve_llamaguard.sh        # ready when curl http://0.0.0.0:8000/v1/models lists the model
export LLAMAGUARD_VLLM_SERVER=http://0.0.0.0:8000/v1 OPENAI_API_KEY=dummy
```

Check the installation with `python -m pytest tests` (no GPU needed).

## Quick start

Steer a model with KAS at a chosen layer and step size:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pipeline.run_pipeline \
  --model_path meta-llama/Llama-3.1-8B-Instruct --layer <LAYER> --eta <ETA>
```

One run (about 15–20 minutes on an H100):

1. collects reference activations from the harmful and harmless reference prompts,
2. fits ActAdd and KAS and saves their steer configs (`steer_config/`),
3. generates completions on 100 JailbreakBench prompts and scores them with LlamaGuard
   (`completions/jailbreakbench_{actadd,kernel}_evaluations.json`),
4. computes Pile and Alpaca perplexity (`loss_evals/`).

Results go to `pipeline/runs/` (change with `--artifact_root`). Add `--prompt_only` to steer
only the prompt tokens. `--steer_method curveball|chars|sas` runs the other methods, and
`--evaluation_protocol caa` runs CAA-style evaluation. See `python -m pipeline.run_pipeline --help`.

## Reproducing the paper

Each experiment has a script under `scripts/`. Hyperparameters are selected on validation splits
and locked in a checksummed manifest, which the test scripts read, so run validation first.

| Paper result | Commands |
|---|---|
| Jailbreak table (ActAdd, KAS, Curveball, CHaRS) | `scripts/jailbreak/validate.sh`, `scripts/jailbreak/validate_chars_all_models.sh all`, then `scripts/jailbreak/test.sh all` |
| CAA tables (all three system-prompt conditions) | `scripts/caa/validate.sh`, then `scripts/caa/test.sh all` |
| CAA, Curveball and CHaRS (negative prompt) | `scripts/caa/validate_curveball.sh all`, `scripts/caa/validate_chars_neg.sh all`, then `scripts/caa/test_curveball_sas.sh all curveball` and `scripts/caa/test_chars_neg.sh all` |
| Single-pair ablation | `scripts/ablations/single_pair.sh all` |
| Number of references | `scripts/ablations/n_train.sh all` |
| Kernel choice | `scripts/ablations/kernel_choice.sh all` |
| TinyBenchmarks utility scores | `scripts/run_tinybench.sh <model> <output_dir> [<steer_config.pt>]` |

All scripts read their settings from environment variables listed in their headers. Set
`CUDA_VISIBLE_DEVICES` before running; `CAMPAIGN` sets the output directory (default
`validation_runs/`). Most scripts take `prepare | run | all`: `prepare` writes a JSONL job file
(and prints an `sbatch` command for Slurm arrays), `run` executes it locally. Completed jobs are
skipped on rerun.

## Hyperparameters

**In a single run**, set hyperparameters with `pipeline.run_pipeline` flags:

| Flag | Controls |
|---|---|
| `--layer` | layer whose residual stream is steered |
| `--eta` | KAS / ActAdd step size (strength) |
| `--step_count` | number of KAS gradient steps |
| `--lambda_reg` | pull back toward the original activation (only acts after the first step) |
| `--kernel_type`, `--kernel_params` | kernel (`rbf`, `polynomial`, `laplacian`) and its parameters as JSON, e.g. `'{"sigma": 48}'` |
| `--prompt_only` | steer only the prompt tokens instead of every token |
| `--n_train`, `--seed` | number of reference pairs and the seed used to sample them |
| `--steer_method`, `--curveball_*`, `--chars_*`, `--sas_*` | the baseline methods and their settings |

`python -m pipeline.run_pipeline --help` lists every flag with its default.

**In the experiment scripts**, set them with environment variables. Each script lists the
ones it reads, with their defaults, in its header, for example:

```bash
LAYER=<LAYER> ETA=<ETA> SEED=<SEED> CUDA_VISIBLE_DEVICES=0 scripts/ablations/kernel_choice.sh all
MODEL_PATH=<MODEL> N_TRAIN="1 2 4 8" CUDA_VISIBLE_DEVICES=0 scripts/ablations/n_train.sh all
```

**Selected hyperparameters.** For the main jailbreak and CAA results, layer and strength are
not set by hand: the validation scripts sweep a grid, select the best setting under the
selection rule, and write it to `$CAMPAIGN/manifests/`. The test scripts then read only that
manifest. To change a grid or the selection rule, edit the grid definitions in
`scripts/build_validation_jobs.py` and the rules in `pipeline/validation_selection.py`;
[VALIDATION_SELECTION.md](VALIDATION_SELECTION.md) describes both. The selected values are
reported in the paper's appendix.

## Repository layout

| Path | Contents |
|---|---|
| `pipeline/` | the pipeline: references, steering methods (`utils/`), evaluation, validation selection |
| `scripts/` | experiment scripts (`jailbreak/`, `caa/`, `ablations/`), job builders and runners, judge server |
| `dataset/` | harmful/harmless splits, JailbreakBench and the CAA behaviors |
| `lm-evaluation-harness/` | lm_eval 0.4.9.2 with steering support, for TinyBenchmarks |
| `tests/` | unit tests |

## Citation

If you use this code, please cite our paper ([arXiv:2610.01062](https://arxiv.org/abs/2610.01062)):

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
```
