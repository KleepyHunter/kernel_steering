# Kernelized Activation Steering (KAS): experiment code for jailbreaking and CAA-defined tasks.

**Content warning**: This repository contains text that is offensive, harmful, or otherwise inappropriate in nature.

Code for the language-model experiments in the paper: jailbreaking (JAILBREAKBENCH),
CAA-style multiple-choice steering, and the KAS ablations. It compares KAS with
ActAdd, Curveball, CHaRS, and (Gemma-2-9B only, opt-in) SAS. The repository started
as a fork of [refusal_direction](https://github.com/andyrdt/refusal_direction)
([paper](https://arxiv.org/abs/2406.11717)). Its pipeline and data are reused here.

- [1. Requirements](#1-requirements)
- [2. Installation](#2-installation)
- [3. Running experiments](#3-running-experiments)
- [4. Experiments → scripts](#4-experiments--scripts)
- [5. Running the pipeline directly](#5-running-the-pipeline-directly)
- [6. Code layout](#6-code-layout)
- [7. Troubleshooting](#7-troubleshooting)

## 1. Requirements

- **OS and GPUs.** Linux with NVIDIA GPUs and a driver that supports CUDA 12.8 (the
  environment installs `torch 2.10.0+cu128`). Each pipeline job uses one GPU: the 7–9B models
  run in bf16 (about 16–19 GB of weights) plus generation and activation memory. The
  LlamaGuard judge needs a GPU of its own, or a share of one (it reserves 40% of a GPU's
  memory by default). Development used <GPU_MODEL> GPUs.
- **Python.** 3.11 exactly. The pinned packages are those of the current working environment,
  and pandas 2.0.3 has no wheels for newer Python versions.
- **Disk.** About 11 GB for the environment and 15–19 GB per model in the Hugging Face cache
  (Gemma-2-9B-IT, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and Llama-Guard-3-8B). SAS also
  downloads Gemma Scope sparse autoencoders.
- **Hugging Face access.** An account that has accepted the licenses of the gated models:
  [`google/gemma-2-9b-it`](https://huggingface.co/google/gemma-2-9b-it),
  [`meta-llama/Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), and
  [`meta-llama/Llama-Guard-3-8B`](https://huggingface.co/meta-llama/Llama-Guard-3-8B).
  `Qwen/Qwen2.5-7B-Instruct` is not gated.
- **Network.** Every run downloads `tatsu-lab/alpaca` and streams `monology/pile-uncopyrighted`
  for the perplexity stage, unless they are already in the Hugging Face cache.

## 2. Installation

### Python environment (conda)

One environment, `refusal_direction`, runs the pipeline and TinyBenchmarks. It reproduces the
current working setup and is defined by two files:

| File | Contents |
|---|---|
| [`requirements.txt`](requirements.txt) | the packages the code imports (pipeline and TinyBenchmarks), at the current versions, plus the environment constraints |
| [`constraints.txt`](constraints.txt) | the current environment's full `pip freeze`, applied by `requirements.txt` with `-c` |

[`environment.yml`](environment.yml) creates the environment with Python 3.11 and installs
`requirements.txt`. The sweep scripts activate an environment with this name. A second step
installs the rest of the frozen environment, including vLLM, at its exact versions. Run both
from the repository root:

```bash
git clone <this repository> refusal_direction
cd refusal_direction
conda env create -f environment.yml
conda activate refusal_direction
pip install --no-deps -r constraints.txt
```

The second step needs `--no-deps` because vLLM declares `opencv-python-headless>=4.13`, which
requires numpy ≥ 2. The current environment runs numpy 1.26.4 without opencv, which vLLM only
uses for image inputs. Afterwards `pip check` reports only that missing opencv.

Key versions: `torch 2.10.0` (CUDA 12.8), `transformers 4.57.6`, `vllm 0.18.0`,
`openai 2.24.0`, `sae-lens 6.39.0`, `transformer-lens 2.18.0`, `scikit-learn 1.8.0`,
`numpy 1.26.4`, `pandas 2.0.3`. There is one deliberate difference from the current
environment: litellm is 1.53.1, because the current 1.40.9 is no longer on PyPI.
`constraints.txt` lists its differences from the freeze at the top.

Without conda, `source setup.sh` does the same with `python3` (it must be Python 3.11). It
creates `./venv`, runs both install steps, and optionally logs in to Hugging Face.

### Hugging Face login

```bash
huggingface-cli login                 # or: export HF_TOKEN=<HF_TOKEN>
export HF_HOME=<PATH_TO_HF_CACHE>     # optional: where models and datasets are cached
```

### LlamaGuard judge (jailbreak experiments only)

Jailbreak ASR is scored by Llama-Guard-3-8B behind vLLM's OpenAI-compatible API. Start it
once in the same environment and leave it running. CAA experiments do not need it.

```bash
LLAMAGUARD_GPU=<GPU_ID> scripts/serve_llamaguard.sh   # also: LLAMAGUARD_PORT=8000, LLAMAGUARD_GPU_MEMORY=0.4
curl http://0.0.0.0:8000/v1/models             # ready when this lists meta-llama/Llama-Guard-3-8B
```

Then, in the shell that runs experiments:

```bash
export LLAMAGUARD_VLLM_SERVER=http://0.0.0.0:8000/v1   # the default; use the server's host and port
export OPENAI_API_KEY=dummy                            # required by the client, not checked by vLLM
```

The first start downloads the model (about 16 GB), so it can take several minutes.

### TinyBenchmarks

TinyBenchmarks utility scores come from the copy of `lm-evaluation-harness` in
[`lm-evaluation-harness/`](lm-evaluation-harness/) (lm_eval 0.4.9.2 with the steering
changes). The environment above installs it in editable mode, so `lm_eval` is available
after `conda activate refusal_direction`. Its `steered` model applies the steer configs that
pipeline runs write to `steer_config/*.pt`. Score a model, unsteered or with a steer config
from a pipeline run:

```bash
scripts/run_tinybench.sh meta-llama/Llama-3.1-8B-Instruct results/tinybench/llama_baseline
scripts/run_tinybench.sh meta-llama/Llama-3.1-8B-Instruct results/tinybench/llama_kas \
  <run>/steer_config/steer_config_kernel.pt
```

`TASKS`, `DEVICE`, `BATCH_SIZE`, and `CUDA_VISIBLE_DEVICES` can be overridden. For
manifest-locked utility tests, run
[`scripts/run_locked_tinybench.py`](scripts/run_locked_tinybench.py) instead. It checks the
manifest and the steer config, then runs the same harness.

The harness copy supports these steer-config actions: `add` (ActAdd), `kernel` (KAS), and the
upstream `ablate`, `clamp`, and `high_dim`. It does not yet support
`curveball_kpca` (Curveball), `chars` (CHaRS), or `sas_sparse` (SAS), so TinyBenchmarks
for those methods stops with `Unknown hook type`.

### Data

The prompt splits are tracked in the repository: `dataset/splits/` (harmful/harmless
train/val/test), `dataset/processed/` (JAILBREAKBENCH and other benchmarks), and
`dataset/caa/` (the six CAA behaviors). To re-download the CAA data from
[nrimsky/CAA](https://github.com/nrimsky/CAA), run `python scripts/download_caa_datasets.py`.

### Check the installation

```bash
python -m pytest tests                       # unit tests, no GPU needed
python -m pipeline.run_pipeline --help       # all pipeline flags
scripts/ablations/kernel_choice.sh prepare   # writes a 6-job file under validation_runs/, no GPU needed
```

## 3. Running experiments

All experiment scripts live under `scripts/`. The conventions are shared:

- **Actions.** Most scripts take `prepare | run | all`; validation scripts also take `select`.
  `prepare` writes a JSONL job file and prints the matching `sbatch` command. `run` executes
  the jobs locally, one after another. `all` does both. The two main validation scripts,
  `scripts/jailbreak/validate.sh` and `scripts/caa/validate.sh`, take no argument and run
  prepare → run → select end to end.
- **Configuration.** Set options through environment variables. Each script lists its own
  overrides in its header. The common ones are `CUDA_VISIBLE_DEVICES` (required to run jobs; the scripts
  default to the placeholder `<GPU_ID>` and stop until it is set), `CAMPAIGN`
  (the output directory, by default under `validation_runs/`), model paths such as `MODEL_PATH`
  or `GEMMA_MODEL`, `LLAMAGUARD_VLLM_SERVER`, and `PYTHON_BIN`.
- **Resuming.** Every job writes a completion marker to `<job file>.status/`, and a rerun skips
  completed jobs. An interrupted campaign continues where it stopped.
- **Slurm.** Submit a prepared job file as an array, adding your cluster's GPU, partition,
  and time options:

  ```bash
  N=$(wc -l < "$JOB_FILE")
  sbatch --array=0-$((N-1)) --gres=gpu:1 \
    --export=ALL,JOB_FILE="$JOB_FILE",STATUS_DIR="$JOB_FILE.status" \
    scripts/slurm_validation_array.sh
  ```

- **Order.** Validation selects hyperparameters and locks them in a SHA-256-checked manifest
  (`$CAMPAIGN/manifests/`). Test scripts read that manifest and refuse any other setting.
  Always run a validation script before its test script.

Scripts can be launched from any directory; they `cd` to the repository root.

## 4. Experiments → scripts

[VALIDATION_SELECTION.md](VALIDATION_SELECTION.md) gives the splits, grids, fixed settings,
and selection rules.

### Jailbreaking (main jailbreak table)

| Step | Methods | Script |
|---|---|---|
| Validation + selection | ActAdd, KAS, Curveball (Gemma, Llama, Qwen) | `scripts/jailbreak/validate.sh` |
| Validation + selection | CHaRS (fixed K per model) | `scripts/jailbreak/validate_chars_all_models.sh all` (one model: `validate_chars.sh`) |
| Test: ASR + perplexity | all of the above, at the selected settings | `scripts/jailbreak/test.sh all` |
| Validation / test | SAS (opt-in) | `scripts/jailbreak/validate_sas.sh all`, then `scripts/jailbreak/test_sas_locked.sh all` |
| TinyBenchmarks utility | selected steer configs | `scripts/run_locked_tinybench.py` (see VALIDATION_SELECTION.md §3) |

`scripts/jailbreak/test.sh` reads the jailbreak manifests and makes one standard-phase run per
selected method and application mode (prompt-only / all-token). Each run reports
JAILBREAKBENCH ASR and Pile / Alpaca perplexity. It also saves the steer config used by
TinyBenchmarks. To include SAS, list its manifest in `MANIFESTS`.

### CAA-style multiple-choice steering (Gemma-2-9B; main CAA table and appendix)

| Step | Methods | Script |
|---|---|---|
| Validation + selection | ActAdd, KAS, Curveball, all prompt conditions | `scripts/caa/validate.sh` (`CAA_PROMPT_CONDITION=all\|none\|pos\|neg`) |
| Locked test | baseline, ActAdd, KAS, Curveball | `scripts/caa/test.sh all` |
| Validation / test | Curveball, negative prompt only | `scripts/caa/validate_curveball.sh all`, then `scripts/caa/test_curveball_sas.sh all curveball` |
| Validation / test | CHaRS | `scripts/caa/validate_chars.sh all`; negative prompt: `validate_chars_neg.sh all`, then `test_chars_neg.sh all` |
| Validation / test | SAS (opt-in) | `scripts/caa/validate_sas.sh all`, then `scripts/caa/test_curveball_sas.sh all sas` |

CHaRS uses the repository's `chars_equal_cov_v2` recipe: independent source/target k-means,
entropy-regularized optimal transport, and the RBF-gated equal-covariance update.
K is fixed, not tuned. It is K=10 for Gemma-2-9B and K=5 for Qwen2.5-7B (the paper's
settings). Llama-3.1-8B uses the default K=5, and the manifest records this fallback.

The main table uses the negative system prompt. The no-prompt and positive-prompt
tables in the appendix come from the same runs (`CAA_PROMPT_CONDITION=all`).

### Ablations (jailbreaking, standard phase)

| Ablation | Script | Default setting |
|---|---|---|
| Jailbreaking from a single contrastive pair | `scripts/ablations/single_pair.sh all` | 3 models × seeds 0–9 × {1 step, 10 steps}; ActAdd at the same eta |
| Number of reference pairs | `scripts/ablations/n_train.sh all` | Gemma, layer 22, eta 150, n ∈ {1,…,128} |
| Kernel choice | `scripts/ablations/kernel_choice.sh all` | Llama, layer 13, eta 5, Laplacian / polynomial / RBF |

Every ablation run computes ASR and perplexity for prompt-only and all-token steering.
With `--n_train` set to a value other than 1 or 128, prompts removed by refusal filtering are
replaced from the rest of the train split, so the run uses exactly n pairs.
Each script prints its per-model settings in its header and accepts overrides
(e.g. `LAYER`, `ETA`, `SEEDS`). The jobs are built by `scripts/build_experiment_jobs.py`.

### Parameter sweeps (exploratory, outside the paper)

`scripts/sweeps/*.sh` are the original single-GPU sweep wrappers. They loop
`pipeline.run_pipeline` over a strength grid and can also run TinyBenchmarks. They activate
conda environments themselves: `refusal_direction` for the pipeline, and `steering-lm-eval`
for TinyBenchmarks by default. With the single environment above, set
`LM_EVAL_CONDA_ENV=refusal_direction`. Run them from the repository root.

## 5. Running the pipeline directly

```bash
python -m pipeline.run_pipeline --model_path google/gemma-2-9b-it --layer 22 --eta 150 --prompt_only
python -m pipeline.run_pipeline --help      # all flags
```

For reference, one standard jailbreak run of KAS on Llama-3.1-8B-Instruct (100 JAILBREAKBENCH
completions plus the perplexity stage) takes about <RUNTIME> on one <GPU_MODEL>, with the judge on a
second GPU.

The default `--steer_method kernel` evaluates ActAdd and KAS in one run (select/test runs add the unsteered baseline).
`--steer_method curveball|chars|sas` select the other methods.
`--evaluation_protocol legacy` (jailbreak), `caa`, or `both` picks the benchmark, and
`--experiment_phase standard|select|test` picks the phase. Validation
selection uses `select`; manifest-locked tests use `test`. Every run, in every phase and protocol, does:

1. collect reference activations (`generate_directions/`)
2. fit the steering direction or state and export steer configs (`steer_config/`)
3. evaluate the steering:
   - jailbreak: generate completions and score them with LlamaGuard-3
     (`completions/<dataset>_<method>_evaluations.json`)
   - CAA: behavior-matching answer probability (`caa_eval/`)
4. compute CE loss / perplexity on Pile and Alpaca (`loss_evals/<method>_loss_eval.json`)

CHaRS always fits its state in the run. A `--chars_state_path` from `scripts/fit_chars_states.py`
is only checked against that refit, and the run stops if the fingerprints differ.

`<method>` is `baseline`, `actadd`, `kernel` (KAS), `curveball`, `chars`, or `sas`. Outputs
go under `--artifact_root` (default `pipeline/runs/`) in a directory named after the run's
hyperparameters (see `Config.artifact_path`), for example:

```
<artifact_root>/seed-1/use_mean_act-False/kernel_type-rbf/kernel_params-None/eta-5.0/lambda_reg-0.1/
  pos--1/layer-12/step_count-1/prompt_only-False/Llama-3.1-8B-Instruct/
    completions/jailbreakbench_kernel_evaluations.json
    loss_evals/kernel_loss_eval.json
    steer_config/steer_config_kernel.pt
```

## 6. Code layout

| Module | Contents |
|---|---|
| `pipeline/run_pipeline.py` | entry point: config → references → interventions → evaluations |
| `pipeline/cli.py` | command-line flags |
| `pipeline/settings.py` | flag resolution and checks, `build_config` |
| `pipeline/config.py` | `Config` and the artifact directory layout |
| `pipeline/phase_validation.py` | select/test phase checks and manifest locking |
| `pipeline/references.py` | reference prompt sampling and filtering (legacy and CAA) |
| `pipeline/interventions.py` | fits ActAdd/KAS, Curveball, CHaRS, SAS, exports steer configs |
| `pipeline/evaluation.py` | completions, LlamaGuard scoring, CE loss, CAA evaluation |
| `pipeline/validation_selection.py`, `caa_validation.py`, `sas_validation.py` | candidate records, selection, manifests, bootstrap CIs |
| `pipeline/utils/` | kernel steering, Curveball, CHaRS, SAS, hook utilities |
| `scripts/build_validation_jobs.py` | select / locked-test job files |
| `scripts/build_experiment_jobs.py` | standard-phase job files (jailbreak test, ablations) |
| `scripts/validation_workflow.sh`, `run_validation_job.py`, `slurm_validation_array.sh` | job runners |
| `scripts/serve_llamaguard.sh` | LlamaGuard-3 judge server |

## 7. Troubleshooting

- **`Connection refused` while scoring jailbreaks.** The LlamaGuard server is not running or
  `LLAMAGUARD_VLLM_SERVER` points to the wrong host or port. Check with
  `curl $LLAMAGUARD_VLLM_SERVER/models`.
- **`401` / `403` when loading a model.** The Hugging Face account has not accepted that
  model's license, or you are not logged in.
- **Out of GPU memory.** Run the judge on a different GPU from the pipeline, or lower
  `LLAMAGUARD_GPU_MEMORY` together with `LLAMAGUARD_MAX_MODEL_LEN=8192`. The judge's default
  131k-token context alone needs about 16 GB of cache. Perplexity uses `--ce_loss_batch_size`
  (default 2).
- **`No space left on device` while installing.** pip unpacks wheels (torch, CUDA libraries,
  vLLM) into the temp directory, several GB at a time. Point it at a larger disk, e.g.
  `export TMPDIR=<PATH_TO_LARGE_DISK>/tmp PIP_NO_CACHE_DIR=1`, before both install steps.
- **`No module named vllm.__main__`.** vLLM 0.18 is started with `vllm serve`, as
  `scripts/serve_llamaguard.sh` does, not with `python -m vllm`.
- **A test script refuses to run.** It needs the manifest from the matching validation
  script, unchanged. Manifests are checked against their `.sha256` file and the current data
  splits.

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
