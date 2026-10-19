# Validation-based selection experiments

This implementation separates hyperparameter selection from final evaluation. `select` runs only on validation data and writes candidate records. `pipeline.validation_selection` verifies the complete grid and writes an immutable manifest plus a SHA-256 sidecar. `test` requires that manifest, checks the current split hashes, and rejects any layer or strength other than the validation-selected configuration.

## Realized splits

CAA uses seed 42 and removes official-test overlaps before sampling. The checked-in data produce:

| Behavior | Reference | Validation | Test |
|---|---:|---:|---:|
| AI Coordination | 310 | 50 | 50 |
| Corrigibility | 240 | 50 | 50 |
| Hallucination | 950 | 50 | 50 |
| Myopic Reward | 900 | 50 | 50 |
| Survival Instinct | 853 | 50 | 50 |
| Sycophancy | 950 | 50 | 50 |

Jailbreaking uses 128 harmful and 128 harmless reference prompts, 39 harmful ASR-validation prompts, 256 harmless utility-validation examples with Alpaca completions, and 100 JAILBREAKBENCH test prompts. The code asserts that harmful reference, ASR validation, and JAILBREAKBENCH are disjoint. TinyBenchmarks are final utility tests only.

## Grids and fixed settings

The registered CAA ActAdd/KAS/Curveball grid is the coarse 20-strength set `{0, 0.01, 0.1, 0.25, 1, 2.5, 4.5, 5, 10, 17.5, 20, 50, 100, 200, 250, 275, 300, 400, 435, 750}` by layers 22 and 23: 40 configurations per method, behavior, and prompt condition. It includes the explicit null intervention and every ActAdd or KAS strength selected in Appendix Tables 7–9. Curveball uses each strength as `curveball_alpha` with CAA extraction at position `-2`, polynomial KPCA, degree 3, automatic gamma, bias 1, 20 components, and inverse-transform alpha 1. The job generator evaluates ActAdd and KAS on shared reference activations and runs Curveball separately. RBF bandwidth is computed from reference activations only; `lambda_reg=0`, `step_count=1`, and answer-token extraction are enforced for KAS.

The jailbreaking ActAdd/KAS/Curveball grid is strengths `{1,2,5,10,20,40,50,100,150,200}` with layers `{22,23}` for Gemma, `{12,13}` for Llama, and `{17,18}` for Qwen. Prompt-only and all-token modes are separate scopes. Curveball is forced to polynomial degree 3, bias 1, 20 KPCA components, and the existing inverse-transform alpha. The selector admits only configurations at or below `1.05 ×` baseline validation perplexity.

SAS (Gemma-2-9B only) is a separate campaign: it is never part of the default (`all`) job sets, and no other method's selection requires it. Its code lives in `pipeline/utils/sas.py` (SAE loading, hook, direction extraction) and `pipeline/sas_validation.py` (recipes, grids, manifest checks). For CAA, SAS uses the same 20 positive strengths as the other methods at layers 22 and 23, each interpreted as `sas_coefficient`, with the fixed Gemma Scope recipe: width `131k`, tau `0.7`, CAA extraction at position `-2`, delta correction, and post-activation application. For jailbreaking, SAS uses the same strength magnitudes with the negative SAS direction, coefficients `{-1,-2,-5,-10,-20,-40,-50,-100,-150,-200}`, width `16k`, tau `0.7`, legacy extraction at position `-1`, delta correction, and post-activation application; jailbreak ties are broken by the smaller coefficient magnitude. A SAS-only campaign is accepted as its own selection scope; if SAS candidates are merged into a cross-method campaign, every Gemma scope must then include them.

## Recommended runnable workflow

The wrapper below creates an isolated campaign directory, so stale pilot runs cannot contaminate complete-grid selection.

### SAS-only sweeps

```bash
# CAA: 720 jobs for all prompt conditions, or 240 for one condition.
CUDA_VISIBLE_DEVICES=<GPU_ID> CAA_PROMPT_CONDITION=all \
  scripts/caa/validate_sas.sh all

# Jailbreak: one baseline plus 40 SAS jobs across both application modes.
CUDA_VISIBLE_DEVICES=<GPU_ID> \
LLAMAGUARD_VLLM_SERVER=http://localhost:8000/v1 \
  scripts/jailbreak/validate_sas.sh all

# Locked held-out tests.
scripts/caa/test_curveball_sas.sh all sas
scripts/jailbreak/test_sas_locked.sh all
```

Each validation script accepts `prepare`, `run`, `select`, or `all`. Use `prepare` before Slurm submission, then `select` after every array job has a completion marker. These scripts select with `--allow-incomplete-grid` for compatibility with existing SAS campaign directories, which contain a few duplicate pilot records; a clean SAS campaign also passes full-grid verification.

### Full cross-method workflow

```bash
cd refusal_direction

CAMPAIGN=/path/to/campaign
GEMMA=/path/to/gemma-2-9b-it
LLAMA=/path/to/Llama-3.1-8B-Instruct
QWEN=/path/to/Qwen2.5-7B-Instruct

# Required by the existing LiteLLM-based LlamaGuard evaluator.
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export LLAMAGUARD_VLLM_SERVER=http://your-llamaguard-host:8000/v1

scripts/validation_workflow.sh prepare-caa "$GEMMA" "$CAMPAIGN"
scripts/validation_workflow.sh prepare-jail "$GEMMA" "$CAMPAIGN"
scripts/validation_workflow.sh prepare-jail "$LLAMA" "$CAMPAIGN"
scripts/validation_workflow.sh prepare-jail "$QWEN" "$CAMPAIGN"
```

Run a single pilot job locally before submitting the arrays:

```bash
scripts/validation_workflow.sh run-index \
  "$CAMPAIGN/jobs/caa_select.jsonl" 0
```

For local sequential execution, use `run-local`. Completed indices are skipped on restart:

```bash
scripts/validation_workflow.sh run-local \
  "$CAMPAIGN/jobs/jail_gemma-2-9b-it.jsonl"
```

For Slurm, submit each JSONL file as a zero-based array. Supply cluster-specific GPU, memory, partition, and time arguments to `sbatch`:

```bash
JOB_FILE="$(realpath "$CAMPAIGN/jobs/caa_select.jsonl")"
STATUS_DIR="${JOB_FILE}.status"
N=$(wc -l < "$JOB_FILE")

sbatch \
  --array="0-$((N-1))" \
  --gres=gpu:1 \
  --export="ALL,JOB_FILE=$JOB_FILE,STATUS_DIR=$STATUS_DIR" \
  scripts/slurm_validation_array.sh
```

After all expected status files exist, lock selections:

```bash
scripts/validation_workflow.sh select-caa "$CAMPAIGN"
scripts/validation_workflow.sh select-jail "$CAMPAIGN"
```

Generate locked final-test jobs:

```bash
scripts/validation_workflow.sh prepare-tests \
  "$GEMMA" "$CAMPAIGN/manifests/caa_validation.json" \
  "$CAMPAIGN" caa_gemma

scripts/validation_workflow.sh prepare-tests \
  "$GEMMA" "$CAMPAIGN/manifests/jailbreak_validation.json" \
  "$CAMPAIGN" jail_gemma
scripts/validation_workflow.sh prepare-tests \
  "$LLAMA" "$CAMPAIGN/manifests/jailbreak_validation.json" \
  "$CAMPAIGN" jail_llama
scripts/validation_workflow.sh prepare-tests \
  "$QWEN" "$CAMPAIGN/manifests/jailbreak_validation.json" \
  "$CAMPAIGN" jail_qwen
```

Run these test JSONL files with the same `run-index`, `run-local`, or Slurm-array commands. Jailbreaking selection requires the existing LlamaGuard-3 service used by the repository to be reachable at `$LLAMAGUARD_VLLM_SERVER` (default: `http://0.0.0.0:8000/v1`).

## 1. Emit and run validation jobs

From `refusal_direction/`, generate the exhaustive CAA commands:

```bash
python scripts/build_validation_jobs.py \
  --output <OUTPUT_DIR>/caa_select.sh \
  caa-select --model-path /path/to/gemma-2-9b-it \
  --artifact-root /path/to/campaign/artifacts/caa_select
bash <OUTPUT_DIR>/caa_select.sh
```

This emits 1,440 runs: 720 joint ActAdd/KAS runs and 720 Curveball runs. Every method has 40 layer/strength candidates per behavior and prompt scope. A single prompt condition emits 480 runs.

Generate jailbreaking commands once per model:

```bash
python scripts/build_validation_jobs.py \
  --output <OUTPUT_DIR>/gemma_jail_select.sh \
  jail-select --model-path /path/to/gemma-2-9b-it \
  --artifact-root /path/to/campaign/artifacts/jail_select
bash <OUTPUT_DIR>/gemma_jail_select.sh
```

Each model's script contains 81 runs: one unsteered baseline, 40 joint ActAdd/KAS runs, and 40 Curveball runs covering ten strengths, two layers, and two application modes. The baseline result is evaluated once and recorded for both application-mode scopes. These shell files may be partitioned across a scheduler; do not change their arguments.

## 2. Lock selections

After all jobs complete, create manifests. Complete-grid verification is enabled by default.

```bash
python -m pipeline.validation_selection select-caa \
  --candidate-root /path/to/campaign/artifacts/caa_select \
  --output /path/to/campaign/manifests/caa_validation.json

python -m pipeline.validation_selection select-jailbreak \
  --candidate-root /path/to/campaign/artifacts/jail_select \
  --utility-cap-ratio 1.05 \
  --output /path/to/campaign/manifests/jailbreak_validation.json
```

The manifests contain split hashes, the complete grid, every validation metric, fixed parameters, tie-break rules, and selected configurations. Existing manifests are not overwritten unless `--overwrite` is explicitly supplied.

## 3. Run the untouched tests

Generate manifest-locked test commands for the CAA model or each jailbreaking model:

```bash
python scripts/build_validation_jobs.py \
  --output <OUTPUT_DIR>/caa_test.sh \
  locked-test \
  --model-path /path/to/gemma-2-9b-it \
  --manifest manifests/caa_validation.json \
  --artifact-root /path/to/campaign/artifacts/caa_test
bash <OUTPUT_DIR>/caa_test.sh
```

```bash
python scripts/build_validation_jobs.py \
  --output <OUTPUT_DIR>/gemma_jail_test.sh \
  locked-test \
  --model-path /path/to/gemma-2-9b-it \
  --manifest manifests/jailbreak_validation.json \
  --artifact-root /path/to/campaign/artifacts/jail_test
bash <OUTPUT_DIR>/gemma_jail_test.sh
```

Test summaries include `validation_selected`. A test aborts before scoring if its configuration or split fingerprints differ from the manifest. CAA Curveball replay also locks extraction position, kernel, degree, gamma, bias, KPCA components, and inverse-transform alpha, and exports `steer_config_curveball.pt`. A method with no utility-admissible configuration has no generated test job.

Run TinyBenchmarks once for the baseline and once for each saved, selected steer config. The wrapper re-checks the manifest and serialized config before invoking the harness:

```bash
python scripts/run_locked_tinybench.py \
  --manifest manifests/jailbreak_validation.json \
  --model-path /path/to/gemma-2-9b-it \
  --method kas \
  --application-mode prompt_only \
  --layer SELECTED_LAYER \
  --strength SELECTED_STRENGTH \
  --steer-path /path/to/steer_config_kernel.pt \
  --output-path results/gemma_kas_prompt_only
```

Use `--method actadd` with `steer_config_dim.pt`, `--method curveball` with `steer_config_curveball.pt`, `--method chars` with `steer_config_chars.pt` (plus `--n-clusters`), or `--method sas` with `steer_config_sas.pt`. For SAS, `--strength` is the locked coefficient, and the wrapper also verifies the serialized Gemma Scope release, SAE identifier, tau, correction flags, and layer. For the one-time unsteered utility result, use `--method baseline` and omit layer, strength, and steer path.

## 4. Paired confidence intervals

For CAA, compare raw AB files with metric `target_probability`. For jailbreaking, pass the evaluation JSON files and metric `is_jailbreak_llamaguard3`. Run the command twice, for KAS–ActAdd and KAS–baseline:

```bash
python -m pipeline.validation_selection bootstrap \
  --method-predictions KAS.json \
  --comparison-predictions ACTADD_OR_BASELINE.json \
  --metric-key target_probability \
  --n-resamples 10000 \
  --seed 42 \
  --output paired_ci.json
```

The prior best-on-test outputs remain untouched and should be labeled only as oracle diagnostics; the manifest-selected outputs are the main-table results.
