# Language Model Evaluation Harness for Steering

A fork of lm-evaluation-harness for highdim and kernel steering.

---

## Install
```bash
git clone <HARNESS_REPO_URL>
cd lm-evaluation-harness
conda create -n steering-lm-eval python=3.11 -y
conda activate steering-lm-eval
pip install -e ".[dev,hf]"
pip install git+https://github.com/felipemaiapolo/tinyBenchmarks
```

Remember to update peft and acceleration first:

```bash
pip install -U accelerate peft
```

### How to run

After running the scripts on the highdim and kernel repo, there'll be some steering file configs, for example `{run_path}/steer_config/steer_config_kernel.pt`. For kernel steering, there'll be 2 config files for non-sequential (1 actadd, 1 kernel) and 1 kernel config file for sequential steering (currently there's no implementation for sequential actadd steering).

To evaluate a steered model on tinyBenchmarks, run:

```bash
lm_eval --model steered \
--model_args pretrained={model_name},steer_path={path_to_steer_config_file} \
--tasks tinyArc,tinyGSM8k,tinyHellaswag,tinyMMLU,tinyTruthfulQA,tinyWinogrande \
--device cuda:0 \
--batch_size auto
```

Fill model_name with the HuggingFace directory for that model (e.g. google/gemma-2-9b-it) and the path_to_steering_config_file with the current FULL DIRECTORY to that file (using partial/inherited ones won't work)

Add environment variables if necessary.
