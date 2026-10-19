"""Command-line options of `python -m pipeline.run_pipeline`."""

import argparse
import ast
import json

from pipeline.submodules.evaluate_caa import CAA_EVAL_TYPES


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y", "t"}:
        return True
    if lowered in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_kernel_params(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value

    text = value.strip()
    if text == "" or text.lower() == "none":
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Backward-compatible fallback for Python-literal dict strings.
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            raise argparse.ArgumentTypeError(
                'Invalid --kernel_params. Use a JSON object, e.g. \'{"sigma": 48}\'.'
            ) from exc

    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            'Invalid --kernel_params. Value must decode to a dictionary, e.g. \'{"sigma": 48}\'.'
        )
    return parsed


def parse_arguments():
    """Parse model path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument(
        '--artifact_root',
        type=str,
        default=None,
        help='Optional isolated root for all artifacts in this campaign.',
    )
    parser.add_argument('--kernel_type', type=str, default='rbf', help='Type of kernel to use')
    parser.add_argument(
        '--kernel_params',
        type=parse_kernel_params,
        default=None,
        help='Kernel parameters as JSON dict, e.g. \'{"sigma": 48}\'.',
    )
    parser.add_argument('--eta', type=float, default=1.0, help='Eta parameter for kernelized steering')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='Lambda regularization parameter for kernelized steering')
    parser.add_argument('--pos', type=int, default=None, help='Position of the layer to intervene. Defaults: -2 for CAA extraction, -1 otherwise')
    parser.add_argument('--layer', type=int, default=24, help='Layer index to intervene')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling datasets')
    parser.add_argument('--n_train', type=int, default=128, help='Number of harmful/harmless training examples to sample')
    parser.add_argument('--step_count', type=int, default=1, help='Number of steering steps to apply')
    parser.add_argument("--use_mean_act", type=bool, default=False, required=False, help='Intervene on all layers instead of a single layer')
    parser.add_argument("--prompt_only", action='store_true', help='Apply steering only to prompt tokens, not generated tokens')
    parser.add_argument("--skip_baseline", action='store_true', help='No effect; accepted for compatibility with existing scripts')
    parser.add_argument(
        '--steer_method',
        type=str,
        default='kernel',
        choices=['kernel', 'curveball', 'chars', 'sas'],
        help='Steering method: kernel (KAS + ActAdd), curveball, chars (CHaRS), or sas.',
    )
    parser.add_argument(
        '--kernel_extraction_source',
        type=str,
        default='legacy',
        choices=['legacy', 'caa'],
        help='Dataset source for kernel extraction. Default: legacy.',
    )
    parser.add_argument(
        '--sae_release',
        type=str,
        default=None,
        help='Optional SAE release override for SAS mode.',
    )
    parser.add_argument(
        '--sae_width',
        type=str,
        default='16k',
        help='SAE width identifier for SAS mode.',
    )
    parser.add_argument(
        '--sas_feature_freq_threshold',
        type=float,
        default=0.01,
        help='Feature frequency threshold tau for SAS extraction.',
    )
    parser.add_argument(
        '--sas_extraction_source',
        type=str,
        default=None,
        choices=['caa', 'legacy'],
        help='SAS extraction source. Default: caa for steer_method=sas, else legacy.',
    )
    parser.add_argument(
        '--sas_coefficient',
        type=float,
        default=-1.0,
        help='Scalar SAS steering coefficient; validation phases enforce the registered experiment grid. Default: -1.0.',
    )
    parser.add_argument(
        '--sas_use_delta_correction',
        type=str_to_bool,
        default=True,
        required=False,
        help='Use decode residual correction in SAS hook.',
    )
    parser.add_argument(
        '--sas_apply_post_activation',
        type=str_to_bool,
        default=True,
        required=False,
        help='Apply SAE activation after sparse steering update.',
    )
    parser.add_argument(
        '--curveball_extraction_source',
        type=str,
        default=None,
        choices=['caa', 'legacy'],
        help='Curveball extraction source. Default: caa for steer_method=curveball, else legacy.',
    )
    parser.add_argument(
        '--curveball_k',
        type=int,
        default=20,
        help='Number of KernelPCA components for curveball.',
    )
    parser.add_argument(
        '--curveball_kernel',
        type=str,
        default='poly',
        choices=['poly', 'rbf', 'sigmoid', 'cosine', 'linear'],
        help='Kernel type for curveball KernelPCA.',
    )
    parser.add_argument(
        '--curveball_degree',
        type=int,
        default=3,
        help='Polynomial degree for curveball when kernel=poly.',
    )
    parser.add_argument(
        '--curveball_gamma',
        type=float,
        default=None,
        help='Kernel coefficient gamma for curveball KernelPCA.',
    )
    parser.add_argument(
        '--curveball_coef0',
        type=float,
        default=1.0,
        help='Independent term coef0 for curveball polynomial/sigmoid kernels.',
    )
    parser.add_argument(
        '--curveball_alpha',
        type=float,
        default=1.0,
        help='Steering coefficient in curveball KPCA reduced space.',
    )
    parser.add_argument(
        '--curveball_fit_inverse_alpha',
        type=float,
        default=1.0,
        help='Ridge alpha for KernelPCA inverse transform in curveball.',
    )
    parser.add_argument(
        '--chars_extraction_source',
        choices=['caa', 'legacy'],
        default=None,
        help='CHaRS reference source. Defaults to caa for CAA and legacy for jailbreak.',
    )
    parser.add_argument(
        '--chars_clusters',
        type=int,
        default=5,
        help=(
            'Number of independently fitted source/target CHaRS clusters. '
            'Validation requires the model-specific locked value.'
        ),
    )
    parser.add_argument(
        '--chars_alpha',
        type=float,
        default=1.0,
        help='CHaRS activation-addition coefficient.',
    )
    parser.add_argument(
        '--chars_state_path',
        default=None,
        help='Optional pre-fitted chars_state.pt; recipe, metadata, and fingerprint are verified.',
    )
    parser.add_argument('--caa_dataset_root', type=str, default='caa', help='Root folder under dataset/ for CAA datasets')
    parser.add_argument('--caa_behavior', type=str, default='refusal', help='CAA behavior file name, e.g. refusal')
    parser.add_argument(
        '--evaluation_protocol',
        type=str,
        default=None,
        choices=['legacy', 'caa', 'both'],
        help='Evaluation protocol. Default: caa for steer_method=sas or curveball, else legacy.',
    )
    parser.add_argument(
        '--caa_eval_types',
        type=str,
        nargs='+',
        default=['ab', 'truthful_qa', 'mmlu'],
        choices=list(CAA_EVAL_TYPES),
        help='CAA eval types to run when protocol includes caa.',
    )
    parser.add_argument(
        '--caa_eval_system_prompt',
        type=str,
        default='none',
        choices=['none', 'pos', 'neg'],
        help='Optional CAA system prompt setting for evaluation.',
    )
    parser.add_argument(
        '--caa_eval_open_ended_score',
        type=str_to_bool,
        default=False,
        required=False,
        help='If true, score open-ended outputs with an OpenAI judge model.',
    )
    parser.add_argument(
        '--caa_open_ended_judge_model',
        type=str,
        default=None,
        help='OpenAI judge model name for open-ended CAA scoring (required if scoring is enabled).',
    )
    parser.add_argument(
        "--experiment_phase",
        choices=["standard", "select", "test"],
        default="standard",
        help="Run the legacy workflow, validation selection, or a manifest-locked test.",
    )
    parser.add_argument(
        "--selection_manifest",
        default=None,
        help="Immutable validation manifest required in test phase.",
    )
    parser.add_argument(
        "--evaluation_interventions",
        nargs="+",
        choices=["baseline", "actadd", "kernel", "curveball", "chars", "sas"],
        default=None,
        help="Interventions to evaluate in this run; baseline is evaluated once.",
    )
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--caa_validation_size", type=int, default=50)
    parser.add_argument(
        "--caa_eval_split", choices=["validation", "test"], default="test"
    )
    parser.add_argument(
        "--jailbreak_eval_split", choices=["validation", "test"], default="test"
    )
    parser.add_argument("--jailbreak_utility_size", type=int, default=256)
    parser.add_argument(
        "--ce_loss_batch_size",
        type=int,
        default=2,
        help="Batch size for benign utility perplexity evaluation.",
    )
    return parser.parse_args()
