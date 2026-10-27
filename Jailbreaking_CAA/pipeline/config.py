
import os
import hashlib

from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Config:
    model_alias: str
    model_path: str
    artifact_root: Optional[str] = None
    seed: int = 42
    kernel_type: str = "rbf"
    kernel_params: dict = None
    eta: float = 1.0
    lambda_reg: float = 0.1
    pos: int = -1  # position of the layer to intervene
    layer: int = 24  # layer index to intervene
    use_mean_act: bool = False
    step_count: int = 1  # number of steering steps to apply
    prompt_only: bool = False  # apply steering only to prompt tokens
    n_train: int = 128
    n_val: int = 32
    filter_train: bool = True
    evaluation_datasets: Tuple[str] = ("jailbreakbench",)
    max_new_tokens: int = 512
    # jailbreak_eval_methodologies: Tuple[str] = ("substring_matching")
    jailbreak_eval_methodologies: Tuple[str] = ("substring_matching", "llamaguard2")
    ce_loss_batch_size: int = 2
    ce_loss_n_batches: int = 2048
    kernel_extraction_source: str = "legacy"
    steer_method: str = "kernel"
    sae_release: Optional[str] = None
    sae_width: str = "16k"
    sas_feature_freq_threshold: float = 0.01
    sas_extraction_source: str = "caa"
    sas_coefficient: float = -1.0
    sas_use_delta_correction: bool = True
    sas_apply_post_activation: bool = True
    curveball_extraction_source: str = "caa"
    curveball_k: int = 20
    curveball_kernel: str = "poly"
    curveball_degree: int = 3
    curveball_gamma: Optional[float] = None
    curveball_coef0: float = 1.0
    curveball_alpha: float = 1.0
    curveball_fit_inverse_alpha: float = 1.0
    chars_extraction_source: str = "caa"
    chars_clusters: int = 5
    chars_alpha: float = 1.0
    chars_state_path: Optional[str] = None
    caa_dataset_root: str = "caa"
    caa_behavior: str = "refusal"
    evaluation_protocol: str = "legacy"
    caa_eval_types: Tuple[str, ...] = ("ab", "truthful_qa", "mmlu")
    caa_eval_system_prompt: Optional[str] = None
    caa_eval_open_ended_score: bool = False
    caa_open_ended_judge_model: Optional[str] = None
    experiment_phase: str = "standard"
    selection_manifest: Optional[str] = None
    evaluation_interventions: Optional[Tuple[str, ...]] = None
    split_seed: int = 42
    caa_validation_size: int = 50
    caa_eval_split: str = "test"
    jailbreak_eval_split: str = "test"
    jailbreak_utility_size: int = 256

    def artifact_path(self) -> str:
        artifact_root = self.artifact_root or os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "runs"
        )
        path_parts = [
            os.path.abspath(artifact_root),
            f"seed-{self.seed}",
            f"use_mean_act-{self.use_mean_act}",
            f"kernel_type-{self.kernel_type}",
            f"kernel_params-{self.kernel_params}",
            f"eta-{self.eta}",
            f"lambda_reg-{self.lambda_reg}",
            f"pos-{self.pos}",
            f"layer-{self.layer}",
            f"step_count-{self.step_count}",
            f"prompt_only-{self.prompt_only}",
        ]

        if self.n_train != 128:
            path_parts.append(f"n_train-{self.n_train}")

        if self.experiment_phase != "standard":
            path_parts.extend(
                [
                    f"experiment_phase-{self.experiment_phase}",
                    f"caa_eval_split-{self.caa_eval_split}",
                    f"jailbreak_eval_split-{self.jailbreak_eval_split}",
                ]
            )
            if self.evaluation_interventions is not None:
                path_parts.append(
                    "evaluation_interventions-"
                    + "-".join(self.evaluation_interventions)
                )

        if self.kernel_extraction_source != "legacy" or self.evaluation_protocol != "legacy":
            path_parts.extend(
                [
                    f"kernel_extraction_source-{self.kernel_extraction_source}",
                    f"caa_dataset_root-{self.caa_dataset_root}",
                    f"caa_behavior-{self.caa_behavior}",
                    f"evaluation_protocol-{self.evaluation_protocol}",
                ]
            )
            if self.caa_eval_system_prompt is not None:
                path_parts.append(f"caa_eval_system_prompt-{self.caa_eval_system_prompt}")
            if self.caa_eval_open_ended_score:
                path_parts.append(f"caa_eval_open_ended_score-{self.caa_eval_open_ended_score}")

        if self.steer_method == "sas":
            path_parts.extend(
                [
                    f"steer_method-{self.steer_method}",
                    f"sas_extraction_source-{self.sas_extraction_source}",
                    f"sae_width-{self.sae_width}",
                    f"sas_feature_freq_threshold-{self.sas_feature_freq_threshold}",
                    f"sas_coefficient-{self.sas_coefficient}",
                ]
            )
            if self.sae_release is not None:
                path_parts.append(f"sae_release-{self.sae_release}")
            path_parts.append(f"sas_use_delta_correction-{self.sas_use_delta_correction}")
            path_parts.append(f"sas_apply_post_activation-{self.sas_apply_post_activation}")
        elif self.steer_method == "curveball":
            path_parts.extend(
                [
                    f"steer_method-{self.steer_method}",
                    f"curveball_extraction_source-{self.curveball_extraction_source}",
                    f"curveball_k-{self.curveball_k}",
                    f"curveball_kernel-{self.curveball_kernel}",
                    f"curveball_degree-{self.curveball_degree}",
                    f"curveball_gamma-{self.curveball_gamma}",
                    f"curveball_coef0-{self.curveball_coef0}",
                    f"curveball_alpha-{self.curveball_alpha}",
                    f"curveball_fit_inverse_alpha-{self.curveball_fit_inverse_alpha}",
                ]
            )
        elif self.steer_method == "chars":
            path_parts.extend(
                [
                    f"steer_method-{self.steer_method}",
                    f"chars_extraction_source-{self.chars_extraction_source}",
                    f"chars_clusters-{self.chars_clusters}",
                    f"chars_alpha-{self.chars_alpha}",
                    "chars_recipe-chars_equal_cov_v2",
                    "chars_hookpoint-block_output",
                ]
            )
            if self.chars_state_path is not None:
                state_path_id = hashlib.sha256(
                    os.path.abspath(self.chars_state_path).encode("utf-8")
                ).hexdigest()[:16]
                path_parts.append(f"chars_state_path-{state_path_id}")

        path_parts.append(self.model_alias)
        return os.path.join(*path_parts)
