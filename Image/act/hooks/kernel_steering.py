# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import math
import typing as t

import torch

from act.hooks.intervention_hook import InterventionHook


def _move_tensor(
    tensor: torch.Tensor,
    device: t.Optional[str],
    dtype: torch.dtype,
) -> torch.Tensor:
    if device is not None:
        tensor = tensor.to(device)
    return tensor.to(dtype)


def rbf_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float,
    bandwidth_mixture: bool = True,
) -> torch.Tensor:
    x_norm_sq = (x**2).sum(dim=1, keepdim=True)
    y_norm_sq = (y**2).sum(dim=1, keepdim=True).T
    distances_sq = x_norm_sq + y_norm_sq - 2 * (x @ y.T)
    distances_sq = distances_sq.clamp_min(0)

    if bandwidth_mixture:
        scaled_sigmas = [s * sigma for s in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]]
        return sum(
            torch.exp(-distances_sq / (2 * scaled_sigma**2))
            for scaled_sigma in scaled_sigmas
        )
    return torch.exp(-distances_sq / (2 * sigma**2))


def laplacian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    cdist_x = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    cdist_y = y.float() if y.dtype in (torch.float16, torch.bfloat16) else y
    kernel = torch.exp(-torch.cdist(cdist_x, cdist_y, p=1) / sigma)
    return kernel.to(x.dtype)


def laplacian_kernel_gradient(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    signs = torch.sign(x.unsqueeze(1) - y.unsqueeze(0))
    return (-(1.0 / sigma) * kernel.unsqueeze(-1) * signs).sum(dim=1)


def compute_median_heuristic_sigma(source: torch.Tensor, target: torch.Tensor) -> float:
    combined = torch.cat([source, target], dim=0)
    cdist_input = (
        combined.float()
        if combined.dtype in (torch.float16, torch.bfloat16)
        else combined
    )
    dists_sq = torch.cdist(cdist_input, cdist_input, p=2) ** 2
    positive_dists = dists_sq[dists_sq > 0]
    if positive_dists.numel() > 0:
        return torch.sqrt(torch.median(positive_dists)).item()
    return float(source.shape[1] ** 0.5)


def compute_kernelized_steering_gradient(
    x: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_type: str = "rbf",
    kernel_params: t.Optional[t.Dict[str, t.Any]] = None,
) -> torch.Tensor:
    if kernel_params is None:
        kernel_params = {"sigma": 1.0} if kernel_type in {"rbf", "laplacian"} else {}

    source = source.to(device=x.device, dtype=x.dtype)
    target = target.to(device=x.device, dtype=x.dtype)
    n_source = source.shape[0]
    n_target = target.shape[0]

    if kernel_type == "rbf":
        sigma = float(kernel_params.get("sigma", 1.0))
        k_target = rbf_kernel(x, target, sigma, bandwidth_mixture=True)
        k_source = rbf_kernel(x, source, sigma, bandwidth_mixture=True)

        weighted_target = k_target @ target
        weighted_source = k_source @ source
        sum_target = k_target.sum(dim=1, keepdim=True)
        sum_source = k_source.sum(dim=1, keepdim=True)

        return (1.0 / (sigma**2)) * (
            (1.0 / n_target) * weighted_target
            - (1.0 / n_source) * weighted_source
            - ((sum_target / n_target - sum_source / n_source) * x)
        )

    if kernel_type == "polynomial":
        degree = int(kernel_params.get("degree", 2))
        offset = float(kernel_params.get("offset", 1.0))
        inner_target = x @ target.T + offset
        inner_source = x @ source.T + offset
        weights_target = degree * (inner_target ** (degree - 1))
        weights_source = degree * (inner_source ** (degree - 1))
        return (1.0 / n_target) * (weights_target @ target) - (
            1.0 / n_source
        ) * (weights_source @ source)

    if kernel_type == "laplacian":
        sigma = float(kernel_params.get("sigma", 1.0))
        k_target = laplacian_kernel(x, target, sigma)
        k_source = laplacian_kernel(x, source, sigma)
        grad_target = laplacian_kernel_gradient(x, target, k_target, sigma)
        grad_source = laplacian_kernel_gradient(x, source, k_source, sigma)
        return (1.0 / n_target) * grad_target - (1.0 / n_source) * grad_source

    raise ValueError(f"Unknown kernel type: {kernel_type}")


class KernelSteeringHook(InterventionHook):
    def __init__(
        self,
        module_name: str,
        device: str = None,
        intervention_position: str = "all",
        dtype: torch.dtype = torch.float32,
        kernel_type: str = "rbf",
        kernel_params: t.Optional[t.Dict[str, t.Any]] = None,
        eta: float = 1.0,
        lambda_reg: float = 0.0,
        step_count: int = 1,
        strength: float = 1.0,
        prompt_only: bool = False,
        **kwargs,
    ):
        super().__init__(
            module_name=module_name,
            device=device,
            intervention_position=intervention_position,
            dtype=dtype,
        )
        self.kernel_type = str(kernel_type)
        self.kernel_params = dict(kernel_params) if kernel_params is not None else None
        self.eta = float(eta)
        self.lambda_reg = float(lambda_reg)
        self.step_count = int(step_count)
        self.strength = float(strength)
        self.prompt_only = bool(prompt_only)

        self.register_buffer("source", torch.empty(0))
        self.register_buffer("target", torch.empty(0))

    def __str__(self):
        return (
            f"KernelSteering("
            f"module_name={self.module_name}, "
            f"kernel_type={self.kernel_type}, "
            f"eta={self.eta:0.2f}, "
            f"strength={self.strength:0.2f}"
            f")"
        )

    def _resolved_kernel_params(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> t.Dict[str, t.Any]:
        if self.kernel_params is not None:
            return dict(self.kernel_params)
        if self.kernel_type == "rbf":
            return {"sigma": compute_median_heuristic_sigma(source, target)}
        if self.kernel_type == "laplacian":
            return {"sigma": 1.0}
        return {}

    def state_dict(self, *args, **kwargs) -> t.Dict[str, t.Any]:
        state = super().state_dict(*args, **kwargs)
        state.update(
            {
                "kernel_type": self.kernel_type,
                "kernel_params": self.kernel_params,
            }
        )
        return state

    def load_state_dict(
        self,
        state_dict: t.Mapping[str, t.Any],
        state_path=None,
        strict: bool = True,
        assign: bool = False,
    ) -> None:
        self.source = _move_tensor(state_dict["source"], self.device, self.dtype)
        self.target = _move_tensor(state_dict["target"], self.device, self.dtype)
        self.kernel_type = str(state_dict.get("kernel_type", self.kernel_type))
        self.kernel_params = state_dict.get("kernel_params", self.kernel_params)
        self._post_load()

    def fit(self, responses: torch.Tensor, labels: torch.Tensor, **kwargs) -> None:
        responses = responses.to(self.dtype)
        labels = labels.to(torch.bool)
        if labels.sum() == 0 or (~labels).sum() == 0:
            raise ValueError("Kernel steering requires at least one source and target response.")

        self.source = _move_tensor(responses[labels], self.device, self.dtype)
        self.target = _move_tensor(responses[~labels], self.device, self.dtype)
        self.kernel_params = self._resolved_kernel_params(self.source, self.target)
        self._post_load()

    def forward(self, module, input_, output):
        if output.ndim == 4:
            raise NotImplementedError("KernelSteeringHook currently supports text activations only.")
        if output.ndim != 3:
            raise NotImplementedError(
                f"KernelSteeringHook expects [batch, seq, d_model] activations, got {tuple(output.shape)}."
            )
        if self.prompt_only and output.shape[1] == 1:
            return output
        if self.strength == 0.0 or self.eta == 0.0 or self.step_count < 1:
            return output

        batch_size, seq_len, d_model = output.shape
        original_dtype = output.dtype
        x = output.reshape(-1, d_model).to(torch.float32)
        x0 = x.clone()

        for step in range(self.step_count):
            grad = compute_kernelized_steering_gradient(
                x=x,
                source=self.source,
                target=self.target,
                kernel_type=self.kernel_type,
                kernel_params=self.kernel_params,
            )
            update = grad - self.lambda_reg * (x - x0)
            update = torch.nn.functional.normalize(update, dim=-1)
            x = x + self.strength * self.eta / math.sqrt(step + 1.0) * update

        return x.view(batch_size, seq_len, d_model).to(original_dtype)
