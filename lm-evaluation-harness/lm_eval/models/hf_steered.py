from contextlib import contextmanager
from functools import partial
import math
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Union

from sympy import Float
import torch
from peft.peft_model import PeftModel
from torch import Tensor, nn
from transformers import PreTrainedModel

from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@contextmanager
def steer(
    model: Union[PreTrainedModel, PeftModel], hook_to_steer: dict[str, Callable]
) -> Generator[None, Any, None]:
    """
    Context manager that temporarily hooks models and steers them.

    Args:
        model: The transformer model to hook
        hook_to_steer: Dictionary mapping hookpoints to steering functions

    Yields:
        None
    """

    def create_hook(hookpoint: str):
        def hook_fn(module: nn.Module, input: Any, output: Tensor):
            # If output is a tuple (like in some transformer layers), take first element
            if isinstance(output, tuple):
                output = (hook_to_steer[hookpoint](output[0]), *output[1:])  # type: ignore
            else:
                output = hook_to_steer[hookpoint](output)

            return output

        return hook_fn

    handles = []
    hookpoints = list(hook_to_steer.keys())

    for name, module in model.base_model.named_modules():
        if name in hookpoints:
            handle = module.register_forward_hook(create_hook(name))
            handles.append(handle)

    if len(handles) != len(hookpoints):
        raise ValueError(f"Not all hookpoints could be resolved: {hookpoints}")

    try:
        yield None
    finally:
        for handle in handles:
            handle.remove()


@register_model("steered")
class SteeredModel(HFLM):
    hook_to_steer: dict[str, Callable]

    def __init__(
        self,
        pretrained: str,
        steer_path: str,
        device: Optional[str] = None,
        **kwargs,
    ):
        """
        HFLM with a steered forward pass.

        To load steering vectors directly, provide the path to a pytorch (.pt) file with content in the following format:

        {
            hookpoint: {
                "steering_vector": <torch.Tensor>,
                "steering_coefficient": <float>,
                "action": <Literal["add", "clamp"]>,
                "bias": <torch.Tensor | None>,
                "head_index": <int | None>,
            },
            ...
        }

        To derive steering vectors from a sparse model loadable with sparsify or sae_lens,
        provide the path to a CSV file with the following columns (example rows are provided below):

        loader,action,sparse_model,hookpoint,feature_index,steering_coefficient,head_index,sae_id,description,
        sparsify,add,EleutherAI/sae-pythia-70m-32k,layers.3,30,10.0,,,,
        sae_lens,add,gemma-scope-2b-pt-res-canonical,layers.20,12082,240.0,,layer_20/width_16k/canonical,increase dogs,
        """
        super().__init__(pretrained=pretrained, device=device, **kwargs)

        if steer_path.endswith(".pt") or steer_path.endswith(".pth"):
            with open(steer_path, "rb") as f:
                steer_config: dict[str, dict[str, Any]] = torch.load(
                    f, weights_only=True
                )
        elif steer_path.endswith(".csv"):
            steer_config = self.derive_steer_config(steer_path)
        else:
            raise ValueError(f"Unknown steer file type: {steer_path}")

        hook_to_steer = {}
        for hookpoint, steer_info in steer_config.items():
            action = steer_info["action"]
            if action not in ("kernel", "rks"):
                steering_vector = (
                    steer_info["steering_vector"].to(self.device).to(self.model.dtype)
                )
            steering_coefficient = float(steer_info.get("steering_coefficient", 1.0))
            head_index = steer_info.get("head_index", None)
            bias = steer_info.get("bias", None)
            if bias is not None:
                bias = bias.to(self.device).to(self.model.dtype)
            prompt_only = steer_info.get("prompt_only", False)

            if action == "add":
                # Steer the model by adding a multiple of a steering vector to all sequence positions.
                assert bias is None, "Bias is not supported for the `add` action."
                hook_to_steer[hookpoint] = partial(
                    self.add,
                    vector=steering_vector * steering_coefficient,
                    head_index=head_index,
                    prompt_only=prompt_only,
                )
            elif action == "ablate":
                assert bias is None, "Bias is not supported for the `ablate` action."
                hook_to_steer[hookpoint] = partial(
                    self.ablate,
                    direction=steering_vector,
                    head_index=head_index,
                    prompt_only=prompt_only,
                )
            elif action == "clamp":
                # Steer the model by clamping the activations to a value in the direction of the steering vector.
                hook_to_steer[hookpoint] = partial(
                    self.clamp,
                    direction=steering_vector / torch.norm(steering_vector),
                    value=steering_coefficient,
                    bias=bias,
                    head_index=head_index,
                )
            elif action == "high_dim":
                upscale_matrix = steer_info["upscale_matrix"].to(self.device).to(self.model.dtype)
                downscale_matrix = steer_info["downscale_matrix"].to(self.device).to(self.model.dtype)
                nonlinearity = steer_info.get("nonlinearity", "leaky_relu")
                leaky_relu_slope = steer_info.get("leaky_relu_slope", 0.7)

                hook_to_steer[hookpoint] = partial(
                    self.high_dim,
                    direction=steering_vector * steering_coefficient,
                    upscale_matrix=upscale_matrix,
                    downscale_matrix=downscale_matrix,
                    nonlinearity=nonlinearity,
                    leaky_relu_slope=leaky_relu_slope,
                    head_index=head_index,
                    prompt_only=prompt_only,
                )
            elif action == "high_dim_ablate":
                upscale_matrix = steer_info["upscale_matrix"].to(self.device).to(self.model.dtype)
                downscale_matrix = steer_info["downscale_matrix"].to(self.device).to(self.model.dtype)
                nonlinearity = steer_info.get("nonlinearity", "leaky_relu")
                leaky_relu_slope = steer_info.get("leaky_relu_slope", 0.7)

                hook_to_steer[hookpoint] = partial(
                    self.high_dim_ablate,
                    direction=steering_vector,
                    upscale_matrix=upscale_matrix,
                    downscale_matrix=downscale_matrix,
                    nonlinearity=nonlinearity,
                    leaky_relu_slope=leaky_relu_slope,
                    head_index=head_index,
                    prompt_only=prompt_only,
                )
            elif action == "kernel":
                source = steer_info["source"].to(self.device).to(self.model.dtype)
                target = steer_info["target"].to(self.device).to(self.model.dtype)
                kernel_type = steer_info.get("kernel_type", "rbf")
                kernel_params = steer_info.get("kernel_params", {})
                eta = steer_info.get("eta", 1.0)
                lambda_reg = steer_info.get("lambda_reg", 0.1)
                step_count = steer_info.get("step_count", 10)

                hook_to_steer[hookpoint] = partial(
                    self.kernel,
                    source=source,
                    target=target,
                    kernel_type=kernel_type,
                    kernel_params=kernel_params,
                    eta=eta,
                    lambda_reg=lambda_reg,
                    step_count=step_count,
                    head_index=head_index,
                    prompt_only=prompt_only,
                )

            elif action == "rks":
                state = steer_info["state"]
                hook_to_steer[hookpoint] = partial(
                    self.rks,
                    witness_direction=state["witness_direction"].to(self.device),
                    steer_direction=state["steer_direction"].to(self.device),
                    mean_shift=state["mean_shift"].to(self.device),
                    threshold=float(state["threshold"].item()),
                    mean_shift_norm=float(state["mean_shift_norm"].item()),
                    degree=int(state["degree"]),
                    alpha=float(steer_info.get("alpha", 1.0)),
                    gain_exponent=int(steer_info.get("gain_exponent", 0)),
                    gain_scale=str(steer_info.get("gain_scale", "budget")),
                    gain_normalizer=float(steer_info.get("gain_normalizer", 1.0)),
                    gain_clip=steer_info.get("gain_clip", None),
                    field_weight=float(steer_info.get("field_weight", 0.0)),
                    field_scale=str(steer_info.get("field_scale", "raw")),
                    mask_update=str(steer_info.get("mask_update", "recompute")),
                    steps=int(steer_info.get("steps", 1)),
                    step_decay=float(steer_info.get("step_decay", 1.0)),
                    lambda_reg=float(steer_info.get("lambda_reg", 0.0)),
                    prompt_only=prompt_only,
                )

            else:
                raise ValueError(f"Unknown hook type: {action}")

        self.hook_to_steer = hook_to_steer

    @classmethod
    def derive_steer_config(cls, steer_path: str):
        """Derive a dictionary of steering vectors from sparse model(/s) specified in a CSV file."""
        import pandas as pd

        df = pd.read_csv(steer_path)
        steer_data: dict[str, dict[str, Any]] = {}

        if any(df["loader"] == "sparsify"):
            from sparsify import SparseCoder
        if any(df["loader"] == "sae_lens"):
            from sae_lens import SAE

            sae_cache = {}

            def load_from_sae_lens(sae_release: str, sae_id: str):
                cache_key = (sae_release, sae_id)
                if cache_key not in sae_cache:
                    sae_cache[cache_key] = SAE.from_pretrained(sae_release, sae_id)[0]

                return sae_cache[cache_key]

        for _, row in df.iterrows():
            action = row.get("action", "add")
            sparse_name = row["sparse_model"]
            hookpoint = row["hookpoint"]
            feature_index = int(row["feature_index"])
            steering_coefficient = float(row["steering_coefficient"])
            loader = row.get("loader", "sparsify")

            if loader == "sparsify":
                name_path = Path(sparse_name)

                sparse_coder = (
                    SparseCoder.load_from_disk(name_path / hookpoint)
                    if name_path.exists()
                    else SparseCoder.load_from_hub(sparse_name, hookpoint)
                )
                assert sparse_coder.W_dec is not None

                steering_vector = sparse_coder.W_dec[feature_index]
                bias = sparse_coder.b_dec

            elif loader == "sae_lens":
                sparse_coder = load_from_sae_lens(
                    sae_release=sparse_name, sae_id=row["sae_id"]
                )
                steering_vector = sparse_coder.W_dec[feature_index]
                bias = sparse_coder.b_dec
                if hookpoint == "" or pd.isna(hookpoint):
                    hookpoint = sparse_coder.cfg.hook_name
            else:
                raise ValueError(f"Unknown loader: {loader}")

            steer_data[hookpoint] = {
                "action": action,
                "steering_coefficient": steering_coefficient,
                "steering_vector": steering_vector,
                "bias": bias,
            }

        return steer_data

    @classmethod
    def add(
        cls,
        acts: Tensor,
        vector: Tensor,
        prompt_only: bool,
        head_index: Optional[int],
    ):
        """Adds the given vector to the activations.

        Args:
            acts (Tensor): The activations tensor to edit of shape [batch, pos, ..., features]
            vector (Tensor): A vector to add of shape [features]
            head_index (int | None): Optional attention head index to add to
            prompt_only (bool): Whether to apply steering only to prompt tokens
        """
        if prompt_only:
            batch_size, seq_len = acts.shape[:2]
            if seq_len <= 1:
                return acts
            
        if head_index is not None:
            acts[:, :, head_index, :] = acts[:, :, head_index, :] + vector
        else:
            acts = acts + vector

        return acts

    @classmethod
    def ablate(
        cls,
        acts: Tensor,
        direction: Tensor,
        prompt_only: bool,
        head_index: Optional[int],
    ):
        if prompt_only:
            batch_size, seq_len = acts.shape[:2]
            if seq_len <= 1:
                return acts

        direction = direction.to(acts)
        direction = direction / (torch.norm(direction, p=2).clamp(min=1e-8))

        if head_index is not None:
            x = acts[:, :, head_index, :]
            proj = torch.sum(x * direction, dim=-1, keepdim=True)
            ablated = acts.clone()
            ablated[:, :, head_index, :] = x - proj * direction
            return ablated

        proj = torch.sum(acts * direction, dim=-1, keepdim=True)
        return acts - proj * direction

    @classmethod
    def clamp(
        cls,
        acts: Tensor,
        direction: Tensor,
        value: float,
        head_index: Optional[int],
        bias: Optional[Tensor] = None,
    ):
        """Clamps the activations to a given value in a specified direction. The direction
        must be a unit vector.

        Args:
            acts (Tensor): The activations tensor to edit of shape [batch, pos, ..., features]
            direction (Tensor): A direction to clamp of shape [features]
            value (float): Value to clamp the direction to
            head_index (int | None): Optional attention head index to clamp
            bias (Tensor | None): Optional bias to add to the activations

        Returns:
            Tensor: The modified activations with the specified direction clamped
        """
        if bias is not None:
            acts = acts - bias

        if head_index is not None:
            x = acts[:, :, head_index, :]
            proj = (x * direction).sum(dim=-1, keepdim=True)
            assert proj == acts @ direction

            clamped = acts.clone()
            clamped[:, :, head_index, :] = x + direction * (value - proj)
        else:
            proj = torch.sum(acts * direction, dim=-1, keepdim=True)
            clamped = acts + direction * (value - proj)

        if bias is not None:
            return clamped + bias

        return clamped

    @classmethod
    def high_dim(
        cls,
        acts: Tensor,
        direction: Tensor,
        upscale_matrix: Tensor,
        downscale_matrix: Tensor,
        nonlinearity: str,
        leaky_relu_slope: float,
        prompt_only: bool,
        head_index: Optional[int],
    ):
        """Projects the activations into a higher-dimensional space, applying the nonlinear function,
        steer them along the given direction, applying the inverse nonlinear function, and projects them back down.

        Args:
            acts (Tensor): The activations tensor to edit of shape [batch, pos, ..., features]
            direction (Tensor): A direction to steer of shape [high_dim]
            upscale_matrix (Tensor): Matrix to project activations into high-dimensional space of shape [features, high_dim]
            downscale_matrix (Tensor): Matrix to project activations back to original space of shape [high_dim, features]
            nonlinearity (str): Nonlinear function to apply in high-dimensional space
            leaky_relu_slope (float): Slope for leaky ReLU nonlinearity
            head_index (int | None): Optional attention head index to steer
        """

        def nonlinearity_fn(activation, nonlinearity="leaky_relu", leaky_relu_slope=0.5):
            if nonlinearity == "leaky_relu":
                return torch.nn.functional.leaky_relu(activation, negative_slope=leaky_relu_slope) 
            elif nonlinearity == "skip_softplus":
                sp = torch.nn.functional.softplus(activation, threshold=20).to(activation)
                return activation + sp
            elif nonlinearity == "none":
                return activation

        def skip_softplus_inverse(x: torch.Tensor, low_thresh=-10, high_thresh=10) -> torch.Tensor:
            """
            Compute the inverse of f(x) = x + ln(1+e^x).
            """
            exact = torch.log((torch.sqrt(1 + 4 * torch.exp(x)) - 1) / 2)

            approx = torch.where(x > high_thresh, x / 2, exact)
            approx = torch.where(x < low_thresh, x, approx)

            return approx
        
        def denonlinearity_fn(activation, nonlinearity="leaky_relu", leaky_relu_slope=0.5):
            if nonlinearity == "leaky_relu":
                activation = activation.clone()
                activation[activation < 0] /= leaky_relu_slope
                return activation
            elif nonlinearity == "skip_softplus":
                return skip_softplus_inverse(activation)
            elif nonlinearity == "none":
                return activation
            
        if prompt_only:
            batch_size, seq_len = acts.shape[:2]
            if seq_len <= 1:
                return acts

        upscale_matrix_device = upscale_matrix.to(acts)
        
        highdim_acts = acts @ upscale_matrix_device
        highdim_acts = nonlinearity_fn(highdim_acts, nonlinearity, leaky_relu_slope)
        if head_index is not None:
            highdim_acts[:, :, head_index, :] = highdim_acts[:, :, head_index, :] + direction
        else:
            highdim_acts = highdim_acts + direction
        highdim_acts = denonlinearity_fn(highdim_acts, nonlinearity, leaky_relu_slope)
        downscale_matrix_device = downscale_matrix.to(acts)
        steered_acts = highdim_acts @ downscale_matrix
        return steered_acts

    @classmethod
    def high_dim_ablate(
        cls,
        acts: Tensor,
        direction: Tensor,
        upscale_matrix: Tensor,
        downscale_matrix: Tensor,
        nonlinearity: str,
        leaky_relu_slope: float,
        prompt_only: bool,
        head_index: Optional[int],
    ):
        def nonlinearity_fn(activation, nonlinearity="leaky_relu", leaky_relu_slope=0.5):
            if nonlinearity == "leaky_relu":
                return torch.nn.functional.leaky_relu(activation, negative_slope=leaky_relu_slope)
            elif nonlinearity == "skip_softplus":
                sp = torch.nn.functional.softplus(activation, threshold=20).to(activation)
                return activation + sp
            elif nonlinearity == "none":
                return activation

        def skip_softplus_inverse(x: torch.Tensor, low_thresh=-10, high_thresh=10) -> torch.Tensor:
            exact = torch.log((torch.sqrt(1 + 4 * torch.exp(x)) - 1) / 2)
            approx = torch.where(x > high_thresh, x / 2, exact)
            approx = torch.where(x < low_thresh, x, approx)
            return approx

        def denonlinearity_fn(activation, nonlinearity="leaky_relu", leaky_relu_slope=0.5):
            if nonlinearity == "leaky_relu":
                activation = activation.clone()
                activation[activation < 0] /= leaky_relu_slope
                return activation
            elif nonlinearity == "skip_softplus":
                return skip_softplus_inverse(activation)
            elif nonlinearity == "none":
                return activation

        if prompt_only:
            batch_size, seq_len = acts.shape[:2]
            if seq_len <= 1:
                return acts

        direction = direction.to(acts)
        direction = direction / (torch.norm(direction, p=2).clamp(min=1e-8))

        upscale_matrix_device = upscale_matrix.to(acts)
        downscale_matrix_device = downscale_matrix.to(acts)

        highdim_acts = acts @ upscale_matrix_device
        highdim_acts = nonlinearity_fn(highdim_acts, nonlinearity, leaky_relu_slope)

        if head_index is not None:
            x = highdim_acts[:, :, head_index, :]
            proj = torch.sum(x * direction, dim=-1, keepdim=True)
            highdim_acts = highdim_acts.clone()
            highdim_acts[:, :, head_index, :] = x - proj * direction
        else:
            proj = torch.sum(highdim_acts * direction, dim=-1, keepdim=True)
            highdim_acts = highdim_acts - proj * direction

        highdim_acts = denonlinearity_fn(highdim_acts, nonlinearity, leaky_relu_slope)
        return highdim_acts @ downscale_matrix_device

    @classmethod
    def rks(
        cls,
        acts: Tensor,
        witness_direction: Tensor,
        steer_direction: Tensor,
        mean_shift: Tensor,
        threshold: float,
        mean_shift_norm: float,
        degree: int,
        alpha: float,
        gain_exponent: int,
        gain_scale: str,
        gain_normalizer: float,
        gain_clip=None,
        field_weight: float = 0.0,
        field_scale: str = "raw",
        mask_update: str = "recompute",
        steps: int = 1,
        step_decay: float = 1.0,
        lambda_reg: float = 0.0,
        prompt_only: bool = False,
    ) -> Tensor:
        """RKS witness steering (mirror of pipeline.utils.rks.apply_rks).

        The mask is a hard comparison, so the projection is taken in fp32 with
        the reduction form -- a bf16 dot or a TF32 matmul carries enough error
        to flip mask bits at realistic activation scales.
        """
        if prompt_only and acts.shape[1] == 1:
            return acts
        if float(alpha) == 0.0:
            return acts

        original_dtype = acts.dtype
        shape = acts.shape
        flat = acts.to(dtype=torch.float32).reshape(-1, shape[-1])
        w = witness_direction.to(torch.float32)
        field = (1.0 - float(field_weight)) * steer_direction.to(
            torch.float32
        ) + float(field_weight) * mean_shift.to(torch.float32)
        field_norm = float(torch.linalg.norm(field))
        if field_norm <= 0.0:
            raise ValueError("RKS steering field collapsed to zero.")

        if gain_scale == "budget":
            field = field / field_norm
            scale = float(alpha) * float(mean_shift_norm)
        else:
            if field_scale == "unit":
                field = field / field_norm
            elif field_scale == "mean_shift":
                field = field / field_norm * float(mean_shift_norm)
            scale = float(alpha)

        origin = flat
        current = flat
        frozen = None
        for step in range(int(steps)):
            projection = (current * w).sum(dim=-1)
            mask = (projection > float(threshold)).to(torch.float32)
            if mask_update == "frozen":
                if frozen is None:
                    frozen = mask
                mask = frozen
            excess = (projection - float(threshold)).clamp_min(0.0)
            exponent = int(gain_exponent)
            powered = (
                torch.ones_like(excess) if exponent == 0 else excess.pow(exponent)
            )
            rho = mask * (float(degree) * powered) / float(gain_normalizer)
            if gain_clip is not None:
                rho = rho.clamp(max=float(gain_clip))
            decay = float(step_decay) ** step
            delta = scale * decay * rho.unsqueeze(-1) * field.unsqueeze(0)
            if lambda_reg:
                delta = delta - scale * decay * float(lambda_reg) * mask.unsqueeze(
                    -1
                ) * (current - origin)
            current = current + delta

        return current.reshape(shape).to(dtype=original_dtype)

    @classmethod
    def kernel(
        cls,
        acts: Tensor,
        source: Tensor,
        target: Tensor,
        kernel_type: str,
        kernel_params: dict,
        eta: float,
        lambda_reg: float,
        step_count: int,
        prompt_only: bool,
        head_index: Optional[int],
    ):
        def rbf_kernel(X: Tensor, Y: Tensor, sigma: float, bandwidth_mixture: bool = False) -> Tensor:
            """RBF (Gaussian) kernel: k(x, y) = exp(-||x-y||^2 / (2*sigma^2))

            Bandwidth is chosen by the median heuristic using the pairwise squared
            distances between X and Y:
                sigma_eff = sqrt(median(||x - y||^2)).
            """
            X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)  # [N, 1]
            Y_norm_sq = (Y ** 2).sum(dim=1, keepdim=True).T  # [1, M]

            X_one = torch.ones((1, Y.shape[0]), device=X.device, dtype=X.dtype)  # [1, M]
            Y_one = torch.ones((X.shape[0], 1), device=Y.device, dtype=Y.dtype)  # [N, 1]

            X_norm_sq = X_norm_sq @ X_one  # [N, M]
            Y_norm_sq = Y_one @ Y_norm_sq  # [N, M]

            distances_sq = X_norm_sq + Y_norm_sq - 2 * (X @ Y.T)

            # # Median heuristic bandwidth: sigma_eff = sqrt(median(||x - y||^2))
            # flat_distances_sq = distances_sq.flatten()
            # positive_mask = flat_distances_sq > 0
            # if positive_mask.any():
            #     median_dist_sq = torch.median(flat_distances_sq[positive_mask])
            #     sigma_eff = torch.sqrt(median_dist_sq)
            # else:
            #     # Fallback: use provided sigma if all distances are zero (degenerate case)
            #     sigma_eff = torch.as_tensor(sigma, device=X.device, dtype=X.dtype)

            scaled_sigmas = [s * sigma for s in [0.1, 0.5, 1.0, 2.0, 5.0]]
            if bandwidth_mixture:
                return sum(torch.exp(-distances_sq / (2 * scaled_sigma ** 2)) for scaled_sigma in scaled_sigmas)
            else:
                return torch.exp(-distances_sq / (2 * sigma ** 2))
            
        def laplacian_kernel(X: Tensor, Y: Tensor, sigma: float) -> Tensor:
            """Laplacian kernel: k(x, y) = exp(-||x-y||_1 / sigma)"""
            distances_l1 = torch.cdist(X, Y, p=1)  # [N, M]
            return torch.exp(-distances_l1 / sigma)

        def laplacian_kernel_gradient(X: Tensor, Y: Tensor, K: Tensor, sigma: float) -> Tensor:
            """Gradient of Laplacian kernel w.r.t. X: ∇_x k(x, y) = -(1/sigma) * k(x, y) * sign(x - y)"""
            X_expanded = X.unsqueeze(1)  # [N, 1, d]
            Y_expanded = Y.unsqueeze(0)  # [1, M, d]
            signs = torch.sign(X_expanded - Y_expanded)  # [N, M, d]
            weighted_signs = -(1.0 / sigma) * K.unsqueeze(-1) * signs  # [N, M, d]
            return weighted_signs.sum(dim=1)  # [N, d]
        
        if prompt_only:
            batch_size, seq_len = acts.shape[:2]
            if seq_len <= 1:
                return acts

        batch_size, seq_len, d_model = acts.shape
        n_A, n_B = source.shape[0], target.shape[0]

        A_local = source.to(acts.device)
        B_local = target.to(acts.device)

        X = acts.view(-1, d_model).to(source.dtype)  # [N, d_model] where N = batch_size * seq_len
        X_0 = X.clone()

        for step in range(step_count):        
            if kernel_type == "rbf":
                sigma = kernel_params.get("sigma", 1.0) # 1.0 by default
                
                # Compute kernel matrices
                K_B = rbf_kernel(X, B_local, sigma, bandwidth_mixture=True)  # [N, n_B]
                K_A = rbf_kernel(X, A_local, sigma, bandwidth_mixture=True)  # [N, n_A]
                
                # Compute kernel-weighted means
                W_B = K_B @ B_local  # [N, d_model]
                W_A = K_A @ A_local  # [N, d_model]
                
                # Compute row sums
                s_B = K_B.sum(dim=1, keepdim=True)  # [N, 1]
                s_A = K_A.sum(dim=1, keepdim=True)  # [N, 1]
                
                # Compute gradient using matrix form
                grad_Pi = (1.0 / (sigma ** 2)) * (
                    (1.0 / n_B) * W_B - (1.0 / n_A) * W_A
                    - ((s_B / n_B - s_A / n_A) * X)
                )

                # normalize gradient for stability
                grad_Pi = torch.nn.functional.normalize(grad_Pi, dim=-1).to(X)

            elif kernel_type == "polynomial":
                degree = kernel_params.get("degree", 2)
                offset = kernel_params.get("offset", 1.0)
                
                # Compute kernel values and gradients
                inner_B = X @ B_local.T + offset  # [N, n_B]
                inner_A = X @ A_local.T + offset  # [N, n_A]
                
                weights_B = degree * (inner_B ** (degree - 1))  # [N, n_B]
                weights_A = degree * (inner_A ** (degree - 1))  # [N, n_A]
                
                grad_Pi = (1.0 / n_B) * (weights_B @ B_local) - (1.0 / n_A) * (weights_A @ A_local)

            elif kernel_type == "laplacian":
                sigma = kernel_params.get("sigma", 1.0)
                
                # Compute kernel matrices
                K_B = laplacian_kernel(X, B_local, sigma)  # [N, n_B]
                K_A = laplacian_kernel(X, A_local, sigma)  # [N, n_A]
                
                # Compute gradients
                grad_B = laplacian_kernel_gradient(X, B_local, K_B, sigma)  # [N, d_model]
                grad_A = laplacian_kernel_gradient(X, A_local, K_A, sigma)  # [N, d_model]
                
                grad_Pi = (1.0 / n_B) * grad_B - (1.0 / n_A) * grad_A
            else:
                raise ValueError(f"Unknown kernel type: {kernel_type}")
            
            # Apply gradient ascent with regularization
            # For the first step, since X = X_0, the regularization term is 0

            X_steered = X + eta / math.sqrt(step + 1.0) * (grad_Pi - lambda_reg * (X - X_0))
            X = X_steered
        
        steered_acts = X.view(batch_size, seq_len, d_model).to(acts.dtype)
        return steered_acts

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            with steer(self.model, self.hook_to_steer):
                return self.model.forward(*args, **kwargs)

    def _model_call(self, *args, **kwargs):
        with steer(self.model, self.hook_to_steer):
            return super()._model_call(*args, **kwargs)

    def _model_generate(self, *args, **kwargs):
        with steer(self.model, self.hook_to_steer):
            return super()._model_generate(*args, **kwargs)
