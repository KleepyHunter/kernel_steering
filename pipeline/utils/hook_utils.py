
import math
import torch
import contextlib
import functools
import numpy as np

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
import numpy as np


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_all_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, 'd_model'],
):
    fwd_pre_hooks = [(model_base.model_block_modules[layer], get_direction_ablation_input_pre_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks = [(model_base.model_attn_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks += [(model_base.model_mlp_modules[layer], get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]

    return fwd_pre_hooks, fwd_hooks

def get_directional_patching_input_pre_hook(direction: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 
        activation += coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_activation_addition_input_pre_hook(vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""], prompt_only: bool = False):
    def hook_fn(module, input):
        nonlocal vector, prompt_only

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        batch_size, seq_len, d_model = activation.shape
        vector = vector.to(activation)
        
        # Prompt-only steering: only apply during prefill phase (seq_len > 1)
        # Prefill phase: seq_len > 1 (processing entire prompt)
        # Decoding phase: seq_len == 1 (processing only the new token)
        if prompt_only:
            # Only apply steering during prefill phase
            if seq_len > 1:
                activation += coeff * vector
            else:
                # Decoding phase - skip steering, return unchanged
                if isinstance(input, tuple):
                    return input
                else:
                    return activation
        else:
            # Apply to all positions (both prefill and decoding)
            activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_kernelized_steering_input_pre_hook(
    source: Float[Tensor, "n_A d_model"],
    target: Float[Tensor, "n_B d_model"],
    kernel_type: str = "rbf",
    kernel_params: dict = None,
    eta: float = 0.1,
    lambda_reg: float = 0,
    step_count: int = 1,
    prompt_only: bool = False
):
    """
    Creates a hook that applies kernelized steering during inference.
    
    For each activation x, computes the kernelized steering gradient and applies:
        x' = x + η * [∇Π(x) - λ(x - x_0)]
    
    where ∇Π(x) is computed using the kernel function with reference activations A and B.
    
    Args:
        source: Source dataset activations [n_A, d_model] (e.g., harmless)
        target: Target dataset activations [n_B, d_model] (e.g., harmful)
        kernel_type: Type of kernel ("rbf", "polynomial", "laplacian")
        kernel_params: Kernel parameters (e.g., {"sigma": 1.0} for RBF)
        eta: Step size for gradient ascent
        lambda_reg: Regularization parameter (not used in first step, but kept for consistency)
        step_count: Current step count (for multi-step steering)
        prompt_length: If specified, only apply steering to positions < prompt_length (prompt-only mode)
    """
    if kernel_params is None:
        kernel_params = {"sigma": 48} if kernel_type == "rbf" else {} # sqrt(d_model)
    
    def hook_fn(module, input):
        nonlocal source, target, kernel_type, kernel_params, eta, lambda_reg, prompt_only
        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input
        
        batch_size, seq_len, d_model = activation.shape
        n_A, n_B = source.shape[0], target.shape[0]

        # Prompt-only steering: only apply during prefill phase (seq_len > 1)
        # Prefill phase: seq_len > 1 (processing entire prompt)
        # Decoding phase: seq_len == 1 (processing only the new token)
        if prompt_only:
            # Only apply steering during prefill phase
            if seq_len > 1:
                # This is the prefill phase - apply steering to all prompt tokens
                X = activation.view(-1, d_model)  # [N, d_model] where N = batch_size * seq_len
            else:
                # This is the decoding phase - skip steering entirely
                # Return activation unchanged
                if isinstance(input, tuple):
                    return input
                else:
                    return activation
        else:
            # No prompt_only mode - apply steering to all positions (both prefill and decoding)
            X = activation.view(-1, d_model)  # [N, d_model] where N = batch_size * seq_len

        # Steer in the wider of the activation and reference dtypes (the all-layer
        # path stores references in float64); a no-op when they already match.
        X = X.to(torch.promote_types(X.dtype, torch.promote_types(source.dtype, target.dtype)))
        X_0 = X.clone()  # Store original for regularization

        for step in range(step_count):
            
            # Move reference activations to same device (use local copies)
            A_local = source.to(X.device)
            B_local = target.to(X.device)

            # Compute kernelized steering gradient
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
            grad_Pi = torch.nn.functional.normalize(grad_Pi, dim=-1).to(X)
            X_steered = X + eta / math.sqrt(step + 1.0) * (grad_Pi - lambda_reg * (X - X_0))
            X = X_steered
        
        # Reshape back to original shape
        # Note: If we're in prompt_only mode and decoding phase, we already returned above
        # So at this point, we're either in prefill phase (prompt_only) or all-token mode
        activation_steered = X.view(batch_size, seq_len, d_model).to(activation.dtype)

        if isinstance(input, tuple):
            return (activation_steered, *input[1:])
        else:
            return activation_steered

    return hook_fn

# Kernel functions for kernelized steering
def rbf_kernel(X: Float[Tensor, "N d"], Y: Float[Tensor, "M d"], sigma: float, bandwidth_mixture: bool = False) -> Float[Tensor, "N M"]:
    """RBF (Gaussian) kernel: k(x, y) = exp(-||x-y||^2 / (2*sigma^2))

    Bandwidth is chosen by the median heuristic using the pairwise squared
    distances between X and Y:
        sigma_eff = sqrt(median(||x - y||^2)).
    """
    X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)  # [N, 1]
    Y_norm_sq = (Y ** 2).sum(dim=1, keepdim=True).T  # [1, M]

    X_one = torch.ones((1, Y.shape[0]), device=X.device, dtype=X.dtype)  # [1, M]
    Y_one = torch.ones((X.shape[0], 1), device=X.device, dtype=X.dtype)  # [N, 1]

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
    # [0.1, 0.5, 1.0, 2.0, 5.0]

    scaled_sigmas = [s * sigma for s in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]]
    if bandwidth_mixture:
        return sum(torch.exp(-distances_sq / (2 * scaled_sigma ** 2)) for scaled_sigma in scaled_sigmas)
    else:
        return torch.exp(-distances_sq / (2 * sigma ** 2))

def laplacian_kernel(X: Float[Tensor, "N d"], Y: Float[Tensor, "M d"], sigma: float) -> Float[Tensor, "N M"]:
    """Laplacian kernel: k(x, y) = exp(-||x-y||_1 / sigma)"""
    cdist_x = X.float() if X.dtype in (torch.float16, torch.bfloat16) else X
    cdist_y = Y.float() if Y.dtype in (torch.float16, torch.bfloat16) else Y
    distances_l1 = torch.cdist(cdist_x, cdist_y, p=1)  # [N, M]
    kernel = torch.exp(-distances_l1 / sigma)
    return kernel.to(X.dtype) if kernel.dtype != X.dtype else kernel

def laplacian_kernel_gradient(X: Float[Tensor, "N d"], Y: Float[Tensor, "M d"], K: Float[Tensor, "N M"], sigma: float) -> Float[Tensor, "N d"]:
    """Gradient of Laplacian kernel w.r.t. X: ∇_x k(x, y) = -(1/sigma) * k(x, y) * sign(x - y)"""
    X_expanded = X.unsqueeze(1)  # [N, 1, d]
    Y_expanded = Y.unsqueeze(0)  # [1, M, d]
    signs = torch.sign(X_expanded - Y_expanded)  # [N, M, d]
    weighted_signs = -(1.0 / sigma) * K.unsqueeze(-1) * signs  # [N, M, d]
    return weighted_signs.sum(dim=1)  # [N, d]

def compute_median_heuristic_sigma(source, target):
    """
    Compute sigma using median heuristic on combined source and target activations.
    
    Args:
        source: Source activations [n_A, d_model]
        target: Target activations [n_B, d_model]
    
    Returns:
        dict: Kernel parameters with computed sigma
    """
    A_B_combined = torch.cat([source, target], dim=0)
    cdist_input = A_B_combined.float() if A_B_combined.dtype in (torch.float16, torch.bfloat16) else A_B_combined
    dists_sq = torch.cdist(cdist_input, cdist_input, p=2) ** 2
    
    # Get median of non-zero distances
    positive_dists = dists_sq[dists_sq > 0]
    if positive_dists.numel() > 0:
        sigma_eff = torch.sqrt(torch.median(positive_dists)).item()
    else:
        # Fallback to sqrt(d_model) if all distances are zero
        sigma_eff = float(source.shape[1] ** 0.5)
    
    kernel_params = {"sigma": sigma_eff}
    return kernel_params

def get_kernelized_steering_hooks(
    model_base,
    layer: int,
    A: Float[Tensor, "n_A d_model"],
    B: Float[Tensor, "n_B d_model"],
    kernel_type: str = "rbf",
    kernel_params: dict = None, # store sigma for rbf, degree and offset for polynomial
    eta: float = 0.1,
    lambda_reg: float = 0.01,
    step_count: int = 1,
    prompt_only: bool = False
):
    """
    Create kernelized steering hooks for a specific position and layer.
    
    Args:
        model_base: ModelBase instance
        pos_idx: Position index (0-based, negative for end-of-instruction positions)
        layer: Layer index
        A: Source dataset activations [n_A, d_model] 
        B: Target dataset activations [n_B, d_model] 
        kernel_type: Type of kernel ("rbf", "polynomial", "laplacian")
        kernel_params: Kernel parameters (e.g., {"sigma": 1.0} for RBF)
        eta: Step size for gradient ascent
        lambda_reg: Regularization parameter
        step_count: Number of steering steps to apply
    
    Returns:
        fwd_pre_hooks: List of forward pre-hooks to apply kernelized steering
        fwd_hooks: List of forward hooks (empty for kernelized steering)
    """
    if kernel_type == "rbf":
        # Compute sigma using median heuristic on combined A and B
        if kernel_params is None:
            kernel_params = {}
        if "sigma" not in kernel_params:
            computed_params = compute_median_heuristic_sigma(A, B)
            kernel_params.update(computed_params)
            print(f"Auto-computed sigma via median heuristic: {kernel_params['sigma']:.4f}")

    
    hook_fn = get_kernelized_steering_input_pre_hook(
        source=A,
        target=B,
        kernel_type=kernel_type,
        kernel_params=kernel_params,
        eta=eta,
        lambda_reg=lambda_reg,
        step_count=step_count,
        prompt_only=prompt_only
    )
    
    fwd_pre_hooks = [(model_base.model_block_modules[layer], hook_fn)]
    fwd_hooks = []
    
    return fwd_pre_hooks, fwd_hooks

def get_curveball_kpca_output_hook(
    kpca,
    z_hat,
    alpha: float,
    prompt_only: bool = False,
):
    z_hat_np = np.asarray(z_hat, dtype=np.float32).reshape(1, -1)
    chunk_size = 4096

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation = output

        batch_size, seq_len, d_model = activation.shape
        if prompt_only and seq_len == 1:
            if isinstance(output, tuple):
                return output
            return activation

        activation_fp32 = activation.to(torch.float32)
        X = activation_fp32.reshape(-1, d_model)
        X_np = X.detach().cpu().numpy()

        steered_chunks = []
        for start in range(0, X_np.shape[0], chunk_size):
            end = start + chunk_size
            x_chunk = X_np[start:end]

            a_curr = kpca.transform(x_chunk)
            a_prime = kpca.inverse_transform(a_curr)
            residual = x_chunk - a_prime

            a_target = a_curr + float(alpha) * z_hat_np
            a_target_hat = kpca.inverse_transform(a_target)
            steered_chunk = (a_target_hat + residual).astype(np.float32, copy=False)
            steered_chunks.append(steered_chunk)

        steered_np = np.concatenate(steered_chunks, axis=0)
        steered = torch.from_numpy(steered_np).to(device=activation.device, dtype=activation.dtype)
        activation_out = steered.view(batch_size, seq_len, d_model)

        if isinstance(output, tuple):
            return (activation_out, *output[1:])
        return activation_out

    return hook_fn
