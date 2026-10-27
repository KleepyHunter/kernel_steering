import torch
import os
import pickle

from typing import List, Optional, Literal
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
import numpy as np
from sklearn.decomposition import KernelPCA

from pipeline.utils.hook_utils import add_hooks, get_kernelized_steering_input_pre_hook
from pipeline.model_utils.model_base import ModelBase

def get_mean_activations_pre_hook(layer, cache: Float[Tensor, "pos layer d_model"], n_samples, positions: List[int]):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn

def get_mean_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    fwd_pre_hooks = [(block_modules[layer], get_mean_activations_pre_hook(layer=layer, cache=mean_activations, n_samples=n_samples, positions=positions)) for layer in range(n_layers)]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return mean_activations

def get_mean_diff(model, tokenizer, target_instructions, source_instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], batch_size=32, positions=[-1]):
    mean_activations_target = get_mean_activations(model, tokenizer, target_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)
    mean_activations_source = get_mean_activations(model, tokenizer, source_instructions, tokenize_instructions_fn, block_modules, batch_size=batch_size, positions=positions)

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_target - mean_activations_source

    return mean_diff

# ============ Kernel Functions ============

def rbf_kernel(X: Float[Tensor, "N d"], Y: Float[Tensor, "M d"], sigma: float) -> Float[Tensor, "N M"]:
    """
    RBF (Gaussian) kernel: k(x, y) = exp(-||x-y||^2 / (2*sigma^2))
    
    Uses efficient matrix form to compute pairwise distances.
    """
    # Compute squared norms for X and Y
    X_norm_sq = (X ** 2).sum(dim=1, keepdim=True)  # [N, 1]
    Y_norm_sq = (Y ** 2).sum(dim=1, keepdim=True).T  # [1, M]
    
    # Compute pairwise squared distances: ||x||^2 + ||y||^2 - 2*x*y^T
    distances_sq = X_norm_sq + Y_norm_sq - 2 * (X @ Y.T)
    
    # Compute kernel values
    return torch.exp(-distances_sq / (2 * sigma ** 2))

def rbf_kernel_gradient(X: Float[Tensor, "N d"], Y: Float[Tensor, "M d"], K: Float[Tensor, "N M"], sigma: float) -> Float[Tensor, "N d"]:
    """
    Gradient of RBF kernel w.r.t. X: ∇_x k(x, y) = (1/sigma^2) * k(x, y) * (y - x)
    
    Returns sum over Y: sum_j ∇_x k(x_i, y_j) for each x_i
    """
    # K has shape [N, M], Y has shape [M, d]
    # For each x_i, we compute sum_j k(x_i, y_j) * (y_j - x_i)
    weighted_sum = K @ Y  # [N, d] - weighted sum of Y
    row_sums = K.sum(dim=1, keepdim=True)  # [N, 1] - sum of kernel values for each row
    
    return (1.0 / (sigma ** 2)) * (weighted_sum - row_sums * X)

def polynomial_kernel(X: Float[Tensor, "N d"], Y: Float[Tensor, "M d"], degree: int, offset: float = 1.0) -> Float[Tensor, "N M"]:
    """
    Polynomial kernel: k(x, y) = (x^T * y + c)^p
    """
    return (X @ Y.T + offset) ** degree

def polynomial_kernel_gradient(X: Float[Tensor, "N d"], Y: Float[Tensor, "M d"], degree: int, offset: float = 1.0) -> Float[Tensor, "N d"]:
    """
    Gradient of polynomial kernel w.r.t. X: ∇_x k(x, y) = p * (x^T * y + c)^(p-1) * y
    """
    inner = X @ Y.T + offset  # [N, M]
    weights = degree * (inner ** (degree - 1))  # [N, M]
    return weights @ Y  # [N, d]

def laplacian_kernel(X: Float[Tensor, "N d"], Y: Float[Tensor, "M d"], sigma: float) -> Float[Tensor, "N M"]:
    """
    Laplacian kernel: k(x, y) = exp(-||x-y||_1 / sigma)
    """
    # Compute L1 distances
    cdist_x = X.float() if X.dtype in (torch.float16, torch.bfloat16) else X
    cdist_y = Y.float() if Y.dtype in (torch.float16, torch.bfloat16) else Y
    distances_l1 = torch.cdist(cdist_x, cdist_y, p=1)  # [N, M]
    kernel = torch.exp(-distances_l1 / sigma)
    return kernel.to(X.dtype) if kernel.dtype != X.dtype else kernel

def laplacian_kernel_gradient(X: Float[Tensor, "N d"], Y: Float[Tensor, "M d"], K: Float[Tensor, "N M"], sigma: float) -> Float[Tensor, "N d"]:
    """
    Gradient of Laplacian kernel w.r.t. X: ∇_x k(x, y) = -(1/sigma) * k(x, y) * sign(x - y)
    """
    # Compute sign(x_i - y_j) for all pairs
    # Expand dimensions for broadcasting
    X_expanded = X.unsqueeze(1)  # [N, 1, d]
    Y_expanded = Y.unsqueeze(0)  # [1, M, d]
    signs = torch.sign(X_expanded - Y_expanded)  # [N, M, d]
    
    # Weight by kernel values: -(1/sigma) * k(x, y) * sign(x - y)
    weighted_signs = -(1.0 / sigma) * K.unsqueeze(-1) * signs  # [N, M, d]
    
    # Sum over Y dimension
    return weighted_signs.sum(dim=1)  # [N, d]

# ============ Kernelized Steering ============

def get_all_activations(
    model,
    tokenizer,
    instructions,
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    batch_size=32,
    positions=[-1],
    outputs=None,
    use_output_hook: bool = False,
):
    """
    Collect all activations (not just means) from instructions for kernelized steering.
    
    Returns activations of shape [n_samples, n_positions, n_layers, d_model]
    """
    torch.cuda.empty_cache()
    
    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    
    # Store activations per layer: [layer_idx][batch_idx] -> [batch_size, n_positions, d_model]
    all_layer_activations = [[] for _ in range(n_layers)]
    
    if outputs is not None and len(outputs) != len(instructions):
        raise ValueError(
            "When provided, outputs must be the same length as instructions. "
            f"Got instructions={len(instructions)}, outputs={len(outputs)}."
        )

    def _store_activation(layer_idx, activation):
        hidden_states: Float[Tensor, "batch_size seq_len d_model"] = activation.clone()
        pos_activations = hidden_states[:, positions, :]
        all_layer_activations[layer_idx].append(pos_activations)

    def make_pre_hook_fn(layer_idx):
        def hook_fn(module, input):
            _store_activation(layer_idx, input[0])
        return hook_fn

    def make_output_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            activation = output[0] if isinstance(output, tuple) else output
            _store_activation(layer_idx, activation)
        return hook_fn

    if use_output_hook:
        fwd_pre_hooks = []
        fwd_hooks = [(block_modules[layer], make_output_hook_fn(layer)) for layer in range(n_layers)]
    else:
        fwd_pre_hooks = [(block_modules[layer], make_pre_hook_fn(layer)) for layer in range(n_layers)]
        fwd_hooks = []
    
    # Collect activations for all layers in one pass
    for i in tqdm(range(0, len(instructions), batch_size), desc="Collecting all activations"):
        batch_instructions = instructions[i:i + batch_size]
        if outputs is None:
            inputs = tokenize_instructions_fn(instructions=batch_instructions)
        else:
            batch_outputs = outputs[i:i + batch_size]
            inputs = tokenize_instructions_fn(instructions=batch_instructions, outputs=batch_outputs)
        
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )
    
    # Process each layer
    processed_layers = []
    for layer in range(n_layers):
        # Concatenate all batches: [total_samples, n_positions, d_model]
        layer_tensor = torch.cat(all_layer_activations[layer], dim=0)
        processed_layers.append(layer_tensor)
    
    # Stack: [n_layers, total_samples, n_positions, d_model]
    # Then permute to [total_samples, n_positions, n_layers, d_model]
    stacked = torch.stack(processed_layers, dim=0)  # [n_layers, total_samples, n_positions, d_model]
    stacked = stacked.permute(1, 2, 0, 3)  # [total_samples, n_positions, n_layers, d_model]
    
    return stacked

def get_kernelized_reference_activations(
    model,
    tokenizer,
    target_instructions,
    source_instructions,
    target_outputs,
    source_outputs,
    tokenize_instructions_fn,
    block_modules: List[torch.nn.Module],
    positions: List[int],
    batch_size: int = 32,
    use_output_hook: bool = False,
):
    """
    Collect reference activations for kernelized steering.
    
    Kernelized steering is activation-dependent and applied during inference via hooks.
    This function collects the reference activations (A = source, B = target) that
    will be used to compute gradients during inference.
    
    Returns:
        A_dict: Dictionary mapping (pos_idx, layer) -> source activations [n_A, d_model]
        B_dict: Dictionary mapping (pos_idx, layer) -> target activations [n_B, d_model]
    """
    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    
    # Get all activations for kernel computation
    all_activations_target = get_all_activations(
        model, tokenizer, target_instructions, tokenize_instructions_fn,
        block_modules, batch_size=batch_size, positions=positions, outputs=target_outputs, use_output_hook=use_output_hook
    )
    all_activations_source = get_all_activations(
        model, tokenizer, source_instructions, tokenize_instructions_fn,
        block_modules, batch_size=batch_size, positions=positions, outputs=source_outputs, use_output_hook=use_output_hook
    )
    
    # Store reference activations per position and layer
    A_dict = {}
    B_dict = {}
    
    for pos_idx in range(n_positions):
        for layer in range(n_layers):
            # Extract activations for this position and layer
            # Shape: [n_samples, n_positions, n_layers, d_model]
            A = all_activations_source[:, pos_idx, layer, :]  # [n_A, d_model]
            B = all_activations_target[:, pos_idx, layer, :]    # [n_B, d_model]
            
            A_dict[(pos_idx-n_positions, layer)] = A
            B_dict[(pos_idx-n_positions, layer)] = B
    
    return A_dict, B_dict


def generate_curveball_state(
    model_base: ModelBase,
    target_instructions,
    source_instructions,
    target_outputs,
    source_outputs,
    layer: int,
    positions: int,
    use_output_hook: bool,
    n_components: int = 20,
    kernel: str = "poly",
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0,
    fit_inverse_alpha: float = 1.0,
):
    positions_list = [positions]
    target_all = get_all_activations(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=target_instructions,
        outputs=target_outputs,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        positions=positions_list,
        use_output_hook=use_output_hook,
    )
    source_all = get_all_activations(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=source_instructions,
        outputs=source_outputs,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        block_modules=model_base.model_block_modules,
        positions=positions_list,
        use_output_hook=use_output_hook,
    )

    target_acts = target_all[:, 0, layer, :].to(torch.float32).cpu().numpy()
    source_acts = source_all[:, 0, layer, :].to(torch.float32).cpu().numpy()
    combined = np.concatenate([target_acts, source_acts], axis=0)

    if combined.shape[0] < 2:
        raise ValueError("Curveball requires at least 2 total samples to fit KernelPCA.")

    if target_acts.shape[0] == 0 or source_acts.shape[0] == 0:
        raise ValueError("Curveball requires non-empty target and source activation sets.")

    n_components_eff = min(int(n_components), combined.shape[0] - 1)
    if n_components_eff < 1:
        raise ValueError("curveball_k must be >= 1 after dataset-size adjustment.")

    kpca = KernelPCA(
        n_components=n_components_eff,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        fit_inverse_transform=True,
        alpha=fit_inverse_alpha,
    )
    kpca.fit(combined)
    z_target = kpca.transform(target_acts)
    z_source = kpca.transform(source_acts)

    z_target_mean = z_target.mean(axis=0)
    z_source_mean = z_source.mean(axis=0)
    delta_z = z_target_mean - z_source_mean
    delta_norm = float(np.linalg.norm(delta_z))
    print(
        "Curveball steering vector norm before normalization "
        f"(layer={layer}, pos={positions}): {delta_norm:.6f}"
    )
    if delta_norm <= 1e-12:
        raise ValueError("Curveball direction norm is near zero; cannot normalize.")

    z_hat = delta_z / delta_norm
    return {
        "kpca": kpca,
        "z_hat": z_hat.astype(np.float32),
        "z_target_mean": z_target_mean.astype(np.float32),
        "z_source_mean": z_source_mean.astype(np.float32),
        "n_components": int(n_components_eff),
        "kernel": kernel,
        "degree": int(degree),
        "gamma": (None if gamma is None else float(gamma)),
        "coef0": float(coef0),
        "fit_inverse_alpha": float(fit_inverse_alpha),
        "layer": int(layer),
        "position": int(positions),
    }

# ============ Main Generation Function ============

def generate_directions(
    model_base: ModelBase,
    target_instructions,
    source_instructions,
    artifact_dir,
    method: Literal["dim", "kernel"] = "dim",
    kernel_params: Optional[dict] = None,
    target_outputs=None,
    source_outputs=None,
    use_output_hook: bool = False,
    steer_method: str = "kernel",
    curveball_k: int = 20,
    curveball_kernel: str = "poly",
    curveball_degree: int = 3,
    curveball_gamma: Optional[float] = None,
    curveball_coef0: float = 1.0,
    curveball_fit_inverse_alpha: float = 1.0,
    positions=None,
    layers: Optional[int] = None,
):
    """
    Generate steering data using either difference-in-means (DiM) or kernelized steering.
    
    For DiM: computes fixed steering vectors (directions)
    For kernelized: collects reference activations for activation-dependent steering during inference
    
    Args:
        model_base: ModelBase instance
        target_instructions: List of target instruction strings
        source_instructions: List of source instruction strings
        artifact_dir: Directory to save artifacts
        method: "dim" for difference-in-means, "kernel" for kernelized steering
        kernel_params: Parameters for kernelized steering (will be saved for later use)
    
    Returns:
        For "dim": directions tensor of shape [n_positions, n_layers, d_model]
        For "kernel": tuple (A_dict, B_dict) of reference activations
    """
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    if steer_method == "curveball":
        if layers is None:
            raise ValueError("Curveball mode requires a target layer via 'layers'.")
        if positions is None:
            positions = -1

        curveball_state = generate_curveball_state(
            model_base=model_base,
            target_instructions=target_instructions,
            source_instructions=source_instructions,
            target_outputs=target_outputs,
            source_outputs=source_outputs,
            layer=layers,
            positions=positions,
            use_output_hook=use_output_hook,
            n_components=curveball_k,
            kernel=curveball_kernel,
            degree=curveball_degree,
            gamma=curveball_gamma,
            coef0=curveball_coef0,
            fit_inverse_alpha=curveball_fit_inverse_alpha,
        )
        with open(f"{artifact_dir}/curveball_state.pkl", "wb") as f:
            pickle.dump(curveball_state, f)
        return curveball_state

    positions = list(range(-len(model_base.eoi_toks), 0))
    if use_output_hook and -2 not in positions:
        positions.append(-2)
        positions = sorted(set(positions))
    print(positions)
    
    if method == "dim":
        directions = get_mean_diff(
            model_base.model, model_base.tokenizer,
            target_instructions, source_instructions,
            model_base.tokenize_instructions_fn,
            model_base.model_block_modules,
            batch_size=32,
            positions=positions
        )

        assert directions.shape == (len(model_base.eoi_toks), model_base.model.config.num_hidden_layers, model_base.model.config.hidden_size)
        assert not directions.isnan().any()

        torch.save(directions, f"{artifact_dir}/mean_diffs.pt")
        return directions
        
    elif method == "kernel":
        # For kernelized steering, we collect reference activations
        # The actual steering happens during inference via hooks
        A_dict, B_dict = get_kernelized_reference_activations(
            model_base.model, model_base.tokenizer,
            target_instructions, source_instructions,
            target_outputs, source_outputs,
            model_base.tokenize_instructions_fn,
            model_base.model_block_modules,
            positions=positions,
            use_output_hook=use_output_hook,
        )
        
        # Save reference activations
        '''
        torch.save({
            'A_dict': A_dict,  # source activations
            'B_dict': B_dict,  # target activations
            'positions': positions,
            'kernel_params': kernel_params or {},
        }, f"{artifact_dir}/kernel_reference_activations.pt")
        '''
        return A_dict, B_dict
    else:
        raise ValueError(f"Unknown method: {method}")

# Sequential kernel steering here
def get_layer_activations_mean_act(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module],
                                   source_activation: Float[Tensor, "n_layers n_A d_model"], target_activation: Float[Tensor, "n_layers n_B d_model"], positions, layer,
                                   batch_size: int = 32, kernel_type: str = "rbf", kernel_params: dict = None, eta: float = 0.1, lambda_reg: float = 0, step_count: int = 1, is_source=False):
    
    """Get activations for a specific layer after having applied sequential steering"""
    
    torch.cuda.empty_cache()
    
    # Store activations for the specified layer
    layer_activations = []

    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone()
        # Extract activations at specified positions
        pos_activations = activation[:, positions, :]  # [batch_size, n_positions, d_model]
        layer_activations.append(pos_activations)

    fwd_pre_hooks = []

    if is_source:
        for prev_layer in range(layer):
            fwd_pre_hooks.append((
                block_modules[prev_layer],
                get_kernelized_steering_input_pre_hook(
                    source=source_activation[prev_layer],
                    target=target_activation[prev_layer],
                    kernel_type=kernel_type,
                    kernel_params=kernel_params,
                    eta=eta,
                    lambda_reg=lambda_reg,
                    step_count=step_count
                )
            ))

    # Create hook for get activations of the specified layer
    fwd_pre_hooks.append((block_modules[layer], hook_fn))
    
    # Collect activations - only forward through up to the specified layer
    for i in range(0, len(instructions), batch_size):
        inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )
    
    # Concatenate all batches: [total_samples, n_positions, d_model]
    activations = torch.cat(layer_activations, dim=0)
    activations = activations.mean(dim=1)  # [total_samples, d_model]

    return activations

def generate_activations_mean_act(model_base: ModelBase, source_instructions, target_instructions, artifact_dir,
                                 kernel_type: str = "rbf", kernel_params: dict = None, eta: float = 0.1, lambda_reg: float = 0, step_count: int = 1):
    
    """Get source and target activations of all layer using sequential steering."""

    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_layers = model_base.model.config.num_hidden_layers
    d_model = model_base.model.config.hidden_size
    positions = list(range(-len(model_base.eoi_toks), 0))
    n_A = len(source_instructions)
    n_B = len(target_instructions)

    # Store the activations in high-precision
    source_activations = torch.zeros((n_layers, n_A, d_model), dtype=torch.float64, device=model_base.model.device)
    target_activations = torch.zeros((n_layers, n_B, d_model), dtype=torch.float64, device=model_base.model.device)

    for layer in tqdm(range(n_layers)):
        layer_source_activations = get_layer_activations_mean_act(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            instructions=source_instructions,
            tokenize_instructions_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            source_activation=source_activations,
            target_activation=target_activations,
            positions=positions,
            layer=layer,
            kernel_type=kernel_type,
            kernel_params=kernel_params,
            eta=eta,
            lambda_reg=lambda_reg,
            step_count=step_count,
            is_source=True
        )
        layer_target_activations = get_layer_activations_mean_act(
            model=model_base.model,
            tokenizer=model_base.tokenizer,
            instructions=target_instructions,
            tokenize_instructions_fn=model_base.tokenize_instructions_fn,
            block_modules=model_base.model_block_modules,
            source_activation=source_activations,
            target_activation=target_activations,
            positions=positions,
            layer=layer,
            kernel_type=kernel_type,
            kernel_params=kernel_params,
            eta=eta,
            lambda_reg=lambda_reg,
            step_count=step_count,
            is_source=False
        )
        source_activations[layer] = layer_source_activations
        target_activations[layer] = layer_target_activations

    torch.save(source_activations, f"{artifact_dir}/source_activations.pt")
    torch.save(target_activations, f"{artifact_dir}/target_activations.pt")

    return source_activations, target_activations



        
