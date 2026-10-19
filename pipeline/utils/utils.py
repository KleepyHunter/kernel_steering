import torch
import einops
from torch import Tensor
from jaxtyping import Int, Float

def get_orthogonalized_matrix(matrix: Float[Tensor, '... d_model'], vec: Float[Tensor, 'd_model']) -> Float[Tensor, '... d_model']:
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)

    proj = einops.einsum(matrix, vec.unsqueeze(-1), '... d_model, d_model single -> ... single') * vec
    return matrix - proj

def compute_median_heuristic_sigma(source: Float[Tensor, 'N d'], target: Float[Tensor, 'M d']) -> dict:
    # Compute sigma using median heuristic on combined A and B
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
    
    # Initialize kernel_params
    kernel_params = {}
    kernel_params["sigma"] = sigma_eff
    return kernel_params
