import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Generic base model - no chat template, just raw instruction

BASE_TEMPLATE = "{instruction}"


def format_instruction_base(
    instruction: str,
    output: str = None,
    include_trailing_whitespace: bool = True
):
    formatted_instruction = BASE_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_base(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_base(instruction=instruction, output=output, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_base(instruction=instruction, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result


def orthogonalize_base_weights(model, direction: Float[Tensor, "d_model"]):
    """Generic orthogonalization for transformer models."""
    # Try to get embed_tokens (LLaMA-style) or wte (GPT-style)
    if hasattr(model.model, 'embed_tokens'):
        model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        model.transformer.wte.weight.data = get_orthogonalized_matrix(model.transformer.wte.weight.data, direction)

    # Get layers
    if hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        return

    for block in layers:
        # LLaMA-style
        if hasattr(block, 'self_attn') and hasattr(block.self_attn, 'o_proj'):
            block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'down_proj'):
            block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T
        # GPT-style
        if hasattr(block, 'attn') and hasattr(block.attn, 'c_proj'):
            block.attn.c_proj.weight.data = get_orthogonalized_matrix(block.attn.c_proj.weight.data.T, direction).T
        if hasattr(block, 'mlp') and hasattr(block.mlp, 'c_proj'):
            block.mlp.c_proj.weight.data = get_orthogonalized_matrix(block.mlp.c_proj.weight.data.T, direction).T


def act_add_base_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    """Generic activation addition for transformer models."""
    # Get layers
    if hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Could not find model layers")

    target_layer = layers[layer - 1]

    # Get the down projection layer
    if hasattr(target_layer, 'mlp') and hasattr(target_layer.mlp, 'down_proj'):
        down_proj = target_layer.mlp.down_proj
    elif hasattr(target_layer, 'mlp') and hasattr(target_layer.mlp, 'c_proj'):
        down_proj = target_layer.mlp.c_proj
    else:
        raise ValueError("Could not find down projection layer")

    dtype = down_proj.weight.dtype
    device = down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)
    down_proj.bias = torch.nn.Parameter(bias)


class BaseModel(ModelBase):
    """Generic base model class for any HuggingFace transformer without chat templates."""

    def _load_model(self, model_path, dtype=torch.bfloat16):

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False)

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_base, tokenizer=self.tokenizer, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        # For base models, use EOS token as the end-of-instruction marker
        return [self.tokenizer.eos_token_id]

    def _get_refusal_toks(self):
        # Common refusal tokens - "I" for English models
        return self.tokenizer.encode("I", add_special_tokens=False)[:1]

    def _get_model_block_modules(self):
        # LLaMA-style
        if hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        # GPT-style
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        else:
            raise ValueError("Could not find model layers")

    def _get_attn_modules(self):
        block_modules = self._get_model_block_modules()
        # LLaMA-style
        if hasattr(block_modules[0], 'self_attn'):
            return torch.nn.ModuleList([block.self_attn for block in block_modules])
        # GPT-style
        elif hasattr(block_modules[0], 'attn'):
            return torch.nn.ModuleList([block.attn for block in block_modules])
        else:
            raise ValueError("Could not find attention modules")

    def _get_mlp_modules(self):
        block_modules = self._get_model_block_modules()
        return torch.nn.ModuleList([block.mlp for block in block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_base_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_base_weights, direction=direction, coeff=coeff, layer=layer)