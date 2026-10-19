from abc import ABC, abstractmethod
from typing import List
import math
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.hook_utils import add_hooks

class ModelBase(ABC):
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.model: AutoModelForCausalLM = self._load_model(model_name_or_path)
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)
        
        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.eoi_toks = self._get_eoi_toks()
        self.refusal_toks = self._get_refusal_toks()

        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()

    def del_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        pass

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        pass

    @abstractmethod
    def _get_refusal_toks(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass

    @abstractmethod
    def _get_attn_modules(self):
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        pass

    @abstractmethod
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        pass

    @abstractmethod
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        pass

    def generate_completions(self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32, max_new_tokens=64):
        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] for x in dataset]

        for i in tqdm(range(0, len(dataset), batch_size)):
            tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i + batch_size])

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(self.model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(self.model.device),
                    generation_config=generation_config,
                )

                generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

                for generation_idx, generation in enumerate(generation_toks):
                    response = self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                    completions.append({
                        'category': categories[i + generation_idx],
                        'prompt': instructions[i + generation_idx],
                        'response': response
                    })

        return completions

    def _extract_ab_label(self, output: str) -> str:
        label = self._extract_choice_label(output)
        if label in {"A", "B"}:
            return label
        raise ValueError(f"Output does not encode A/B label: {output}")

    def _extract_choice_label(self, output: str) -> str:
        normalized = output.strip().upper()

        exact = re.fullmatch(r"\(?\s*([A-Z])\s*\)?", normalized)
        if exact is not None:
            return exact.group(1)

        bracketed = re.search(r"\(([A-Z])\)", normalized)
        if bracketed is not None:
            return bracketed.group(1)

        raise ValueError(f"Output does not encode a choice label: {output}")

    def _get_choice_token_id(self, label: str) -> int:
        normalized = label.strip().upper()
        if not re.fullmatch(r"[A-Z]", normalized):
            raise ValueError(f"Choice label must be a single letter A-Z. Got: {label}")

        token_id = self.tokenizer.convert_tokens_to_ids(normalized)
        if token_id is None or token_id == self.tokenizer.unk_token_id:
            enc = self.tokenizer.encode(normalized, add_special_tokens=False)
            if not enc:
                raise ValueError(f"Could not resolve token id for '{normalized}'.")
            token_id = enc[0]

        return int(token_id)

    def _get_ab_token_ids(self):
        return self._get_choice_token_id("A"), self._get_choice_token_id("B")

    def _ab_next_token_probs(
        self,
        instruction: str,
        fwd_pre_hooks=[],
        fwd_hooks=[],
    ):
        """
        CAA-style single-step probabilities p(A), p(B) from next-token logits
        after appending '(' to the prompt.
        """
        full_tokens = self.tokenize_instructions_fn(instructions=[instruction], outputs=["("])
        input_ids = full_tokens.input_ids.to(self.model.device)
        attention_mask = full_tokens.attention_mask.to(self.model.device)

        with torch.no_grad():
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        seq_len = int(attention_mask[0].sum().item())
        last_token_logits = outputs.logits[0, seq_len - 1, :].to(torch.float32)
        last_token_probs = torch.softmax(last_token_logits, dim=-1)

        a_id, b_id = self._get_ab_token_ids()
        return float(last_token_probs[a_id].item()), float(last_token_probs[b_id].item())

    def _sequence_logprob_for_output(
        self,
        instruction: str,
        output: str,
        fwd_pre_hooks=[],
        fwd_hooks=[],
    ) -> float:
        """
        Compatibility shim: returns single-step log P(label | instruction + '('),
        where label is A/B extracted from output.
        """
        label = self._extract_ab_label(output)
        a_prob, b_prob = self._ab_next_token_probs(
            instruction=instruction,
            fwd_pre_hooks=fwd_pre_hooks,
            fwd_hooks=fwd_hooks,
        )
        p = a_prob if label == "A" else b_prob
        return math.log(max(p, 1e-45))

    def score_target_labels(
        self,
        instructions: List[str],
        target_labels: List[str],
        fwd_pre_hooks=[],
        fwd_hooks=[],
    ):
        """
        Score p(target_label | instruction + '(') for arbitrary single-letter labels.
        Returns one record per input with target probability and logprob.
        """
        if len(instructions) != len(target_labels):
            raise ValueError("instructions and target_labels must have equal length.")

        results = []
        for instruction, target_label in tqdm(
            list(zip(instructions, target_labels)),
            desc="Scoring target labels",
        ):
            label = self._extract_choice_label(target_label)
            target_token_id = self._get_choice_token_id(label)

            full_tokens = self.tokenize_instructions_fn(instructions=[instruction], outputs=["("])
            input_ids = full_tokens.input_ids.to(self.model.device)
            attention_mask = full_tokens.attention_mask.to(self.model.device)

            with torch.no_grad():
                with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            seq_len = int(attention_mask[0].sum().item())
            last_token_logits = outputs.logits[0, seq_len - 1, :].to(torch.float32)
            last_token_probs = torch.softmax(last_token_logits, dim=-1)

            target_probability = float(last_token_probs[target_token_id].item())
            target_logprob = math.log(max(target_probability, 1e-45))
            results.append(
                {
                    "target_label": label,
                    "target_probability": target_probability,
                    "target_logprob": target_logprob,
                }
            )

        return results

    def score_choice_pairs(
        self,
        instructions: List[str],
        choice_as: List[str],
        choice_bs: List[str],
        fwd_pre_hooks=[],
        fwd_hooks=[],
    ):
        """
        Score binary A/B choices with CAA-style single-step next-token probs.
        Returns one record per input containing `a_prob` and `b_prob`.
        """
        if not (len(instructions) == len(choice_as) == len(choice_bs)):
            raise ValueError("instructions, choice_as and choice_bs must have equal length.")

        results = []
        for instruction, choice_a, choice_b in tqdm(
            list(zip(instructions, choice_as, choice_bs)),
            desc="Scoring A/B choices",
        ):
            label_a = self._extract_ab_label(choice_a)
            label_b = self._extract_ab_label(choice_b)
            p_a_token, p_b_token = self._ab_next_token_probs(
                instruction=instruction,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
            )
            a_prob = p_a_token if label_a == "A" else p_b_token
            b_prob = p_a_token if label_b == "A" else p_b_token
            a_logprob = math.log(max(a_prob, 1e-45))
            b_logprob = math.log(max(b_prob, 1e-45))

            results.append(
                {
                    "a_logprob": a_logprob,
                    "b_logprob": b_logprob,
                    "a_prob": a_prob,
                    "b_prob": b_prob,
                }
            )

        return results
