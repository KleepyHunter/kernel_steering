import torch
import itertools
import json
import random

from datasets import load_dataset

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase
from pipeline.validation_selection import fingerprint_records

def batch_iterator_chat_completions(dataset_instructions, dataset_outputs, tokenize_instructions_fn, batch_size, eoi_toks):
    it_instructions = iter(dataset_instructions)
    it_outputs = iter(dataset_outputs)
    while True:
        instructions_batch = list(itertools.islice(it_instructions, batch_size))
        outputs_batch = list(itertools.islice(it_outputs, batch_size))
        if not instructions_batch or not outputs_batch:
            break
        inputs = tokenize_instructions_fn(instructions=instructions_batch, outputs=outputs_batch)

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0 # loss should not be computed for last token position

        # also mask out all tokens before the eoi token region
        for b in range(inputs["input_ids"].shape[0]):
            for i in range(inputs["input_ids"].shape[1]):

                if torch.all(inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks):
                    loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
                    break

                # normally the above condition works. but the tokenization instruction tokens in Llama2 is not clean, and so we need this hack
                if eoi_toks.shape[0] == 6 and (inputs["input_ids"][b, i:i+eoi_toks.shape[0]] == eoi_toks).sum().item() >= eoi_toks.shape[0] - 2:
                    loss_mask[b, :i + eoi_toks.shape[0] - 1] = 0
                    break

        yield inputs, loss_mask 

def batch_iterator_custom_completions(completions_file_path: str, tokenize_instructions_fn, batch_size, eoi_toks):
    """Yields batches from the custom completions."""

    custom_completions = json.load(open(completions_file_path, 'r'))

    instructions, completions = [], []

    for i in range(len(custom_completions)):
        instructions.append(custom_completions[i]['prompt'])
        completions.append(custom_completions[i]['response'])

    return batch_iterator_chat_completions(instructions, completions, tokenize_instructions_fn, batch_size, eoi_toks)

def batch_iterator_alpaca(tokenize_instructions_fn, batch_size, eoi_toks):
    """Yields batches from the Alpaca dataset."""

    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=42)

    instructions, completions = [], []

    for i in range(len(dataset)):
        if dataset[i]['input'].strip() == '': # filter for instructions that do not have inputs
            instructions.append(dataset[i]['instruction'])
            completions.append(dataset[i]['output'])

    return batch_iterator_chat_completions(instructions, completions, tokenize_instructions_fn, batch_size, eoi_toks)

def batch_iterator_pile(tokenizer, batch_size, max_length):
    """Yields batches from the Pile dataset."""
    dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True, trust_remote_code=True)

    it_dataset = iter(dataset)
    while True:
        batch = list(itertools.islice(it_dataset, batch_size))
        if not batch:
            break
        inputs = tokenizer([b['text'] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=max_length)

        loss_mask = inputs["attention_mask"].clone()
        loss_mask[:, -1] = 0 # loss should not be computed for last token position

        yield inputs, loss_mask

def _activation_tensor(value):
    return value[0] if isinstance(value, tuple) else value


def _replace_activation(value, activation):
    if isinstance(value, tuple):
        return (activation, *value[1:])
    return activation


def _prompt_masked_pre_hook(hook, prompt_mask):
    def masked_hook(module, inputs):
        steered = hook(module, inputs)
        original_activation = _activation_tensor(inputs)
        steered_activation = _activation_tensor(steered)
        mask = prompt_mask.to(
            device=original_activation.device,
            dtype=original_activation.dtype,
        ).unsqueeze(-1)
        blended = original_activation + mask * (
            steered_activation - original_activation
        )
        return _replace_activation(steered, blended)

    return masked_hook


def _prompt_masked_output_hook(hook, prompt_mask):
    def masked_hook(module, inputs, output):
        steered = hook(module, inputs, output)
        original_activation = _activation_tensor(output)
        steered_activation = _activation_tensor(steered)
        mask = prompt_mask.to(
            device=original_activation.device,
            dtype=original_activation.dtype,
        ).unsqueeze(-1)
        blended = original_activation + mask * (
            steered_activation - original_activation
        )
        return _replace_activation(steered, blended)

    return masked_hook


def compute_loss_over_dataset(
    model,
    tokenizer,
    batch_iterator,
    n_batches=256,
    fwd_pre_hooks=[],
    fwd_hooks=[],
    prompt_only_masking=False,
):
    accumulated_loss = torch.tensor(0, dtype=torch.float64, device=model.device)
    accumulated_n_tokens = torch.tensor(0, dtype=torch.int64, device=model.device)

    batch_idx = 0
    for inputs, loss_mask in batch_iterator:
        if n_batches != -1 and batch_idx >= n_batches:
            break

        inputs = inputs.to(model.device)
        loss_mask = loss_mask.to(model.device)

        input_ids = inputs["input_ids"]
        active_pre_hooks = fwd_pre_hooks
        active_hooks = fwd_hooks
        if prompt_only_masking:
            # loss_mask[j] scores token j+1.  The first nonzero position is
            # therefore still part of the prompt and must be steered.
            scored_before = loss_mask.cumsum(dim=1) - loss_mask
            prompt_mask = (scored_before == 0) & inputs["attention_mask"].bool()
            active_pre_hooks = [
                (module, _prompt_masked_pre_hook(hook, prompt_mask))
                for module, hook in fwd_pre_hooks
            ]
            active_hooks = [
                (module, _prompt_masked_output_hook(hook, prompt_mask))
                for module, hook in fwd_hooks
            ]

        with torch.inference_mode(), add_hooks(
            module_forward_pre_hooks=active_pre_hooks,
            module_forward_hooks=active_hooks,
        ):
            model_outputs = model(**inputs)

        logits = model_outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs_for_labels = log_probs[:, :-1].gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        # add a last column of zeros to log_probs_for_labels to match the shape of loss_mask
        log_probs_for_labels = torch.cat(
            [
                log_probs_for_labels,
                torch.zeros(log_probs_for_labels.shape[0]).unsqueeze(-1).to(log_probs_for_labels)
            ],
            dim=-1
        )

        # apply loss_mask
        log_probs_for_labels = log_probs_for_labels * loss_mask.to(log_probs_for_labels.device)

        accumulated_loss += -log_probs_for_labels.sum()
        accumulated_n_tokens += loss_mask.sum()

        batch_idx += 1
    
    ce_loss = accumulated_loss / accumulated_n_tokens
    perplexity = torch.exp(ce_loss)    

    return ce_loss, perplexity, accumulated_n_tokens

def load_alpaca_validation_examples(
    harmless_validation,
    *,
    n_examples=256,
    seed=42,
):
    """Resolve a fixed harmless-validation subset to Alpaca completions."""
    candidates = []
    for item in harmless_validation:
        instruction = item["instruction"] if isinstance(item, dict) else str(item)
        if instruction.strip():
            candidates.append(instruction)
    candidates = sorted(set(candidates))
    if len(candidates) < n_examples:
        raise ValueError(
            f"Only {len(candidates)} unique harmless validation instructions; "
            f"cannot sample {n_examples}."
        )
    selected = random.Random(seed).sample(candidates, n_examples)

    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    completions = {
        item["instruction"]: item["output"]
        for item in alpaca
        if item["input"].strip() == ""
    }
    missing = [instruction for instruction in selected if instruction not in completions]
    if missing:
        raise ValueError(
            f"{len(missing)} selected harmless instructions have no Alpaca completion."
        )

    examples = [
        {"instruction": instruction, "output": completions[instruction]}
        for instruction in selected
    ]
    metadata = {
        "seed": seed,
        "n_examples": len(examples),
        "source_pool_size": len(candidates),
        "split_fingerprint": fingerprint_records(examples),
    }
    return examples, metadata


def evaluate_alpaca_validation(
    model_base: ModelBase,
    examples,
    *,
    fwd_pre_hooks=None,
    fwd_hooks=None,
    batch_size=16,
    prompt_only=False,
):
    """Teacher-forced perplexity on every item in the locked utility split."""
    fwd_pre_hooks = [] if fwd_pre_hooks is None else fwd_pre_hooks
    fwd_hooks = [] if fwd_hooks is None else fwd_hooks
    iterator = batch_iterator_chat_completions(
        [item["instruction"] for item in examples],
        [item["output"] for item in examples],
        model_base.tokenize_instructions_fn,
        batch_size,
        torch.tensor(model_base.eoi_toks),
    )
    ce_loss, perplexity, n_tokens = compute_loss_over_dataset(
        model_base.model,
        model_base.tokenizer,
        iterator,
        n_batches=-1,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        prompt_only_masking=prompt_only,
    )
    return {
        "ce_loss": ce_loss.item(),
        "perplexity": perplexity.item(),
        "n_tokens": n_tokens.item(),
        "n_examples": len(examples),
        "split_fingerprint": fingerprint_records(examples),
    }


def evaluate_loss(
    model_base: ModelBase,
    fwd_pre_hooks=[],
    fwd_hooks=[],
    batch_size=16,
    n_batches=256,
    max_seq_length=256,
    dataset_labels=["pile", "alpaca", "alpaca_custom_completions"],
    completions_file_path=None
):
    result = {}

    for label in dataset_labels:
        if label == 'pile':
            dataset_iterator = batch_iterator_pile(model_base.tokenizer, batch_size=batch_size, max_length=max_seq_length)
            n = n_batches
        elif label == 'alpaca':
            dataset_iterator = batch_iterator_alpaca(model_base.tokenize_instructions_fn, batch_size=batch_size, eoi_toks=torch.tensor(model_base.eoi_toks))
            n = n_batches
        elif label == 'alpaca_custom_completions':
            assert completions_file_path is not None, "A file path must be passed to load the completions"

            dataset_iterator = batch_iterator_custom_completions(
                completions_file_path=completions_file_path,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                batch_size=batch_size,
                eoi_toks=torch.tensor(model_base.eoi_toks)
            )
            n = -1 # process all completions
        else:
            raise ValueError(f"Unknown dataset label: {label}")

        ce_loss, perplexity, n_tokens = compute_loss_over_dataset(model_base.model, model_base.tokenizer, dataset_iterator, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, n_batches=n)
        print(f"{label.upper()} DATASET:")
        print(f"CE loss: {ce_loss.item()}, Perplexity: {perplexity.item()}, N tokens: {n_tokens.item()}")

        result[label] = {
            "ce_loss": ce_loss.item(),
            "perplexity": perplexity.item(),
            "n_tokens": n_tokens.item()
        }

    return result