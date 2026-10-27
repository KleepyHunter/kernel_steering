import os
import json
from typing import Any, Dict, List, Optional, Tuple

from pipeline.validation_selection import partition_caa_generation_records

dataset_dir_path = os.path.dirname(os.path.realpath(__file__))

SPLITS = ['train', 'val', 'test']
HARMTYPES = ['harmless', 'harmful']

SPLIT_DATASET_FILENAME = os.path.join(dataset_dir_path, 'splits/{harmtype}_{split}.json')

PROCESSED_DATASET_NAMES = ["advbench", "tdc2023", "maliciousinstruct", "harmbench_val", "harmbench_test", "jailbreakbench", "strongreject", "alpaca"]
CAA_SPLITS = ["generate", "test"]
CAA_EVAL_TYPES = ["ab", "open_ended", "truthful_qa", "mmlu"]

def load_dataset_split(harmtype: str, split: str, instructions_only: bool=False):
    assert harmtype in HARMTYPES
    assert split in SPLITS

    file_path = SPLIT_DATASET_FILENAME.format(harmtype=harmtype, split=split)

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]

    return dataset

def load_dataset(dataset_name, instructions_only: bool=False):
    assert dataset_name in PROCESSED_DATASET_NAMES, f"Valid datasets: {PROCESSED_DATASET_NAMES}"

    file_path = os.path.join(dataset_dir_path, 'processed', f"{dataset_name}.json")

    with open(file_path, 'r') as f:
        dataset = json.load(f)

    if instructions_only:
        dataset = [d['instruction'] for d in dataset]
 
    return dataset

def _resolve_caa_file_path(dataset_root: str, split: str, behavior: str) -> str:
    split_dir = os.path.join(dataset_dir_path, dataset_root, split)

    candidates = [
        os.path.join(split_dir, f"{behavior}.json"),
        os.path.join(split_dir, behavior, "generate_dataset.json"),
        os.path.join(split_dir, behavior, "test_dataset_ab.json"),
        os.path.join(split_dir, behavior, "test_dataset_open_ended.json"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    available = []
    if os.path.exists(split_dir):
        for entry in sorted(os.listdir(split_dir)):
            entry_path = os.path.join(split_dir, entry)
            if os.path.isdir(entry_path):
                available.append(entry)
            elif entry.endswith(".json"):
                available.append(entry[:-5])

    raise FileNotFoundError(
        f"CAA dataset file not found for split='{split}', behavior='{behavior}'. "
        f"Looked under: {split_dir}. "
        f"Available behaviors/files: {available if available else 'none'}. "
        "Run scripts/download_caa_datasets.py first or set --caa_dataset_root."
    )


def load_caa_pairs(
    behavior: str,
    split: str = "generate",
    dataset_root: str = "caa",
):
    assert split in CAA_SPLITS, f"Valid splits: {CAA_SPLITS}"

    file_path = _resolve_caa_file_path(dataset_root=dataset_root, split=split, behavior=behavior)

    with open(file_path, "r") as f:
        data = json.load(f)

    required_fields = {"question", "answer_matching_behavior", "answer_not_matching_behavior"}
    pairs = []
    for i, item in enumerate(data):
        missing = required_fields - set(item.keys())
        if missing:
            raise ValueError(f"CAA example {i} in {file_path} is missing fields: {sorted(missing)}")
        pairs.append(
            {
                "instruction": item["question"],
                "positive_output": item["answer_matching_behavior"],
                "negative_output": item["answer_not_matching_behavior"],
            }
        )

    return pairs


def _load_json_file(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _validate_fields(item: Dict, required_fields: set[str], index: int, file_path: str):
    missing = required_fields - set(item.keys())
    if missing:
        raise ValueError(
            f"Example {index} in {file_path} is missing fields: {sorted(missing)}"
        )


def _resolve_caa_test_file_path(
    eval_type: str,
    dataset_root: str = "caa",
    behavior: Optional[str] = None,
) -> str:
    test_dir = os.path.join(dataset_dir_path, dataset_root, "test")

    if eval_type in {"ab", "open_ended"}:
        if behavior is None:
            raise ValueError(f"CAA eval type '{eval_type}' requires a behavior name.")
        filename = "test_dataset_ab.json" if eval_type == "ab" else "test_dataset_open_ended.json"
        path = os.path.join(test_dir, behavior, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"CAA test dataset not found for eval_type='{eval_type}', behavior='{behavior}': {path}"
            )
        return path

    if eval_type == "truthful_qa":
        path = os.path.join(test_dir, "truthfulqa", "truthful_qa.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CAA TruthfulQA dataset not found: {path}")
        return path

    if eval_type == "mmlu":
        path = os.path.join(test_dir, "mmlu", "mmlu.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CAA MMLU dataset not found: {path}")
        return path

    raise ValueError(f"Unsupported CAA eval type: {eval_type}. Valid types: {CAA_EVAL_TYPES}")


def _normalize_caa_ab(item: Dict, behavior: str) -> Dict:
    return {
        "prompt": item["question"],
        "choice_a": "(A)",
        "choice_b": "(B)",
        "answer_matching_behavior": item["answer_matching_behavior"],
        "answer_not_matching_behavior": item["answer_not_matching_behavior"],
        "category": behavior,
    }


def _normalize_caa_open_ended(item: Dict, behavior: str) -> Dict:
    return {
        "prompt": item["question"],
        "category": behavior,
    }


def _normalize_caa_truthfulqa_or_mmlu(item: Dict) -> Dict:
    return {
        "prompt": item["prompt"],
        "choice_a": "(A)",
        "choice_b": "(B)",
        "correct": item["correct"],
        "incorrect": item["incorrect"],
        "category": item["category"],
    }


def load_caa_test_dataset(
    eval_type: str,
    dataset_root: str = "caa",
    behavior: Optional[str] = None,
) -> List[Dict]:
    assert eval_type in CAA_EVAL_TYPES, f"Valid CAA eval types: {CAA_EVAL_TYPES}"

    file_path = _resolve_caa_test_file_path(
        eval_type=eval_type,
        dataset_root=dataset_root,
        behavior=behavior,
    )
    raw_data = _load_json_file(file_path)

    normalized: List[Dict] = []
    if eval_type == "ab":
        required = {"question", "answer_matching_behavior", "answer_not_matching_behavior"}
        for i, item in enumerate(raw_data):
            _validate_fields(item, required, i, file_path)
            normalized.append(_normalize_caa_ab(item, behavior=behavior))
        return normalized

    if eval_type == "open_ended":
        required = {"question"}
        for i, item in enumerate(raw_data):
            _validate_fields(item, required, i, file_path)
            normalized.append(_normalize_caa_open_ended(item, behavior=behavior))
        return normalized

    required = {"prompt", "correct", "incorrect", "category"}
    for i, item in enumerate(raw_data):
        _validate_fields(item, required, i, file_path)
        normalized.append(_normalize_caa_truthfulqa_or_mmlu(item))
    return normalized


def load_caa_reference_validation(
    behavior: str,
    dataset_root: str = "caa",
    validation_size: int = 50,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]], Dict[str, Any]]:
    """Create disjoint CAA reference and validation splits for selection."""
    generation_path = _resolve_caa_file_path(
        dataset_root=dataset_root,
        split="generate",
        behavior=behavior,
    )
    test_path = _resolve_caa_test_file_path(
        eval_type="ab",
        dataset_root=dataset_root,
        behavior=behavior,
    )
    generation_records = _load_json_file(generation_path)
    test_records = _load_json_file(test_path)

    reference_raw, validation_raw, metadata = partition_caa_generation_records(
        generation_records=generation_records,
        test_records=test_records,
        behavior=behavior,
        validation_size=validation_size,
        seed=seed,
    )

    required = {"question", "answer_matching_behavior", "answer_not_matching_behavior"}
    reference: List[Dict[str, str]] = []
    validation: List[Dict[str, Any]] = []
    for i, item in enumerate(reference_raw):
        _validate_fields(item, required, i, generation_path)
        reference.append(
            {
                "instruction": item["question"],
                "positive_output": item["answer_matching_behavior"],
                "negative_output": item["answer_not_matching_behavior"],
            }
        )
    for i, item in enumerate(validation_raw):
        _validate_fields(item, required, i, generation_path)
        validation.append(_normalize_caa_ab(item, behavior=behavior))

    metadata.update(
        {
            "behavior": behavior,
            "generation_path": os.path.abspath(generation_path),
            "test_path": os.path.abspath(test_path),
        }
    )
    return reference, validation, metadata
