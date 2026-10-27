# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer

from act.datasets import get_dataloader, get_dataset
from act.datasets.coco_captions import CocoStylesDataset


@pytest.fixture(scope="session")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="session")
def dummy_data(tokenizer):
    dataset, collator = get_dataset(
        "jigsaw",
        Path("tests/data/"),
        split="train",
        subsets=["toxic", "non-toxic"],
        tokenizer=tokenizer,
    )
    return {"dataset": dataset, "collator": collator}


@pytest.fixture(scope="session")
def toxic_data(tokenizer):
    dataset, collator = get_dataset(
        "jigsaw",
        Path("tests/data/"),
        split="train",
        subsets=["toxic"],
        tokenizer=tokenizer,
    )
    return {"dataset": dataset, "collator": collator}


@pytest.fixture(scope="session")
def coco_concepts(tokenizer):
    dataset, collator = get_dataset(
        "coco-captions-concepts",
        Path("tests/data"),
        split="train",
        subsets=["pink_elephant", "none"],
        tokenizer=None,
    )
    return {"dataset": dataset, "collator": collator}


def test_get_dataset(dummy_data):
    assert (
        dummy_data["dataset"] is not None
    )  # assuming non-empty datasets for simplicity


def test_get_dataloader(dummy_data):
    dataloader = get_dataloader(
        dummy_data["dataset"],
        batch_size=2,
        num_workers=0,
        collate_fn=dummy_data["collator"],
        drop_last=True,
        shuffle=False,
    )

    # check if the dataloader is iterable and returns correct batches
    for i, batch in enumerate(dataloader):
        assert len(batch["input_ids"]) == 2  # assuming a batch size of 2


def test_get_dataloader_balanced(dummy_data, toxic_data):
    dataloader = get_dataloader(
        dummy_data["dataset"],
        batch_size=2,
        num_workers=0,
        collate_fn=dummy_data["collator"],
        drop_last=True,
        shuffle=False,
        balanced=True,
        seed=42,  # A fixed seed for reproducibility
    )
    batch = next(iter(dataloader))
    subsets = batch["subset"]
    assert "toxic" in subsets and "non-toxic" in subsets

    dataloader = get_dataloader(
        toxic_data["dataset"],
        batch_size=2,
        num_workers=0,
        collate_fn=dummy_data["collator"],
        drop_last=True,
        shuffle=False,
        balanced=True,
        seed=42,  # A fixed seed for reproducibility
    )
    batch = next(iter(dataloader))
    subsets = batch["subset"]
    assert "toxic" in subsets and "non-toxic" not in subsets


def test_coco_concepts(coco_concepts):
    dataset = coco_concepts["dataset"]
    assert "pink_elephant" in dataset[0]["subset"]
    assert "pink elephant" in dataset[0]["prompt"]
    assert "pink_elephant" not in dataset[1]["subset"]
    assert "pink elephant" not in dataset[1]["prompt"]
    dataset[len(dataset) - 1]


def test_coco_styles_uses_configured_data_directory(tmp_path):
    captions_dir = tmp_path / "coco_captions_2017"
    captions_dir.mkdir()
    with (captions_dir / "captions_train2017.json").open("w") as outfile:
        json.dump(
            {"annotations": [{"caption": "A city street", "id": 1}]}, outfile
        )
    with (tmp_path / "style_prompts.json").open("w") as outfile:
        json.dump({"cyberpunk": "cyberpunk, neon_lights"}, outfile)

    dataset = CocoStylesDataset(
        captions_dir,
        split="train",
        subsets=["cyberpunk"],
    )

    assert dataset.style_prompts_path == tmp_path / "style_prompts.json"
    assert dataset[0]["prompt"] == "A city street cyberpunk, neon_lights"
