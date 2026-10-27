# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from transformers import BatchEncoding

from act.evaluations import evaluate_0shot

SYSTEM_PROMPT = [
    "You are a chatbot that tells if a sentence is about fantasy.",
    "You are a chatbot that tells if a sentence is about football.",
]
ANSWERS = ["no", "yes"]
DATA = {
    "sentence": [
        "Hello, my name is John and I play a sport with a ball.",
        "Unicorns are horses with a horn and some magic.",
    ],
    "prompt": [
        "Can you introduce yourself?",
        "Dragons are like big lizards and breath fire.",
    ],
    "sentence2": [
        "Messi is the best player.",
        "I once saw a strange insect.",
    ],
}

SYSTEM_PROMPT_MC = [
    "Given the following 2 sentences, select the one that talks more about fantasy.",
    "Given the following 2 sentences, select the one that talks more about football.",
]
ANSWERS_MC = ["A", "B"]
DATA_MC = {
    "sentence": [
        "Hello, my name is John and I play a sport with a ball.",
        "Unicorns are horses with a horn and some magic.",
    ],
    "prompt": [
        "Can you introduce yourself?",
        "Dragons are like big lizards and breath fire.",
    ],
    "sentence2": [
        "Messi is the best player.",
        "I once saw a strange insect.",
    ],
}


class _ChatTokenizer:
    def __init__(self, encoded):
        self.encoded = encoded
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return self.encoded


def test_prepare_chat_inputs_accepts_batch_encoding():
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.tensor([[1, 1, 0]])
    tokenizer = _ChatTokenizer(
        BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})
    )

    model_inputs = evaluate_0shot.prepare_chat_inputs(
        tokenizer=tokenizer,
        messages=[],
        device="cpu",
        max_length=128,
    )

    assert torch.equal(model_inputs["input_ids"], input_ids)
    assert torch.equal(model_inputs["attention_mask"], attention_mask)
    assert tokenizer.kwargs["truncation"] is True
    assert tokenizer.kwargs["max_length"] == 128


def test_prepare_chat_inputs_accepts_tensor_and_adds_attention_mask():
    input_ids = torch.tensor([[4, 5]])
    tokenizer = _ChatTokenizer(input_ids)

    model_inputs = evaluate_0shot.prepare_chat_inputs(
        tokenizer=tokenizer,
        messages=[],
        device="cpu",
    )

    assert torch.equal(model_inputs["input_ids"], input_ids)
    assert torch.equal(model_inputs["attention_mask"], torch.ones_like(input_ids))


def test_get_single_token_id_omits_special_tokens():
    class Tokenizer:
        def encode(self, answer, add_special_tokens):
            assert answer == "Yes"
            assert add_special_tokens is False
            return [42]

    assert evaluate_0shot.get_single_token_id(Tokenizer(), "Yes") == 42


def test_get_single_token_id_rejects_multiple_tokens():
    class Tokenizer:
        def encode(self, answer, add_special_tokens):
            return [1, 2]

    with pytest.raises(ValueError, match="exactly one token"):
        evaluate_0shot.get_single_token_id(Tokenizer(), "not a single token")


@pytest.mark.skip(reason="Uses Llama-3-8B-instruct, too large.")
@pytest.mark.parametrize(
    "system_prompt,answers,data,use_second_csv,prepend_answers",
    [
        (SYSTEM_PROMPT, ANSWERS, DATA, False, 0),
        (SYSTEM_PROMPT_MC, ANSWERS_MC, DATA_MC, False, 0),
        (SYSTEM_PROMPT_MC, ANSWERS_MC, DATA_MC, True, 0),
        (SYSTEM_PROMPT_MC, ANSWERS_MC, DATA_MC, False, 1),
        (SYSTEM_PROMPT_MC, ANSWERS_MC, DATA_MC, True, 1),
    ],
)
def test_0shot_e2e(system_prompt, answers, data, use_second_csv, prepend_answers):
    with tempfile.TemporaryDirectory(dir="/tmp/") as tempfolder:
        csv_file = Path(tempfolder) / "test.csv"
        out_file = Path(tempfolder) / "out.csv"
        df = pd.DataFrame(data=data)
        df.to_csv(csv_file)

        second_csv_argv = []
        if use_second_csv:
            csv_file2 = Path(tempfolder) / "test.csv"
            df.to_csv(csv_file2)
            second_csv_argv = [
                "--data-path2",
                str(csv_file2),
            ]

        parser = evaluate_0shot.get_parser()
        args = parser.parse_args(
            [
                "--device",
                "cpu",
                "--system-prompt",
                *system_prompt,
                "--system-answers",
                *answers,
                "--prepend-answers",
                str(prepend_answers),
                "--col-sentence1",
                "sentence",
                "--data-path",
                str(csv_file),
                "--output-file",
                str(out_file),
                *second_csv_argv,
            ]
        )
        evaluate_0shot.main(args)
        df_out = pd.read_csv(out_file)

        assert len(df_out) == 2
        assert "q0_llm_answer" in df_out.columns
        assert "q1_llm_answer" in df_out.columns
        assert df_out["q0_llm_answer"].values[1] in ["yes", "A"]
