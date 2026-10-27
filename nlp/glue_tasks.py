"""GLUE task metadata for BERT fine-tuning."""

from __future__ import annotations

# HF `datasets` builder config name -> (sentence1_key, sentence2_key or None)
GLUE_TASK_KEYS: dict[str, tuple[str, str | None]] = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def glue_num_labels(task: str) -> int:
    t = task.lower()
    if t == "mnli":
        return 3
    if t == "stsb":
        return 1
    return 2


def is_glue_regression(task: str) -> bool:
    return task.lower() == "stsb"


def mnli_eval_split_names(task: str) -> list[str]:
    return ["validation_matched", "validation_mismatched"] if task.lower() == "mnli" else ["validation"]
