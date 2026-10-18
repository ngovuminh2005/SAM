"""
Official GLUE dev metrics aligned with the usual leaderboard / BERT table columns:

  CoLA: Mcc.          -> matthews_correlation
  SST-2: Acc.         -> accuracy
  MRPC: Acc./F1.      -> accuracy, f1
  STS-B: Pear./Spea.  -> pearson, spearmanr (or spearman_correlation)
  QQP: Acc./F1.       -> accuracy, f1
  MNLI: Acc.          -> accuracy (matched + mismatched splits)
  QNLI, RTE, WNLI: Acc. -> accuracy

Only these quantities are kept in metrics_history / prints; any extra keys from
`evaluate` (e.g. precision/recall on some tasks) are dropped.

Scaling: HuggingFace ``evaluate`` often returns ratios in [-1, 1] or [0, 1] (e.g. Matthews
0.52, accuracy 0.93). GLUE leaderboard / BERT tables use **0–100** (52, 93). Values with
``abs(x) <= 1`` are multiplied by 100 before rounding; larger magnitudes are left as-is
(already on percent scale).
"""

from __future__ import annotations

from typing import Any

import numpy as np

GLUE_TASK_NAMES = frozenset(
    {"cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"}
)

# Decimal places in JSON / logs (user request).
METRIC_DECIMALS = 3


def _round3(x: float) -> float:
    return round(float(x), METRIC_DECIMALS)


def _to_leaderboard_scale(x: float) -> float:
    """HF glue often returns [0,1] or Matthews in [-1,1]; tables use 0-100."""
    xf = float(x)
    if abs(xf) <= 1.000001:
        return xf * 100.0
    return xf


def _is_scalar_number(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return False
    if isinstance(v, (float, np.floating)):
        return True
    if isinstance(v, int):
        return True
    return False


def extract_official_glue_metrics(task: str, split: str, computed: dict[str, Any]) -> dict[str, float]:
    """
    Filter `computed` to official GLUE metrics for `task`, using canonical key names.

    MNLI: per split returns matched_accuracy or mismatched_accuracy only; caller should
    merge dicts across splits and then set mnli_accuracy_mean if both are present.
    """
    t = task.lower().replace("-", "")
    if t == "sts-b":
        t = "stsb"
    if t not in GLUE_TASK_NAMES:
        raise RuntimeError(f"Unknown GLUE task for metrics: {task!r}")

    lower_map = {str(k).lower(): v for k, v in computed.items()}

    def getf(key: str) -> float | None:
        if key not in lower_map:
            return None
        v = lower_map[key]
        if not _is_scalar_number(v):
            return None
        return float(v)

    def scaled(v: float | None) -> float | None:
        if v is None:
            return None
        return _round3(_to_leaderboard_scale(v))

    if t == "cola":
        v = getf("matthews_correlation")
        s = scaled(v)
        return {"mcc": s} if s is not None else {}

    if t == "stsb":
        out: dict[str, float] = {}
        p = scaled(getf("pearson"))
        if p is not None:
            out["pearson"] = p
        s_raw = getf("spearmanr")
        if s_raw is None:
            s_raw = getf("spearman_correlation")
        s = scaled(s_raw)
        if s is not None:
            out["spearman"] = s
        return out

    if t == "mnli":
        v = getf("accuracy")
        if v is None:
            return {}
        s = scaled(v)
        if s is None:
            return {}
        if split == "validation_matched":
            return {"matched_accuracy": s}
        if split == "validation_mismatched":
            return {"mismatched_accuracy": s}
        return {"accuracy": s}

    if t in ("sst2", "qnli", "rte", "wnli"):
        v = getf("accuracy")
        s = scaled(v)
        return {"accuracy": s} if s is not None else {}

    if t in ("mrpc", "qqp"):
        out = {}
        a = scaled(getf("accuracy"))
        f = scaled(getf("f1"))
        if a is not None:
            out["accuracy"] = a
        if f is not None:
            out["f1"] = f
        return out

    return {}


def add_mnli_mean_if_complete(metrics: dict[str, float]) -> None:
    """In-place: set mnli_accuracy_mean when matched and mismatched accuracies exist."""
    if "matched_accuracy" in metrics and "mismatched_accuracy" in metrics:
        metrics["mnli_accuracy_mean"] = _round3(
            0.5 * (metrics["matched_accuracy"] + metrics["mismatched_accuracy"])
        )


def format_official_metrics_line(task: str, m: dict[str, float]) -> str:
    """One-line summary matching common GLUE table columns (task-specific)."""
    t = task.lower().replace("-", "")
    if t == "sts-b":
        t = "stsb"
    if t == "cola":
        return f"Mcc={m.get('mcc', float('nan')):.{METRIC_DECIMALS}f}"
    if t == "sst2" or t in ("qnli", "rte", "wnli"):
        return f"Acc={m.get('accuracy', float('nan')):.{METRIC_DECIMALS}f}"
    if t in ("mrpc", "qqp"):
        return (
            f"Acc={m.get('accuracy', float('nan')):.{METRIC_DECIMALS}f} "
            f"F1={m.get('f1', float('nan')):.{METRIC_DECIMALS}f}"
        )
    if t == "stsb":
        return (
            f"Pearson={m.get('pearson', float('nan')):.{METRIC_DECIMALS}f} "
            f"Spearman={m.get('spearman', float('nan')):.{METRIC_DECIMALS}f}"
        )
    if t == "mnli":
        a, b = m.get("matched_accuracy"), m.get("mismatched_accuracy")
        mean = m.get("mnli_accuracy_mean")
        d = METRIC_DECIMALS
        if a is not None and b is not None:
            if mean is not None:
                return f"Acc matched/mismatched={a:.{d}f}/{b:.{d}f} mean={mean:.{d}f}"
            return f"Acc matched/mismatched={a:.{d}f}/{b:.{d}f}"
        return str(m)
    return str(m)
