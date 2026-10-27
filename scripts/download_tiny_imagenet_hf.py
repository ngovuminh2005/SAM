#!/usr/bin/env python3
"""
Download zh-plus/tiny-imagenet and export to ImageFolder layout.

Output layout (default: ~/.datasets/tiny-imagenet):
  train/<class_id>/*.jpg
  val/<class_id>/*.jpg
"""

import argparse
import os
from pathlib import Path


def _ensure_deps() -> None:
    try:
        import datasets  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'datasets'. Install with: pip install datasets"
        ) from exc


def _class_folder(label: int, label_names: list[str] | None) -> str:
    # Prefer canonical Tiny-ImageNet synset ids (e.g., n01443537) when available.
    if label_names is not None and 0 <= label < len(label_names):
        return str(label_names[label])
    return f"class_{label:03d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="zh-plus/tiny-imagenet")
    parser.add_argument("--out", default=str(Path.home() / ".datasets" / "tiny-imagenet"))
    parser.add_argument("--max-train", type=int, default=0, help="debug only; 0 means all")
    parser.add_argument("--max-val", type=int, default=0, help="debug only; 0 means all")
    args = parser.parse_args()

    _ensure_deps()
    from datasets import load_dataset

    out_dir = Path(os.path.expanduser(args.out)).resolve()
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    ds_train = load_dataset(args.repo, split="train")
    ds_val = load_dataset(args.repo, split="valid")

    label_names = None
    label_feature = ds_train.features.get("label")
    if hasattr(label_feature, "names"):
        label_names = list(label_feature.names)

    if len(ds_train) < 100000 or len(ds_val) < 10000:
        raise RuntimeError(f"Unexpected split sizes: train={len(ds_train)}, valid={len(ds_val)}")

    n_train = args.max_train if args.max_train > 0 else len(ds_train)
    n_val = args.max_val if args.max_val > 0 else len(ds_val)

    for i in range(n_train):
        ex = ds_train[i]
        label = int(ex["label"])
        cls_dir = train_dir / _class_folder(label, label_names)
        cls_dir.mkdir(parents=True, exist_ok=True)
        ex["image"].save(cls_dir / f"{i:08d}.jpg", format="JPEG", quality=95)

    for i in range(n_val):
        ex = ds_val[i]
        label = int(ex["label"])
        cls_dir = val_dir / _class_folder(label, label_names)
        cls_dir.mkdir(parents=True, exist_ok=True)
        ex["image"].save(cls_dir / f"{i:08d}.jpg", format="JPEG", quality=95)

    print(f"Done. TinyImageNet exported to: {out_dir}")
    print(f"train={n_train}, val={n_val}")


if __name__ == "__main__":
    main()
