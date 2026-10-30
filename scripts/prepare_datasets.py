#!/usr/bin/env python3
"""Prepare project datasets under datasets/ using the Hugging Face mirrors."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets"


def prepare_cifar(name: str, force: bool) -> None:
    from datasets import load_dataset
    repo = {"cifar10": "uoft-cs/cifar10", "cifar100": "uoft-cs/cifar100"}[name]
    out = DATA_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for split, source_split in (("train", "train"), ("test", "test")):
        target = out / f"{split}.npz"
        if target.is_file() and not force:
            print(f"[skip] {target} already exists", flush=True)
            continue
        data = load_dataset(repo, split=source_split)
        label_key = "label" if "label" in data.features else "fine_label"
        count = len(data)
        import numpy as np
        images = np.empty((count, 32, 32, 3), dtype=np.uint8)
        labels = np.empty((count,), dtype=np.int64)
        for index, row in enumerate(data):
            image = row["img"]
            images[index] = np.asarray(image.convert("RGB"), dtype=np.uint8)
            labels[index] = int(row[label_key])
            if (index + 1) % 10000 == 0:
                print(f"[{name}] {split}: {index + 1}/{count}", flush=True)
        temp = target.with_suffix(".npz.tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, images=images, labels=labels)
        temp.replace(target)
        print(f"[done] {target}: {count} images", flush=True)
    data = load_dataset(repo, split="train[:1]")
    label_key = "label" if "label" in data.features else "fine_label"
    classes = data.features[label_key].names
    metadata = out / "metadata.json"
    temporary = metadata.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"dataset": name, "classes": classes}, indent=2), encoding="utf-8")
    temporary.replace(metadata)


def prepare_tiny(force: bool) -> None:
    train = DATA_ROOT / "tiny-imagenet" / "train"
    val = DATA_ROOT / "tiny-imagenet" / "val"
    if train.is_dir() and val.is_dir() and not force:
        print("[skip] Tiny-ImageNet is already exported", flush=True)
        return
    script = ROOT / "scripts" / "download_tiny_imagenet_hf.py"
    import subprocess
    subprocess.run([sys.executable, str(script), "--out", str(DATA_ROOT / "tiny-imagenet")], check=True)


def main() -> None:
    global DATA_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "cifar10", "cifar100", "tiny-imagenet"), default="all")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--force", action="store_true", help="replace existing prepared data")
    args = parser.parse_args()
    DATA_ROOT = args.data_root.expanduser().resolve()
    os.environ.setdefault("HF_HOME", str(DATA_ROOT / ".hf-cache"))
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    selected = ("cifar10", "cifar100", "tiny-imagenet") if args.dataset == "all" else (args.dataset,)
    for name in selected:
        if name == "tiny-imagenet":
            prepare_tiny(args.force)
        else:
            prepare_cifar(name, args.force)
    print(f"Datasets prepared under {DATA_ROOT}")


if __name__ == "__main__":
    main()
