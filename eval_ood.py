"""Evaluate finished runs on ImageNet and the four OOD test sets of wise-ft.

ImageNet comes from the run's validation .ffcv file; ImageNetV2, ImageNet-R, ImageNet-A and
ImageNet-Sketch are image folders. All use the run's own validation setting (center crop 224/256,
resize with INTER_AREA, flip TTA, autocast precision). Metrics go to <run>/ood_metrics.json.

    python eval_ood.py logs/<uuid> [logs/<uuid> ...] --data-root ../wise-ft/datasets
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch as ch
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import Squeeze, ToDevice, ToTensor, ToTorchImage
from torchvision import datasets, models

from train_imagenet import BlurPoolConv2d, DEFAULT_CROP_RATIO, IMAGENET_MEAN, IMAGENET_STD, NormalizeImageTorch

# ImageNet indices of the 200 classes, from wise-ft (Hendrycks et al.)
IMAGENET_A_CLASSES = [
    6, 11, 13, 15, 17, 22, 23, 27, 30, 37, 39, 42, 47, 50, 57, 70, 71, 76, 79, 89, 90, 94, 96, 97, 99, 105,
    107, 108, 110, 113, 124, 125, 130, 132, 143, 144, 150, 151, 207, 234, 235, 254, 277, 283, 287, 291, 295,
    298, 301, 306, 307, 308, 309, 310, 311, 313, 314, 315, 317, 319, 323, 324, 326, 327, 330, 334, 335, 336,
    347, 361, 363, 372, 378, 386, 397, 400, 401, 402, 404, 407, 411, 416, 417, 420, 425, 428, 430, 437, 438,
    445, 456, 457, 461, 462, 470, 472, 483, 486, 488, 492, 496, 514, 516, 528, 530, 539, 542, 543, 549, 552,
    557, 561, 562, 569, 572, 573, 575, 579, 589, 606, 607, 609, 614, 626, 627, 640, 641, 642, 643, 658, 668,
    677, 682, 684, 687, 701, 704, 719, 736, 746, 749, 752, 758, 763, 765, 768, 773, 774, 776, 779, 780, 786,
    792, 797, 802, 803, 804, 813, 815, 820, 823, 831, 833, 835, 839, 845, 847, 850, 859, 862, 870, 879, 880,
    888, 890, 897, 900, 907, 913, 924, 932, 933, 934, 937, 943, 945, 947, 951, 954, 956, 957, 959, 971, 972,
    980, 981, 984, 986, 987, 988]
IMAGENET_R_CLASSES = [
    1, 2, 4, 6, 8, 9, 11, 13, 22, 23, 26, 29, 31, 39, 47, 63, 71, 76, 79, 84, 90, 94, 96, 97, 99, 100, 105,
    107, 113, 122, 125, 130, 132, 144, 145, 147, 148, 150, 151, 155, 160, 161, 162, 163, 171, 172, 178, 187,
    195, 199, 203, 207, 208, 219, 231, 232, 234, 235, 242, 245, 247, 250, 251, 254, 259, 260, 263, 265, 267,
    269, 276, 277, 281, 288, 289, 291, 292, 293, 296, 299, 301, 308, 309, 310, 311, 314, 315, 319, 323, 327,
    330, 334, 335, 337, 338, 340, 341, 344, 347, 353, 355, 361, 362, 365, 366, 367, 368, 372, 388, 390, 393,
    397, 401, 407, 413, 414, 425, 428, 430, 435, 437, 441, 447, 448, 457, 462, 463, 469, 470, 471, 472, 476,
    483, 487, 515, 546, 555, 558, 570, 579, 583, 587, 593, 594, 596, 609, 613, 617, 621, 629, 637, 657, 658,
    701, 717, 724, 763, 768, 774, 776, 779, 780, 787, 805, 812, 815, 820, 824, 833, 847, 852, 866, 875, 883,
    889, 895, 907, 928, 931, 932, 933, 934, 936, 937, 943, 945, 947, 948, 949, 951, 953, 954, 957, 963, 965,
    967, 980, 981, 983, 988]

# name: (folder, expected images, classes of the 1000-way output)
OOD_SETS = {
    'ImageNetV2': ('imagenetv2-matched-frequency-format-val', 10000, None),
    'ImageNetR': ('imagenet-r', 30000, IMAGENET_R_CLASSES),
    'ImageNetA': ('imagenet-a', 7500, IMAGENET_A_CLASSES),
    'ImageNetSketch': ('sketch', 50889, None),
}
DEVICE = ch.device('cuda:0')


def load_model(run, params):
    model = getattr(models, params['model.arch'])()

    def apply_blurpool(module):
        for name, child in module.named_children():
            if isinstance(child, ch.nn.Conv2d) and np.max(child.stride) > 1 and child.in_channels >= 16:
                setattr(module, name, BlurPoolConv2d(child))
            else:
                apply_blurpool(child)

    if params['training.use_blurpool']:
        apply_blurpool(model)
    state = ch.load(run / 'final_weights.pt', map_location='cpu', weights_only=True)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in state.items()})
    return model.to(memory_format=ch.channels_last).to(DEVICE).eval()


def ffcv_batches(path, resolution, batch_size, num_workers):
    decoder = CenterCropRGBImageDecoder((resolution, resolution), ratio=DEFAULT_CROP_RATIO)
    image = [decoder, ToTensor(), ToDevice(DEVICE, non_blocking=True), ToTorchImage(),
             NormalizeImageTorch(IMAGENET_MEAN, IMAGENET_STD, np.float16)]
    label = [IntDecoder(), ToTensor(), Squeeze(), ToDevice(DEVICE, non_blocking=True)]
    return Loader(path, batch_size=batch_size, num_workers=num_workers, order=OrderOption.SEQUENTIAL,
                  drop_last=False, pipelines={'image': image, 'label': label})


class CenterCropResize:
    """The same crop and resize as ffcv's CenterCropRGBImageDecoder."""
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, image):
        pixels = np.asarray(image)
        height, width = pixels.shape[:2]
        crop = int(DEFAULT_CROP_RATIO * min(height, width))
        top, left = (height - crop) // 2, (width - crop) // 2
        pixels = cv2.resize(pixels[top:top + crop, left:left + crop], (self.resolution, self.resolution),
                            interpolation=cv2.INTER_AREA)
        return ch.from_numpy(pixels).permute(2, 0, 1)


def folder_batches(name, root, resolution, batch_size, num_workers):
    folder, expected, _ = OOD_SETS[name]
    dataset = datasets.ImageFolder(root / folder, transform=CenterCropResize(resolution))
    if name == 'ImageNetV2':
        # Folder names are ImageNet class indices, not lexicographic indices
        dataset.samples = [(path, int(Path(path).parent.name)) for path, _ in dataset.samples]
    if len(dataset) != expected:
        raise ValueError(f'{name}: found {len(dataset)} images in {root / folder}, expected {expected}')
    loader = ch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    mean = ch.tensor(IMAGENET_MEAN, device=DEVICE, dtype=ch.float32).view(1, 3, 1, 1)
    std = ch.tensor(IMAGENET_STD, device=DEVICE, dtype=ch.float32).view(1, 3, 1, 1)
    for images, target in loader:
        images = images.to(DEVICE, non_blocking=True, memory_format=ch.channels_last)
        yield ((images.float() - mean) / std).half(), target.to(DEVICE, non_blocking=True)


@ch.no_grad()
def evaluate(model, batches, classes, lr_tta, dtype):
    subset = None if classes is None else ch.tensor(classes, device=DEVICE)
    top1, top5, total = 0, 0, 0
    with ch.autocast('cuda', dtype=dtype):
        for images, target in batches:
            output = model(images)
            if lr_tta:
                output += model(ch.flip(images, dims=[3]))
            if subset is not None:
                output = output[:, subset]
            hits = output.topk(5, dim=1).indices.eq(target[:, None])
            top1 += hits[:, 0].sum().item()
            top5 += hits.any(dim=1).sum().item()
            total += target.numel()
    return dict(top1=top1 / total, top5=top5 / total, images=total)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='+', type=Path, help='run folders with params.json and final_weights.pt')
    parser.add_argument('--data-root', type=Path, required=True, help='folder holding the OOD test sets')
    parser.add_argument('--val-dataset', help="ImageNet .ffcv file, if the run's own path is not on this machine")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()
    for folder, _, _ in OOD_SETS.values():
        if not (args.data_root / folder).is_dir():
            parser.error(f'missing test set {args.data_root / folder}')
    cv2.setNumThreads(0)

    names = ['ImageNet', *OOD_SETS]
    rows = []
    for run in args.runs:
        params = json.loads((run / 'params.json').read_text())
        model = load_model(run, params)
        resolution, lr_tta = params['validation.resolution'], params['validation.lr_tta']
        dtype = {'fp16': ch.float16, 'bf16': ch.bfloat16}[params['training.precision']]
        metrics = {}
        for name in names:
            if name == 'ImageNet':
                val_dataset = args.val_dataset or params['data.val_dataset']
                batches = ffcv_batches(val_dataset, resolution, args.batch_size, args.num_workers)
                classes = None
            else:
                batches = folder_batches(name, args.data_root, resolution, args.batch_size, args.num_workers)
                classes = OOD_SETS[name][2]
            metrics[name] = evaluate(model, batches, classes, lr_tta, dtype)
            print(f'{run.name} {name}: top1 {metrics[name]["top1"]:.4f} top5 {metrics[name]["top5"]:.4f}', flush=True)
        (run / 'ood_metrics.json').write_text(json.dumps(dict(
            checkpoint='final_weights.pt', method=params['sam.method'], rho=params['sam.rho'],
            seed=params['training.seed'], resolution=resolution, lr_tta=lr_tta, val_dataset=val_dataset,
            metrics=metrics), indent=2))
        rows.append((run.name[:8], params['sam.method'], params['sam.rho'], metrics))
        del model

    print('\nrun      method       rho    ' + ' '.join(f'{name:>14s}' for name in names) + '   (top-1 %)')
    for short, method, rho, metrics in rows:
        print(f'{short} {method:12s} {rho:<6g} ' + ' '.join(f'{100 * metrics[name]["top1"]:14.2f}' for name in names))


if __name__ == '__main__':
    main()
