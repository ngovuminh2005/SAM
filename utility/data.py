import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import random

from .augmentation import Cutout
from utility.args import Args


Args.add_argument("--dataThreads", type=int, help="Number of CPU threads for dataloaders.")
Args.add_argument("--dataDir", type=str, help="main directory to store datasets")
Args.add_argument("--batchSize", type=int, help="batch size")
Args.add_argument("--dataset", type=str, help="Dataset")
Args.add_argument("--imageSize", type=int, help="resize images after augmentation. 0 for no resizing")
Args.add_argument("--flip", type=bool, help="flip horizontally")
Args.add_argument("--crop", type=bool, help="crop 32x32 padding 4")
Args.add_argument("--cut", type=bool, help="cutout")
Args.add_argument("--cutoutProp", type=float, help="Probability for cutout augmenation.")
Args.add_argument("--randAugment", type=bool, help="")
Args.add_argument("--randAugment_magnitude", type=int, help="")
Args.add_argument("--mixup", type=bool, help="")
Args.add_argument("--mixupProp", type=float, help="")
Args.add_argument("--normalize", type=str, help="std or min_max")

def _resolve_imagenet_root(data_dir: str) -> str:
    """
    Return the directory whose *immediate* children are train/ and val/ (ImageFolder layout).

    Accepts either:
      <dataDir>/ImageNet/train|val  (legacy layout in this repo)
      <dataDir>/train|val           (common on clusters: data_dir is ILSVRC root)
    """
    nested = os.path.join(data_dir, "ImageNet")
    t1, v1 = os.path.join(nested, "train"), os.path.join(nested, "val")
    if os.path.isdir(t1) and os.path.isdir(v1):
        return nested
    t2, v2 = os.path.join(data_dir, "train"), os.path.join(data_dir, "val")
    if os.path.isdir(t2) and os.path.isdir(v2):
        return data_dir
    raise RuntimeError(
        "ImageNet train/val not found.\n"
        f"  Tried: {t1}/ and {v1}/\n"
        f"  Tried: {t2}/ and {v2}/\n"
        "Set dataDir (or --dataDir) to either:\n"
        "  - the parent folder of ImageNet/ (so .../ImageNet/train exists), or\n"
        "  - the ILSVRC root that directly contains train/ and val/.\n"
        "Override without editing .ini: export SAM_DATASET_ROOT=/path/to/that/parent"
    )


def _resolve_tiny_imagenet_root(data_dir: str) -> str:
    """
    Return TinyImageNet root with immediate train/ and val/ subfolders.

    Accepts either:
      <dataDir>/TinyImageNet/train|val
      <dataDir>/tiny-imagenet/train|val
      <dataDir>/train|val
    """
    candidates = [
        os.path.join(data_dir, "TinyImageNet"),
        os.path.join(data_dir, "tiny-imagenet"),
        data_dir,
    ]
    for root in candidates:
        if os.path.isdir(os.path.join(root, "train")) and os.path.isdir(os.path.join(root, "val")):
            return root
    raise RuntimeError(
        "TinyImageNet train/val not found.\n"
        f"  Tried: {os.path.join(data_dir, 'TinyImageNet', 'train')}/ and {os.path.join(data_dir, 'TinyImageNet', 'val')}/\n"
        f"  Tried: {os.path.join(data_dir, 'tiny-imagenet', 'train')}/ and {os.path.join(data_dir, 'tiny-imagenet', 'val')}/\n"
        f"  Tried: {os.path.join(data_dir, 'train')}/ and {os.path.join(data_dir, 'val')}/\n"
        "Set dataDir (or --dataDir) to a parent folder containing TinyImageNet."
    )


class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, root, train = True, download = None, transform = []):
        self.train = train
        base = _resolve_imagenet_root(root)
        if train:
            dir = os.path.join(base, "train")
            transform.transforms = [
                transforms.RandomResizedCrop(Args.imageSize if Args.imageSize is not None else 224, scale=(0.08, 1.0), ratio=(3. / 4, 4. / 3.))
                ] + transform.transforms
        else:
            dir = os.path.join(base, "val")
            transform.transforms = [
                CenterCrop(),
                ] + transform.transforms

        super(ImageNet, self).__init__(dir, transform = transform)


class TinyImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, root, train = True, download = None, transform = []):
        del download
        base = _resolve_tiny_imagenet_root(root)
        split_dir = "train" if train else "val"
        dir = os.path.join(base, split_dir)
        super(TinyImageNet, self).__init__(dir, transform = transform)


class _PreparedCIFAR(torch.utils.data.Dataset):
    """CIFAR archive exported from the Hugging Face parquet mirror."""
    folder = None

    def __init__(self, root, train=True, download=None, transform=None, target_transform=None):
        del download
        split = "train" if train else "test"
        base = os.path.join(root, self.folder)
        archive = os.path.join(base, f"{split}.npz")
        metadata = os.path.join(base, "metadata.json")
        if not os.path.isfile(archive) or not os.path.isfile(metadata):
            raise RuntimeError(
                f"Prepared CIFAR data not found at {base}. Run "
                "python scripts/prepare_datasets.py --dataset " + self.folder
            )
        with np.load(archive) as values:
            self.data = values["images"]
            self.targets = values["labels"].astype(np.int64).tolist()
        import json
        with open(metadata, "r", encoding="utf-8") as stream:
            self.classes = json.load(stream)["classes"]
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = transforms.functional.to_pil_image(self.data[index])
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class PreparedCIFAR10(_PreparedCIFAR):
    folder = "cifar10"


class PreparedCIFAR100(_PreparedCIFAR):
    folder = "cifar100"

# name, numClasses
availableDatasets = {
    "ImageNet": [ImageNet, 1000],
    "TinyImageNet": [TinyImageNet, 200],
    "CIFAR10" : [PreparedCIFAR10, 10],
    "CIFAR100": [PreparedCIFAR100, 100],
}
dataSetStatistics = {
    "ImageNet": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    "TinyImageNet": [[0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]],
    "CIFAR10" : [[0.4913999140262604, 0.48215872049331665, 0.4465313255786896], [0.24703197181224823, 0.243484228849411, 0.26158687472343445]],
    "CIFAR100": [[0.5070753693580627, 0.4865487813949585, 0.44091784954071045], [0.2673334777355194, 0.25643861293792725, 0.2761503756046295]],
}


def _get_label_noise_config() -> tuple[float, int]:
    """
    Label-noise controls via environment variables (keeps Args parser backward-compatible):
      LABEL_NOISE_RATE: float in [0, 1], default 0.0
      LABEL_NOISE_SEED: int, default 12345
    """
    rate_raw = os.environ.get("LABEL_NOISE_RATE", "0")
    seed_raw = os.environ.get("LABEL_NOISE_SEED", "12345")
    try:
        rate = float(rate_raw)
    except ValueError as error:
        raise RuntimeError(f"Invalid LABEL_NOISE_RATE='{rate_raw}'. Must be float in [0, 1].") from error
    if rate < 0 or rate > 1:
        raise RuntimeError(f"LABEL_NOISE_RATE must be in [0, 1], got {rate}.")
    try:
        seed = int(seed_raw)
    except ValueError as error:
        raise RuntimeError(f"Invalid LABEL_NOISE_SEED='{seed_raw}'. Must be an integer.") from error
    return rate, seed


def _apply_symmetric_label_noise(dataset, num_classes: int, noise_rate: float, noise_seed: int) -> int:
    """Resample selected labels uniformly, allowing the original class.

    Returns the number of labels that actually changed.
    """
    if noise_rate <= 0:
        return 0
    if not hasattr(dataset, "targets"):
        raise RuntimeError("Dataset does not expose .targets; cannot apply label noise.")
    targets = np.asarray(dataset.targets, dtype=np.int64)
    if targets.ndim != 1:
        targets = targets.reshape(-1)
    n = len(targets)
    rng = np.random.default_rng(noise_seed)
    num_noisy = int(round(n * noise_rate))
    if num_noisy <= 0:
        return 0
    noisy_indices = rng.choice(n, size=num_noisy, replace=False)
    original = targets[noisy_indices]
    # Uniform over all classes, including the original label.
    new_labels = rng.integers(0, num_classes, size=num_noisy)
    targets[noisy_indices] = new_labels
    dataset.targets = targets.tolist()
    return int(np.count_nonzero(new_labels != original))

def worker_init_fn(id):
    """ manually seed each workers np random generator. see https://github.com/pytorch/pytorch/issues/5059"""
    uint64_seed = torch.initial_seed()
    np.random.seed([uint64_seed >> 32, uint64_seed & 0xffff_ffff])


class NormalizeMinMax(torch.nn.Module):
    """Normalize a x\\in[0,1] tensor to x\\in[-1,1]"""
    def __init__(self, inplace = False):
        super().__init__()
        self.min = -1
        self.max = 1
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.inplace:
            tensor = tensor.clone()

        tensor.mul_(self.max-self.min).add_(self.min)
        return tensor

class CenterCrop(torch.nn.Module):
    def __init__(self, crop_fraction = 0.875):
        super().__init__()
        self.crop_fraction = crop_fraction

    def forward(self, img):
        size = int(min(img.size) * self.crop_fraction)
        return transforms.functional.center_crop(img, size)


class DataLoader:
    def __init__(self):
        self.datasetName = Args.dataset
        try:
            self.dataset, self.numClasses = availableDatasets[self.datasetName]
        except KeyError:
            raise NameError(f"Dataset {self.datasetName} not found. Available datasets are: {', '.join(availableDatasets.keys())}")

        if Args.normalize not in ["std", "min_max"]:
            raise RuntimeError(f"Arguments 'normalize' must be 'std' or 'min_max' but is '{Args.normalize}'")

        if Args.normalize == "std":
            if self.datasetName in dataSetStatistics:
                mean, std = dataSetStatistics[self.datasetName]
            else:
                print("Calculating Mean and Std...")
                mean, std = self._get_statistics()
                print(f"Mean = {list(map(float,mean))}, Std = {list(map(float,std))}")


        transform_list = []
        if Args.mixup:
            raise NotImplementedError #@TODO add mixup
        if Args.randAugment:
            transform_list.append(transforms.RandAugment(magnitude = Args.randAugment_magnitude))
        if Args.flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        if Args.cut:
            cutout_mask = mean if Args.normalize == "std" else 0.5
            transform_list.append(Cutout(mask_val=cutout_mask))
        if Args.crop and self.datasetName in ["CIFAR10","CIFAR100"]:
            transform_list.append(transforms.RandomCrop(size=(32, 32), padding=4))
        if Args.crop and self.datasetName == "TinyImageNet":
            transform_list.append(transforms.RandomCrop(size=(64, 64), padding=4))

        if Args.imageSize != 0:
            transform_list += [
                transforms.Resize(Args.imageSize, interpolation = transforms.InterpolationMode.BICUBIC),
            ]

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean, std) if Args.normalize == "std" else NormalizeMinMax(),
        ]
        train_transform = transforms.Compose(transform_list)

        if Args.imageSize != 0:
            test_transform = [
                transforms.Resize(Args.imageSize, interpolation = transforms.InterpolationMode.BICUBIC),
            ]
        else:
            test_transform = []

        test_transform = transforms.Compose(test_transform + [
            transforms.ToTensor(),
            transforms.Normalize(mean, std) if Args.normalize == "std" else NormalizeMinMax(),
        ])

        train_set = self.dataset(root=Args.dataDir, train=True, download=True, transform=train_transform)
        test_set  = self.dataset(root=Args.dataDir, train=False, download=True, transform=test_transform)
        
        if self.datasetName in ["CIFAR10", "CIFAR100"]:
            noise_rate, noise_seed = _get_label_noise_config()
            if noise_rate > 0:
                num_noisy = _apply_symmetric_label_noise(
                    train_set,
                    num_classes=self.numClasses,
                    noise_rate=noise_rate,
                    noise_seed=noise_seed,
                )
                if torch.distributed.get_rank() == 0:
                    print(
                        f"[LABEL_NOISE] dataset={self.datasetName} rate={noise_rate:.2f} "
                        f"seed={noise_seed} mode=uniform "
                        f"selected_labels={int(round(len(train_set) * noise_rate))}/{len(train_set)} "
                        f"noisy_labels={num_noisy}/{len(train_set)} "
                        f"actual_noise_rate={num_noisy / len(train_set):.6f}"
                    )

        if self.datasetName == "ImageNet":
            assert len(train_set) == 1281167, "train data is incomplete!"
            assert len(test_set) == 50000, "test data is incomplete!"
        if self.datasetName == "TinyImageNet":
            assert len(train_set) == 100000, "TinyImageNet train data is incomplete!"
            assert len(test_set) == 10000, "TinyImageNet val data is incomplete!"
        
        # to get random order of train data the seed has to be set manually. the seed value must be constant among all processes. 
        seed = torch.tensor(random.getrandbits(32)).cuda(torch.distributed.get_rank() % torch.cuda.device_count())
        torch.distributed.broadcast(seed, src = 0)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True , seed = seed.item())
        test_sampler  = torch.utils.data.distributed.DistributedSampler(test_set , shuffle=False)
        
        #calc BS per worker
        batch_size = Args.batchSize // torch.distributed.get_world_size() + int(Args.batchSize % torch.distributed.get_world_size() > torch.distributed.get_rank())
        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=Args.dataThreads, worker_init_fn=worker_init_fn, sampler = train_sampler)
        self.test  = torch.utils.data.DataLoader(test_set , batch_size=batch_size, num_workers=Args.dataThreads, worker_init_fn=worker_init_fn, sampler = test_sampler)

    def _get_statistics(self):
        data = self.dataset(root=Args.dataDir, train=True, download=True, transform=transforms.ToTensor())
        firstMoment = torch.zeros(3)
        secondMoment = torch.zeros(3)
        N = 0
        for inputs, _ in torch.utils.data.DataLoader(data, batch_size=64, num_workers=Args.dataThreads):
            N += inputs.shape[0]
            for i in range(3):
                firstMoment[i] += inputs[:,i,:,:].mean()*inputs.shape[0]
                secondMoment[i] += (inputs[:,i,:,:]**2).mean()*inputs.shape[0]
        firstMoment .div_(N)
        secondMoment.div_(N)
        return firstMoment, torch.sqrt(secondMoment-firstMoment**2)
