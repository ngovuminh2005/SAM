"""Evaluation-only datasets and preprocessing derived from checkpoint settings."""
from pathlib import Path
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from utility.data import PreparedCIFAR10, PreparedCIFAR100, dataSetStatistics


def evaluation_transform(id_dataset, params):
    steps = []
    size = int(params.get('imageSize', 0))
    if size:
        steps.append(transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC))
    steps.append(transforms.ToTensor())
    normalization = params.get('normalize', 'std')
    if normalization == 'std':
        mean, std = dataSetStatistics[id_dataset]
        steps.append(transforms.Normalize(mean, std))
    elif normalization == 'min_max':
        steps.append(transforms.Normalize((.5, .5, .5), (.5, .5, .5)))
    else:
        raise ValueError(f'Unknown checkpoint normalization {normalization}')
    return transforms.Compose(steps)


def clean_dataset(name, root, transform):
    if name in ('CIFAR10', 'CIFAR100'):
        return {'CIFAR10': PreparedCIFAR10, 'CIFAR100': PreparedCIFAR100}[name](root, train=False, transform=transform)
    if name == 'SVHN':
        location = Path(root) / 'SVHN'
        if not (location / 'test_32x32.mat').is_file():
            raise FileNotFoundError(f'SVHN test data missing: {location}/test_32x32.mat; see scripts/ood/prepare_datasets.sh')
        return datasets.SVHN(str(location), split='test', download=False, transform=transform)
    raise ValueError(f'Unsupported evaluation dataset {name}')


class CorruptedCIFAR(Dataset):
    def __init__(self, root, corruption, severity, transform, samples_per_severity=10000):
        self.images_path = str(Path(root) / f'{corruption}.npy')
        labels_path = Path(root) / 'labels.npy'
        if not Path(self.images_path).is_file() or not labels_path.is_file():
            raise FileNotFoundError(f'Missing {self.images_path} or {labels_path}; see scripts/ood/prepare_datasets.sh')
        data = np.load(self.images_path, mmap_mode='r', allow_pickle=False)
        labels = np.load(labels_path, allow_pickle=False)
        n = samples_per_severity
        if severity not in range(1, 6):
            raise ValueError('severity must be in [1, 5]')
        if data.shape != (5*n, 32, 32, 3) or data.dtype != np.uint8:
            raise ValueError(f'Invalid corruption array shape/dtype: {data.shape}, {data.dtype}')
        if labels.ndim != 1 or len(labels) not in (n, 5*n) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('Expected labels.npy with 10000 or 50000 integer labels')
        self.offset = (severity - 1)*n
        self.targets = labels if len(labels) == n else labels[self.offset:self.offset+n]
        self.count = n
        self.transform = transform
        self._images = None

    def __len__(self):
        return self.count

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_images'] = None
        return state

    def __getitem__(self, index):
        if self._images is None:
            self._images = np.load(self.images_path, mmap_mode='r', allow_pickle=False)
        image = transforms.functional.to_pil_image(np.array(self._images[self.offset+index], copy=True))
        return self.transform(image), int(self.targets[index])
