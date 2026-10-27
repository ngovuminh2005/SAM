#!/usr/bin/env python3
"""Download official corruption archives or the SVHN test split."""
import argparse
from pathlib import Path
from torchvision.datasets import SVHN
from torchvision.datasets.utils import download_and_extract_archive

ARCHIVES = {
 'cifar10_c': ('https://zenodo.org/records/2535967/files/CIFAR-10-C.tar', '56bf5dcef84df0e2308c6dcbcbbd8499', 'CIFAR-10-C'),
 'cifar100_c': ('https://zenodo.org/records/3555552/files/CIFAR-100-C.tar', '11f0ed0f1191edbf9fa23466ae6021d3', 'CIFAR-100-C'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=('cifar10_c','cifar100_c','svhn'), required=True)
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[1]/'datasets')
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        if name == 'svhn':
            data = SVHN(str(args.data_root/'SVHN'), split='test', download=True)
            print(f'[READY] SVHN test: {len(data)} examples')
        else:
            url, checksum, folder = ARCHIVES[name]
            expected = ('gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                        'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast',
                        'elastic_transform', 'pixelate', 'jpeg_compression', 'labels')
            if all((args.data_root/folder/(item+'.npy')).is_file() for item in expected):
                print(f'[SKIP] {args.data_root/folder} already contains all selected files')
                continue
            download_and_extract_archive(url, str(args.data_root/'.downloads'),
                extract_root=str(args.data_root), filename=folder+'.tar', md5=checksum,
                remove_finished=True)
            if not (args.data_root/folder/'labels.npy').is_file():
                raise RuntimeError(f'Archive did not contain expected {folder}/labels.npy')
            print(f'[READY] {args.data_root/folder}')


if __name__ == '__main__':
    main()
