"""Download and validate the four WiSE-FT ImageNet distribution shifts."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import zipfile


DATASETS = [
    dict(name='imagenet-a', archive='imagenet-a.tar', images=7500, classes=200,
         size=687552512, url='https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar'),
    dict(name='imagenet-r', archive='imagenet-r.tar', images=30000, classes=200,
         size=2191079936, url='https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'),
    dict(name='imagenetv2-matched-frequency-format-val', archive='imagenetv2-matched-frequency.tar.gz',
         images=10000, classes=1000, size=1264079360,
         sha256='f0c37fdf925916b19ea1323cd9a2208cdb6959ba2c32eef2a7fc393835c9ca7c',
         url='https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/'
             'd626240be2538720e83103a0e1178d24aca8b12c/imagenetv2-matched-frequency.tar.gz?download=true'),
    dict(name='sketch', archive='ImageNet-Sketch.zip', images=50889, classes=1000,
         sha256='4d3ce6b82a7c48aabf14aeac105eb533e75707f152cc858f55bfc022031a72ed',
         url='https://huggingface.co/datasets/nateraw/imagenet-sketch-data/resolve/'
             'd00ab762ad9e29dcd6b08a9d542b2057550162d1/ImageNet-Sketch.zip?download=true'),
]
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}


def ready(path, spec):
    if not path.is_dir():
        return False
    numeric = spec['name'].startswith('imagenetv2')
    pattern = r'[0-9]+' if numeric else r'n[0-9]{8}'
    classes = [p for p in path.iterdir() if p.is_dir() and re.fullmatch(pattern, p.name)]
    if len(classes) != spec['classes']:
        return False
    if numeric and {int(p.name) for p in classes} != set(range(1000)):
        return False
    count = sum(sum(1 for p in directory.rglob('*')
                    if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES) for directory in classes)
    return count == spec['images']


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def extract(archive, staging):
    def safe(name):
        target = (staging / name).resolve()
        if target != staging and staging not in target.parents:
            raise ValueError('Unsafe archive member: ' + name)

    if archive.name.endswith('.zip'):
        with zipfile.ZipFile(archive) as source:
            for name in source.namelist():
                safe(name)
            source.extractall(staging)
    else:
        with tarfile.open(archive, 'r:*') as source:
            for member in source.getmembers():
                safe(member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError('Unsupported archive member: ' + member.name)
            source.extractall(staging)


def prepare(root, spec, keep_archives=False):
    destination = root / spec['name']
    if ready(destination, spec):
        print('[SKIP]', spec['name'], 'already prepared', flush=True)
        return
    if destination.exists():
        raise RuntimeError('Incomplete destination exists; move it aside: ' + str(destination))
    archive = root / spec['archive']
    partial = root / (spec['archive'] + '.part')
    if not archive.exists():
        print('[DOWNLOAD]', spec['name'], flush=True)
        if not (spec.get('size') and partial.exists() and partial.stat().st_size == spec['size']):
            subprocess.run(['curl', '-fL', '--retry', '5', '--retry-delay', '5',
                            '--connect-timeout', '20', '-C', '-', '-o', str(partial), spec['url']], check=True)
        partial.rename(archive)
    if spec.get('size') and archive.stat().st_size != spec['size']:
        raise RuntimeError('Archive size mismatch: ' + str(archive))
    checksum = digest(archive)
    if spec.get('sha256') and checksum != spec['sha256']:
        raise RuntimeError('Archive checksum mismatch: ' + str(archive))
    print('[EXTRACT]', spec['name'], flush=True)
    staging = root / '.test-extract' / spec['name']
    staging.mkdir(parents=True, exist_ok=True)
    extract(archive, staging)
    candidates = [staging, *[p for p in staging.iterdir() if p.is_dir()]]
    candidates = [p for p in candidates if ready(p, spec)]
    if len(candidates) != 1:
        raise RuntimeError('Extracted class/image counts are incorrect for ' + spec['name'])
    candidates[0].rename(destination)
    if staging.exists():
        shutil.rmtree(staging)
    manifest_path = root / '.imagenet-tests-manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[spec['name']] = dict(url=spec['url'], sha256=checksum,
                                images=spec['images'], classes=spec['classes'])
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(manifest_path)
    if not keep_archives:
        archive.unlink()
    print('[READY]', spec['name'], spec['images'], 'images', spec['classes'], 'classes', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=os.environ.get('DATA_ROOT') or
                        str(Path(__file__).resolve().parents[1] / 'datasets'))
    parser.add_argument('--keep-archives', action='store_true')
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.prepare-tests.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for spec in DATASETS:
            prepare(root, spec, args.keep_archives)
    print('[READY] All four WiSE-FT test datasets:', root, flush=True)


if __name__ == '__main__':
    main()
