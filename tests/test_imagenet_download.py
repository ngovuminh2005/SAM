import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


class ImageNetDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fixtures = self.root / 'fixtures'
        self.fixtures.mkdir()
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.classes = ('n00000001', 'n00000002')
        train = self.fixtures / 'ILSVRC2012_img_train.tar'
        with tarfile.open(train, 'w') as outer:
            for index, name in enumerate(self.classes):
                stream = io.BytesIO()
                with tarfile.open(fileobj=stream, mode='w') as inner:
                    for image in range(2 if index == 0 else 1):
                        self.add_member(inner, f'{name}_{image}.JPEG', b'original JPEG bytes')
                self.add_member(outer, name + '.tar', stream.getvalue())
        val = self.fixtures / 'ILSVRC2012_img_val.tar'
        with tarfile.open(val, 'w') as archive:
            for index in range(2):
                self.add_member(archive, f'val_{index}.JPEG', b'validation JPEG bytes')
        mapping = '\n'.join(f'mkdir -p {name}\nmv val_{index}.JPEG {name}/' for index, name in enumerate(self.classes)) + '\n'
        (self.fixtures / 'valprep.sh').write_text(mapping)
        script = Path('scripts/get_imagenet.sh').read_text()
        for original, updated in {
            '1281167': '3', '50000': '2', '-eq 1000': '-eq 2',
            '1d675b47d978889d74fa0da5fadfb00e': hashlib.md5(train.read_bytes()).hexdigest(),
            '29b22e2961454d5413ddabcf34fc5622': hashlib.md5(val.read_bytes()).hexdigest(),
            'a05b97191cba09e0b65ffc4fde811fc6bd9898a4': hashlib.sha1(b'blob ' + str(len(mapping.encode())).encode() + b'\0' + mapping.encode()).hexdigest(),
        }.items():
            script = script.replace(original, updated)
        self.script = self.root / 'get_imagenet.sh'
        self.script.write_text(script)
        self.write_executable('df', '#!/bin/sh\nprintf "Filesystem 1024-blocks Used Available Capacity Mounted\\nfixture 999999999 0 999999999 0%% /\\n"\n')
        self.write_executable('wget', f'''#!{sys.executable}
import os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
name = args[-1].rsplit('/', 1)[-1]
with open(os.environ['DOWNLOAD_LOG'], 'a') as log:
    log.write(name + '\\n')
target = args[args.index('-O') + 1]
if os.environ.get('CORRUPT_DOWNLOAD'):
    Path(target).write_bytes(b'login page')
else:
    shutil.copyfile(Path(os.environ['DOWNLOAD_FIXTURES']) / name, target)
''')
        self.data_root = self.root / 'data with spaces'
        self.env = {**os.environ, 'PATH': str(self.bin) + os.pathsep + os.environ['PATH'],
                    'DATA_ROOT': str(self.data_root), 'DOWNLOAD_FIXTURES': str(self.fixtures),
                    'DOWNLOAD_LOG': str(self.root / 'downloads.log')}

    @staticmethod
    def add_member(archive, name, data):
        info = tarfile.TarInfo(name)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))

    def write_executable(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)

    def run_download(self, *args, env=None):
        return subprocess.run(['bash', str(self.script), *args], env=env or self.env, text=True, capture_output=True)

    def test_prepare_layout_and_rerun_without_download(self):
        result = self.run_download()
        self.assertEqual(result.returncode, 0, result.stderr)
        root = self.data_root / 'ImageNet'
        self.assertEqual(sorted(p.name for p in (root / 'train').iterdir()), list(self.classes))
        self.assertEqual(sorted(p.name for p in (root / 'val').iterdir()), list(self.classes))
        self.assertEqual(len(list(root.glob('train/*/*.JPEG'))), 3)
        self.assertEqual(len(list(root.glob('val/*/*.JPEG'))), 2)
        self.assertEqual((root / 'train/n00000001/n00000001_0.JPEG').read_bytes(), b'original JPEG bytes')
        before = (self.root / 'downloads.log').read_text()
        result = self.run_download()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / 'downloads.log').read_text(), before)

    def test_reject_corrupt_download_before_extracting(self):
        result = self.run_download('--train', env={**self.env, 'CORRUPT_DOWNLOAD': '1'})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Invalid ILSVRC2012_img_train.tar', result.stderr)
        self.assertFalse((self.data_root / 'ImageNet/train').exists())

    def test_resume_interrupted_class_extraction(self):
        self.write_executable('tar', '''#!/bin/bash
set -e
/bin/tar "$@"
for arg in "$@"; do
    if [[ "${arg}" == */n00000001.tar && ! -f "${DOWNLOAD_LOG}.interrupted" ]]; then
        touch "${DOWNLOAD_LOG}.interrupted"
        exit 4
    fi
done
''')
        first = self.run_download('--train')
        self.assertNotEqual(first.returncode, 0)
        self.assertFalse((self.data_root / 'ImageNet/train').exists())
        second = self.run_download('--train')
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual((self.root / 'downloads.log').read_text().splitlines(), ['ILSVRC2012_img_train.tar'])
        self.assertEqual(len(list((self.data_root / 'ImageNet').glob('train/*/*.JPEG'))), 3)

    def test_reject_modified_validation_mapping(self):
        (self.fixtures / 'valprep.sh').write_text('exit 0\n')
        result = self.run_download('--val')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('mapping checksum mismatch', result.stderr)
        self.assertFalse((self.data_root / 'ImageNet/val').exists())

    def test_reject_insufficient_disk_and_unknown_option(self):
        self.write_executable('df', '#!/bin/sh\nprintf "Filesystem 1024-blocks Used Available Capacity Mounted\\nfixture 100 0 100 0%% /\\n"\n')
        result = self.run_download('--train')
        self.assertEqual(result.returncode, 2)
        self.assertIn('300 GiB', result.stderr)
        self.assertFalse((self.root / 'downloads.log').exists())
        self.assertEqual(self.run_download('--unknown').returncode, 2)

    def test_dataset_dispatch_and_legacy_wrapper(self):
        repo = self.root / 'repo'
        scripts = repo / 'scripts'
        (scripts / 'ood').mkdir(parents=True)
        for path in ('download_datasets.sh', 'ood/prepare_datasets.sh'):
            (scripts / path).write_text((Path('scripts') / path).read_text())
        (scripts / 'get_imagenet.sh').write_text('printf "imagenet:%s\\n" "$DATA_ROOT"\n')
        for path in ('prepare_datasets.py', 'prepare_ood_datasets.py'):
            (scripts / path).write_text('import json, sys\nprint(json.dumps(sys.argv[1:]))\n')
        env = {**self.env, 'PYTHON': sys.executable}
        default = subprocess.run(['bash', str(scripts / 'download_datasets.sh')], env=env, text=True, capture_output=True)
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(default.stdout.strip(), f'imagenet:{self.data_root}')
        selected = subprocess.run(['bash', str(scripts / 'download_datasets.sh'), 'cifar10', 'svhn'], env=env, text=True, capture_output=True)
        self.assertEqual(selected.returncode, 0, selected.stderr)
        args = [json.loads(line) for line in selected.stdout.splitlines()]
        self.assertEqual(args[0], ['--dataset', 'cifar10', '--data-root', str(self.data_root)])
        self.assertEqual(args[1], ['--datasets', 'svhn', '--data-root', str(self.data_root)])
        legacy = subprocess.run(['bash', str(scripts / 'ood/prepare_datasets.sh')], env=env, text=True, capture_output=True)
        self.assertEqual(legacy.returncode, 0, legacy.stderr)
        self.assertEqual([json.loads(line)[1] for line in legacy.stdout.splitlines()], ['cifar10_c', 'cifar100_c', 'svhn'])


if __name__ == '__main__':
    unittest.main()
