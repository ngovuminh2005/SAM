"""Download the OCR-VQA images used by llava_v1_5_mix665k.json, saved as .jpg like LLaVA.

The official loadDataset.py stops in pdb at every image; this downloads only the
80K images the LLaVA json references, in parallel, and resumes on rerun.

    python scripts/download_ocrvqa.py --dataset data/llava_finetune/ocr_vqa/dataset.json \
        --llava data/llava_finetune/llava_v1_5_mix665k.json --out data/llava_finetune/ocr_vqa/images
"""

import argparse
import io
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
from tqdm import tqdm


def fetch(url, path, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
            image = Image.open(io.BytesIO(data))
            tmp = path + '.tmp'
            if image.format == 'JPEG':
                with open(tmp, 'wb') as handle:
                    handle.write(data)
            else:
                image.convert('RGB').save(tmp, 'JPEG', quality=95)
            os.replace(tmp, path)
            return None
        except Exception as error:  # network errors, dead links, broken images
            last = error
    return f'{url}: {last}'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--dataset', required=True, help='OCR-VQA dataset.json')
    parser.add_argument('--llava', required=True, help='llava_v1_5_mix665k.json')
    parser.add_argument('--out', required=True, help='output folder (ocr_vqa/images)')
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()

    with open(args.dataset) as handle:
        dataset = json.load(handle)
    with open(args.llava) as handle:
        needed = {os.path.splitext(os.path.basename(x['image']))[0]
                  for x in json.load(handle) if x.get('image', '').startswith('ocr_vqa/')}
    unknown = needed - dataset.keys()
    if unknown:
        print(f'{len(unknown)} images referenced by the LLaVA json are not in dataset.json')
    os.makedirs(args.out, exist_ok=True)
    todo = [(dataset[k]['imageURL'], os.path.join(args.out, f'{k}.jpg'))
            for k in sorted(needed & dataset.keys())
            if not os.path.exists(os.path.join(args.out, f'{k}.jpg'))]
    print(f'{len(needed)} images needed, {len(todo)} left to download')

    failures = []
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch, url, path) for url, path in todo]
        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.result():
                failures.append(future.result())
    if failures:
        log = os.path.join(args.out, '..', 'failed_downloads.txt')
        with open(log, 'w') as handle:
            handle.write('\n'.join(failures) + '\n')
        print(f'{len(failures)} downloads failed; rerun to retry (list in {os.path.abspath(log)})')
    else:
        print('done')


if __name__ == '__main__':
    main()
