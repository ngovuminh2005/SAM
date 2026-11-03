"""Run the training data pipeline on real samples and report problems.

Checks, per data source (coco, gqa, ocr_vqa, textvqa, vg, text-only):
missing images, tokenization mismatches (samples whose labels end up fully
masked), conversation contexts, image tensors and collated batches.

    python scripts/check_data.py --tokenizer lmsys/vicuna-7b-v1.5 --samples 500
"""

import argparse
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import transformers

from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.train.train import DataArguments, DataCollatorForSupervisedDataset, LazySupervisedDataset


def source_of(sample):
    return sample['image'].split('/')[0] if 'image' in sample else 'text-only'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data_path', default='./data/llava_finetune/llava_v1_5_mix665k.json')
    parser.add_argument('--image_folder', default='./data/llava_finetune')
    parser.add_argument('--tokenizer', default='lmsys/vicuna-7b-v1.5')
    parser.add_argument('--vision_tower', default='openai/clip-vit-large-patch14-336')
    parser.add_argument('--version', default='v1_w_context')
    parser.add_argument('--model_max_length', type=int, default=2048)
    parser.add_argument('--samples', type=int, default=500, help='samples per data source')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer, model_max_length=args.model_max_length, padding_side='right', use_fast=False)
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
    data_args = DataArguments(data_path=args.data_path, lazy_preprocess=True, is_multimodal=True,
                              image_folder=args.image_folder, image_aspect_ratio='pad')
    data_args.image_processor = transformers.CLIPImageProcessor.from_pretrained(args.vision_tower)
    data_args.mm_use_im_start_end = False
    dataset = LazySupervisedDataset(args.data_path, tokenizer, data_args)
    samples = dataset.list_data_dict

    missing = Counter(source_of(s) for s in samples
                      if 'image' in s and not os.path.exists(os.path.join(args.image_folder, s['image'])))
    by_source = defaultdict(list)
    for i, s in enumerate(samples):
        by_source[source_of(s)].append(i)
    print(f'{len(samples)} samples; per source: {dict((k, len(v)) for k, v in sorted(by_source.items()))}')
    print(f'missing images: {dict(missing) or 0}')
    print(f'tokenizer legacy={tokenizer.legacy} pad={tokenizer.pad_token!r} version={args.version}')

    random.seed(args.seed)
    collator = DataCollatorForSupervisedDataset(tokenizer)
    for source, indices in sorted(by_source.items()):
        chosen = [i for i in indices if source not in missing or
                  os.path.exists(os.path.join(args.image_folder, samples[i]['image']))]
        chosen = random.sample(chosen, min(args.samples, len(chosen)))
        if not chosen:
            print(f'{source:10s} skipped (no images on disk)')
            continue
        stats = Counter()
        items = []
        for i in chosen:
            item = dataset[i]
            items.append(item)
            labels = item['labels']
            image_tokens = int((item['input_ids'] == IMAGE_TOKEN_INDEX).sum())
            stats['fully_masked'] += int((labels != IGNORE_INDEX).sum() == 0)
            stats['image_token_wrong'] += int(image_tokens != (1 if 'image' in samples[i] else 0))
            stats['context_has_image_tag'] += int('<image>' in item.get('conversations_context', ''))
            stats['context_empty'] += int(not item.get('conversations_context', '').strip('</s> '))
            stats['image_shape_wrong'] += int(tuple(item['image'].shape) != (3, 336, 336))
            # Text tokens plus 575 extra image positions beyond model_max_length are truncated in the model.
            stats['over_max_length'] += int(len(labels) + 575 * image_tokens > args.model_max_length)
        batch = collator(items[:8])
        report = ', '.join(f'{k}={v}' for k, v in sorted(stats.items()))
        print(f'{source:10s} n={len(chosen):4d} {report}; batch input_ids {tuple(batch["input_ids"].shape)} '
              f'images {tuple(batch["images"].shape)} contexts {len(batch["conversations_context"])}')
        example = items[0]
        if source in ('coco', 'text-only'):
            print(f'           context example: {example["conversations_context"][:150]!r}')


if __name__ == '__main__':
    main()
