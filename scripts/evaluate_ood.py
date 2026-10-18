#!/usr/bin/env python3
"""Evaluate a saved vision checkpoint on one configured OOD task."""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ood import OODSetting, create_task
from ood.checkpoint import load_vision_checkpoint
from utility import tracking


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', required=True, help='JSON task config under configs/ood')
    parser.add_argument('--run-dir', required=True, help='Training run containing params.json and final.model')
    parser.add_argument('--checkpoint', default='final.model')
    parser.add_argument('--data-root', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--max-examples', default=None, type=int, help='Diagnostic subset per split/case; 0 means full')
    args = parser.parse_args()
    setting = OODSetting.from_json(args.setting,
        data_root=args.data_root or os.environ.get('SAM_DATASET_ROOT'), max_examples=args.max_examples)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu')
    model, params, checkpoint = load_vision_checkpoint(args.run_dir, args.checkpoint, setting.id_dataset, device)
    task = create_task(setting, params)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = Path(args.output_dir).resolve() if args.output_dir else checkpoint.parent / 'ood' / setting.name / stamp
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'settings': asdict(setting), 'checkpoint': str(checkpoint),
                'training_params': str(checkpoint.parent / 'params.json'), 'device': str(device)}
    tracking.start({**params, 'ood': asdict(setting), 'checkpoint': str(checkpoint)}, output,
                   name=f'{checkpoint.parent.name}_{setting.name}_{stamp}', job_type='ood')
    partial = []
    def on_case(result):
        partial.append(result)
        tracking.log({'ood_case': len(partial), **{f'ood/{result["case"]}/{k}': v for k, v in result['metrics'].items()}})
        write_json(output / 'partial_results.json', {**metadata, 'cases': partial})
        print(f'[OOD] {result["case"]}: {result["metrics"]}', flush=True)
    result = {**metadata, **task.run(model, device, on_case=on_case)}
    result['conventions'] = {'scale': 'accuracy, AUROC, AUPR and FPR are fractions in [0,1]',
        'scores': 'larger scores indicate OOD',
        'fpr95': 'ID is positive: fraction of OOD accepted when ID TPR >= 0.95',
        'mean_error': 'unnormalized corruption error, not AlexNet-normalized mCE'}
    write_json(output / 'results.json', result)
    rows = [{**{k:v for k,v in case.items() if k != 'metrics'}, **case['metrics']} for case in result['cases']]
    columns = sorted({key for row in rows for key in row})
    with (output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    tracking.log({'ood_case': len(partial), **{f'ood/aggregate/{key}': value for key, value in result['aggregate'].items()}})
    tracking.summary({'ood': result['aggregate'], 'diagnostic_subset': result['diagnostic_subset']})
    print(f'[SAVE] {output}/results.json and results.csv', flush=True)


if __name__ == '__main__':
    tracking.execute(main)
