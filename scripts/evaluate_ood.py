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
    parser.add_argument('--severities', nargs='+', type=int, default=None,
                        help='Corruption severities to evaluate, each in [1, 5]')
    args = parser.parse_args()
    setting = OODSetting.from_json(args.setting,
        data_root=args.data_root or os.environ.get('SAM_DATASET_ROOT'), max_examples=args.max_examples,
        severities=args.severities)
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
    tracking.start({'severity': setting.severities[0] if len(setting.severities) == 1
                    else ','.join(map(str, setting.severities))}, output,
                   name=f'{checkpoint.parent.name}_{setting.name}_{stamp}', job_type='ood')
    partial = []
    def on_case(result):
        partial.append(result)
        if setting.kind == 'corruption' and not result.get('is_clean'):
            tracking.log({f"ood/{result['corruption']}": result['metrics']['accuracy']})
        elif setting.kind == 'detection':
            tracking.log({f'ood/{key}': value for key, value in result['metrics'].items()})
        write_json(output / 'partial_results.json', {**metadata, 'cases': partial})
        if setting.kind == 'corruption':
            if not result.get('is_clean'):
                print(f"{result['corruption']}: accuracy={result['metrics']['accuracy']:.4f}", flush=True)
        else:
            print(f'[OOD] {result["case"]}: {result["metrics"]}', flush=True)
    result = {**metadata, **task.run(model, device, on_case=on_case)}
    if setting.kind == 'corruption' and len(setting.severities) == 1:
        # Each severity-specific run averages metrics across the selected corruptions.
        result['aggregate']['avg_accuracy'] = result['aggregate']['mean_accuracy']
        result['aggregate']['avg_error'] = result['aggregate']['mean_error']
        result['aggregate']['avg_loss'] = result['aggregate']['mean_loss']
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
    aggregate = result['aggregate']
    if setting.kind == 'corruption':
        avg_metrics = {'avg_accuracy': aggregate.get('avg_accuracy', aggregate.get('mean_accuracy'))}
        tracking.log({f'ood/{key}': value for key, value in avg_metrics.items() if value is not None})
        tracking.summary({'ood': avg_metrics})
    else:
        tracking.log({f'ood/{key}': value for key, value in aggregate.items()})
        tracking.summary({'ood': aggregate})
    print(f'[SAVE] {output}/results.json and results.csv', flush=True)


if __name__ == '__main__':
    tracking.execute(main)
