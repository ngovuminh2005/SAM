#!/usr/bin/env python3
"""Create a W&B-style summary table from CIFAR-C OOD evaluation results."""
import argparse
import csv
import json
from pathlib import Path


def model_label(params):
    model = params.get('model', 'unknown')
    if model == 'resnet':
        return f"ResNet{params.get('depth', '')}"
    if model == 'WRN':
        return f"WRN-{params.get('depth', '')}-{params.get('widthFactor', '')}"
    if model == 'PyramidNet':
        return f"PyramidNet-{params.get('depth', '')}"
    if model == 'DenseNet':
        return f"DenseNet-{params.get('depth', '')}"
    return str(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', required=True, type=Path)
    parser.add_argument('--markdown', required=True, type=Path)
    parser.add_argument('--csv', required=True, type=Path)
    args = parser.parse_args()

    records = []
    for result_path in sorted(args.results_root.glob('*/results.json')):
        result = json.loads(result_path.read_text())
        aggregate = result.get('aggregate', {})
        params_path = Path(result['training_params'])
        params = json.loads(params_path.read_text())
        run_name = result_path.parent.name
        dataset = result.get('settings', {}).get('id_dataset', params.get('dataset', ''))
        severities = result.get('settings', {}).get('severities', [])
        records.append({
            'W&B run': run_name,
            'W&B group': f"ood_{dataset.lower()}_c_sev{'-'.join(map(str, severities))}",
            'Dataset': dataset,
            'Model': model_label(params),
            'Severity': ','.join(map(str, severities)),
            'Avg accuracy (%)': _percent(aggregate.get('avg_accuracy', aggregate.get('mean_accuracy'))),
            'Checkpoint': str(result.get('checkpoint', '')),
            'Results': str(result_path),
        })

    records.sort(key=lambda row: (row['Dataset'], row['Model'], row['W&B run']))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else [
        'W&B run', 'W&B group', 'Dataset', 'Model', 'Severity',
        'Avg accuracy (%)', 'Checkpoint', 'Results'
    ]
    with args.csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    headers = ['W&B run', 'State', 'Dataset', 'Model', 'Severity', 'Avg accuracy', 'Results']
    lines = ['# CIFAR-C OOD evaluation', '', '| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in records:
        result_link = f"[{Path(row['Results']).name}]({row['Results']})"
        lines.append('| ' + ' | '.join([
            row['W&B run'], 'Finished', row['Dataset'], row['Model'], row['Severity'],
            _fmt(row['Avg accuracy (%)']), result_link,
        ]) + ' |')
    if not records:
        lines.append('| No completed evaluations | — | — | — | — | — | — |')
    lines.extend(['', f"Completed evaluations: {len(records)}", ''])
    args.markdown.write_text('\n'.join(lines))
    print(f'[SUMMARY] {len(records)} completed evaluation(s)')


def _percent(value):
    return '' if value is None else round(float(value) * 100, 4)


def _fmt(value):
    return '—' if value == '' else f'{value:.2f}%'


if __name__ == '__main__':
    main()
