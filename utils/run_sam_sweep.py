"""Create/resume a W&B sweep; each run performs its final OOD suite."""

import argparse
import json
import os
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'configs' / 'sam_sweep.yaml'))
    parser.add_argument('--sweep-id', default=None, help='Resume an existing sweep instead of creating one.')
    parser.add_argument('--project', default=os.environ.get('WANDB_PROJECT', 'wise-ft'))
    parser.add_argument('--entity', default=os.environ.get('WANDB_ENTITY'))
    parser.add_argument('--count', type=int, default=None,
                        help='Optional maximum runs for this agent; default: no agent limit.')
    parser.add_argument('--dry-run', action='store_true', help='Show configuration without contacting W&B.')
    args = parser.parse_args()
    if args.count is not None and args.count < 1:
        parser.error('--count must be positive')
    try:
        with Path(args.config).expanduser().open() as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict) or not isinstance(config.get('parameters'), dict):
            raise ValueError('Expected a sweep mapping with parameters')
    except (OSError, yaml.YAMLError, ValueError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(dict(config=config, sweep_id=args.sweep_id, project=args.project,
                              entity=args.entity, max_runs=args.count), indent=2))
        return
    import wandb
    os.chdir(ROOT)
    # Training subprocesses use the agent's interpreter unless explicitly configured.
    os.environ.setdefault('PYTHON', sys.executable)
    os.environ['WANDB_MODE'] = 'online'
    os.environ['WANDB_PROJECT'] = args.project
    if args.entity:
        os.environ['WANDB_ENTITY'] = args.entity
    sweep_id = args.sweep_id or wandb.sweep(config, project=args.project, entity=args.entity)
    print('Sweep:', sweep_id, flush=True)
    wandb.agent(sweep_id, project=args.project, entity=args.entity, count=args.count)


if __name__ == '__main__':
    main()
