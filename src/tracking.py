"""One W&B run spanning fine-tuning and the complete final evaluation."""

from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path


_active_run = None


def parameter_run_name(args):
    """Name a run from effective CLI/INI values, with an optional user suffix."""
    def number(value):
        text = format(Decimal(str(value)), 'f')
        if '.' in text:
            text = text.rstrip('0').rstrip('.')
        return text.replace('-', 'm').replace('.', 'p')

    method = getattr(args, 'sam', 'none').lower()
    optimizer = getattr(args, 'optimizer', 'adamw').lower()
    parts = [method, optimizer]
    if method != 'none':
        parts.append('rho' + number(getattr(args, 'sam_rho', 0.05)))
    for label, key, default in [('lr', 'lr', 0.001), ('wd', 'wd', 0.1),
                                ('bs', 'batch_size', 128), ('ep', 'epochs', 10)]:
        parts.append(label + number(getattr(args, key, default)))
    extras = {
        'condsam': [('beta', 'condsam_beta', 0.95), ('eps', 'condsam_eps', 1e-8),
                    ('poweriters', 'condsam_power_iters', 1)],
        'friendlysam': [('sigma', 'friendlysam_sigma', 1.0), ('lambda', 'friendlysam_lambda', 0.95)],
        'fishersam': [('eta', 'fishersam_eta', 1.0)],
    }
    for label, key, default in extras.get(method, []):
        parts.append(label + number(getattr(args, key, default)))
    if optimizer == 'sgd':
        parts.append('mom' + number(getattr(args, 'momentum', 0.9)))
        if getattr(args, 'nesterov', False):
            parts.append('nesterov')
    if getattr(args, 'exp_name', None):
        parts.append(args.exp_name)
    return '-'.join(parts)


@contextmanager
def experiment_run(args, job_type='train'):
    global _active_run
    mode = getattr(args, 'wandb_mode', 'offline')
    if _active_run is not None or mode == 'disabled':
        yield _active_run
        return
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError('Install wandb or use --wandb-mode disabled') from error
    directory = Path(getattr(args, 'wandb_dir', None) or args.save or '.').expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project=getattr(args, 'wandb_project', 'wise-ft'),
                     entity=getattr(args, 'wandb_entity', None),
                     name=parameter_run_name(args),
                     group=getattr(args, 'wandb_group', None),
                     mode=mode, dir=str(directory), config=dict(vars(args)), job_type=job_type)
    _active_run = run
    try:
        run.define_metric('train/*', step_metric='train/step')
        run.define_metric('epoch_eval/*', step_metric='train/epoch')
        run.define_metric('wise_ft/*', step_metric='wise_ft/alpha')
        yield run
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)
    finally:
        _active_run = None


def log(values):
    if _active_run is not None:
        _active_run.log(values)


def log_final_results(settings, report_path):
    if _active_run is None:
        return
    import wandb
    rows = []
    for setting in settings:
        for key, value in setting['metrics'].items():
            dataset, metric = key.split(':', 1)
            rows.append([setting['alpha'], dataset, metric, value])
    _active_run.log({'final_evaluation': wandb.Table(
        columns=['alpha', 'dataset', 'metric', 'value'], data=rows)})
    _active_run.summary['evaluation/settings'] = len(settings)
    _active_run.summary['evaluation/report'] = str(report_path)
