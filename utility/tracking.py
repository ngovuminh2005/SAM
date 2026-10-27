"""Shared W&B tracking for vision, GLUE and Hessian entrypoints."""
import os
from pathlib import Path
import torch.distributed as dist

_run = None


def start(config, directory, name=None, job_type='train'):
    global _run
    if dist.is_initialized() and dist.get_rank() != 0:
        return None
    base = Path(directory).resolve()
    os.environ.setdefault("WANDB_CACHE_DIR", str(base / "wandb-cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(base / "wandb-config"))
    import wandb
    mode = os.environ.get('WANDB_MODE', 'online')
    if mode == 'online' and not wandb.login(prompt=False, verify=False):
        mode = 'offline'
        print('[W&B] No login found; saving offline. Run env/bin/wandb login to enable online tracking.')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    configuration = dict(config)
    configuration.update(label_noise_rate=float(os.environ.get('LABEL_NOISE_RATE', '0')),
                         label_noise_seed=int(os.environ.get('LABEL_NOISE_SEED', '12345')))
    _run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'CondSAM'),
                      name=os.environ.get('WANDB_NAME') or name or directory.name,
                      group=os.environ.get('WANDB_RUN_GROUP'),
                      job_type=job_type, config=configuration,
                      dir=str(directory), mode=mode,
                      settings=wandb.Settings(init_timeout=30))
    if _run is not None:
        _run.define_metric("epoch")
        _run.define_metric("train_step")
        _run.define_metric("cs/*", step_metric="train_step")
        for prefix in ("train", "test", "validation", "optimizer", "time"):
            _run.define_metric(prefix + "/*", step_metric="epoch")
    return _run


def log(values):
    if _run is not None:
        _run.log(values)


def summary(values):
    if _run is not None:
        _run.summary.update(values)


def execute(main):
    """Finish an active run with the correct exit status even on exceptions."""
    status = 0
    try:
        return main()
    except BaseException:
        status = 1
        raise
    finally:
        global _run
        if _run is not None:
            _run.finish(exit_code=status)
            _run = None
