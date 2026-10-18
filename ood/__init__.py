"""Post-training corruption robustness and OOD detection evaluation."""
from .base import AbstractOODTask, OODCase
from .settings import OODSetting
from .tasks import CorruptionTask, DetectionTask


def create_task(setting, checkpoint_params):
    from .data import evaluation_transform
    return {'corruption': CorruptionTask, 'detection': DetectionTask}[setting.kind](setting, evaluation_transform(setting.id_dataset, checkpoint_params))
