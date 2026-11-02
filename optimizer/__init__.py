"""Perturbation policies, independent of the optimizer that updates weights."""

from .asam import ASAM
from .condsam import CondSAM
from .fishersam import FisherSAM
from .friendlysam import FriendlySAM
from .mongesam import MongeSAM
from .sam import SAM
from .training import apply_perturbations, freeze_batchnorm_statistics, \
    select_perturbation_parameters

POLICIES = {
    'sam': SAM,
    'asam': ASAM,
    'condsam': CondSAM,
    'friendlysam': FriendlySAM,
    'fishersam': FisherSAM,
    'mongesam': MongeSAM,
}

__all__ = ['SAM', 'ASAM', 'CondSAM', 'FriendlySAM', 'FisherSAM', 'MongeSAM', 'POLICIES',
           'apply_perturbations', 'freeze_batchnorm_statistics',
           'select_perturbation_parameters']
