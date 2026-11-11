"""Perturbation policies, independent of the optimizer that updates weights."""

from .asam import ASAM
from .condsam import CondSAM
from .condsam1 import CondSAM1
from .condsam2 import CondSAM2
from .condsam3 import CondSAM3
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
    'condsam1': CondSAM1,
    'condsam2': CondSAM2,
    'condsam3': CondSAM3,
    'friendlysam': FriendlySAM,
    'fishersam': FisherSAM,
    'mongesam': MongeSAM,
}

__all__ = ['SAM', 'ASAM', 'CondSAM', 'CondSAM1', 'CondSAM2', 'CondSAM3', 'FriendlySAM', 'FisherSAM', 'MongeSAM', 'POLICIES',
           'apply_perturbations', 'freeze_batchnorm_statistics',
           'select_perturbation_parameters']
