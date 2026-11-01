"""Read perturbation defaults without coupling policies to configuration files."""

import configparser
import math
from pathlib import Path


DEFAULT_SAM_CONFIG = Path(__file__).resolve().parents[1] / 'optimizer' / 'defaults.ini'

# All radii are declared in [SAM]; only the selected policy's radius is applied.
_RADII = {name + 'Rho': name for name in
          ('sam', 'asam', 'condsam', 'friendlysam', 'fishersam', 'mongesam')}

# INI section/key -> CLI destination and value type.
_OPTIONS = {
    'SAM': {key: ('sam_rho', float) for key in _RADII},
    'FriendlySAM': {'sigma': ('friendlysam_sigma', float),
                    'lmbda': ('friendlysam_lambda', float)},
    'FisherSAM': {'eta': ('fishersam_eta', float)},
    'CondSAM': {'beta': ('condsam_beta', float),
                'eps': ('condsam_eps', float),
                'power_iters': ('condsam_power_iters', int)},
}


def load_sam_defaults(path, method='none'):
    """Return only configured defaults; reject typos and unused legacy options."""
    config = configparser.ConfigParser(interpolation=None)
    config.optionxform = str
    try:
        with Path(path).expanduser().open() as stream:
            config.read_file(stream)
    except (OSError, configparser.Error) as error:
        raise ValueError(f'Cannot read SAM config {path}: {error}') from error
    if config.defaults():
        raise ValueError('Use explicit SAM sections instead of [DEFAULT]')
    defaults = {}
    for section in config.sections():
        if section not in _OPTIONS:
            raise ValueError(f'Unknown SAM config section [{section}]')
        for key, raw in config.items(section):
            if key not in _OPTIONS[section]:
                raise ValueError(f'Unknown SAM option [{section}] {key}')
            target, convert = _OPTIONS[section][key]
            try:
                value = convert(raw)
            except ValueError as error:
                raise ValueError(f'Invalid [{section}] {key}: {raw}') from error
            if not math.isfinite(value):
                raise ValueError(f'[{section}] {key} must be finite')
            if key in ('beta', 'lmbda'):
                valid = 0 <= value < 1
                expected = 'in [0, 1)'
            elif key in ('eps', 'power_iters'):
                valid = value > 0
                expected = 'positive'
            else:
                valid = value >= 0
                expected = 'nonnegative'
            if not valid:
                raise ValueError(f'[{section}] {key} must be {expected}')
            if section != 'SAM' or _RADII[key] == method.lower():
                defaults[target] = value
    return defaults
