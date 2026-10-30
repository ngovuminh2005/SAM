import configparser
import os
import argparse as ar
from typing import Any

# Repository root (directory containing main.py); used to resolve relative logDir paths.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class MetaClass(type):
    def __getattr__(cls, key: str) -> Any:
        return cls.get(key)

class Args(metaclass=MetaClass):
    """
    Argument parsing used from default argpase module. Changes:
        - defaults are not allowed in source code
        - a default value must be provided for each argument in the input file
        - bool args are handeled as follows:
            -> --arg --> True
            -> --arg True --> True
            -> --arg False --> False
            -> (not given) --> default from ini file
    """
    data: ar.Namespace = None
    parser = ar.ArgumentParser()
    
    @classmethod
    def get(cls, key: str) -> Any:
        if cls.data is None:
            raise RuntimeError(f"Arguments were not parsed yet.")
        return getattr(cls.data, key)
    
    @classmethod
    def add_argument(cls, *args, **kwargs) -> None:
        if "default" in kwargs:
            raise RuntimeError("Defining defaults in source code is prohibited. Use input file instead.")
        if "type" in kwargs:
            if kwargs["type"] is bool:
                kwargs["nargs"] = "?"
                kwargs["const"] = True
        cls.parser.add_argument(*args, **kwargs)
    
    @classmethod
    def parse_args(cls) -> None:
        #only read input file name
        init_file_name_parser = ar.ArgumentParser(description="Blank Project",add_help=False)
        init_file_name_parser.add_argument("-i", "--ifile", help="input parameter file", default='configs/default.ini',metavar="FILE")
        args, remaining_argv = init_file_name_parser.parse_known_args()
        
        config = configparser.ConfigParser()
        config.optionxform = str # make config parse case sensitive
        out = config.read([args.ifile])
        if not out:
            raise RuntimeError(f"Input file '{args.ifile}' not found.")

        defaults = {}
        for sec in config.sections():
            defaults.update(**dict(config[sec]))

        default_keys = list(defaults.keys())
        for action in cls.parser._actions:
            if not isinstance(action, ar._HelpAction):
                if action.dest not in defaults:
                    raise RuntimeError(f"No default argument given for {action.dest}.")
                # necessary to convert strings to bool
                if action.type is bool:
                    action.type = make_bool
                # split lists seperated by whitespace in defaults file
                if action.nargs == "*":
                    values = defaults[action.dest].split()
                    defaults[action.dest] = [action.type(value) if action.type else value for value in values]

                default_keys.remove(action.dest)

        if default_keys:
            raise RuntimeError(f"Parameter(s) {default_keys} were given in input file but not defined before.")

        cls.parser.set_defaults(**defaults)
        cls.data = cls.parser.parse_args(remaining_argv)
        cls._normalize_log_dir_paths()
        cls._normalize_glue_output_root()
        cls._normalize_data_dir_path()
        cls._apply_dataset_root_env()

    @classmethod
    def _normalize_log_dir_paths(cls) -> None:
        """Resolve relative logDir to an absolute path under the repo root (portable across machines)."""
        if cls.data is None or not getattr(cls.data, "logDir", None):
            return
        if not os.path.isabs(cls.data.logDir):
            cls.data.logDir = os.path.normpath(os.path.join(_REPO_ROOT, cls.data.logDir))

    @classmethod
    def _normalize_glue_output_root(cls) -> None:
        """Resolve relative glue_output_root under repo root (NLP entrypoints only register this arg)."""
        if cls.data is None or not getattr(cls.data, "glue_output_root", None):
            return
        p = os.path.expanduser(str(cls.data.glue_output_root))
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(_REPO_ROOT, p))
        cls.data.glue_output_root = p

    @classmethod
    def _normalize_data_dir_path(cls) -> None:
        """Expand ~ and resolve relative dataDir under the repo root (same portability as logDir)."""
        if cls.data is None or not getattr(cls.data, "dataDir", None):
            return
        d = os.path.expanduser(str(cls.data.dataDir))
        if not os.path.isabs(d):
            d = os.path.normpath(os.path.join(_REPO_ROOT, d))
        cls.data.dataDir = d

    @classmethod
    def _apply_dataset_root_env(cls) -> None:
        """SAM_DATASET_ROOT / SAM_VARIANT_DATA_ROOT overrides dataDir (cluster/shared datasets)."""
        if cls.data is None:
            return
        env = os.environ.get("SAM_DATASET_ROOT") or os.environ.get("SAM_VARIANT_DATA_ROOT")
        if env:
            cls.data.dataDir = os.path.normpath(os.path.expanduser(env))

    @classmethod
    def parse_args_contin(cls, defaults: dict) -> None:
        defaults = dict(defaults)
        for key, value in {"condsamRho": -1.0, "condsamBeta": 0.95, "condsamEps": 1e-8, "condsamPowerIters": 1, "evalEvery": 20, "topK": []}.items():
            defaults.setdefault(key, value)
        defaults["contin"] = True #make sure contin argument is not overwritten by defaults from param.json file
        cls.parser.set_defaults(**defaults) #read defaults from param.json file
        
        # parse ifile argument here only to not throw an error by real parser
        init_file_name_parser = ar.ArgumentParser(description="",add_help=False)
        init_file_name_parser.add_argument("-i", "--ifile", help="input parameter file", default='input.ini',metavar="FILE")
        _, remaining_argv = init_file_name_parser.parse_known_args()
        
        cls.data = cls.parser.parse_args(remaining_argv)
        cls._normalize_log_dir_paths()
        cls._normalize_glue_output_root()
        cls._normalize_data_dir_path()
        cls._apply_dataset_root_env()


def make_bool(arg: Any) -> bool:
    if isinstance(arg, str):
        if arg.lower() in ['true', '1', 't', 'y', 'yes']:
            return True
        elif arg.lower() in ['false', '0', 'f', 'n', 'no']:
            return False
        else:
            raise RuntimeError(f"Could not convert string to bool: {arg}")
    else:
        return bool(arg)
