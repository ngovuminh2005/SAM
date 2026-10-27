import torch

from .sgd import SGD
from .sam import SAM
from .condsam import CondSAM
from .condsam1 import CondSam1
from .condsam2 import CondSam2
from .condsam_adam import CondSAM_adam
from .asam import ASAM
from .msam import MSAM
from .esam import ESAM
from .adamW import AdamW
from .adamW_msam import AdamW_MSAM
from .adamW_sam import AdamW_SAM
from .adam_sam import Adam_SAM
from .adamW_friendlysam import AdamW_FriendlySAM
from .adam_friendlysam import Adam_FriendlySAM
from .looksam import LookSAM
from .friendlysam import FriendlySAM
from .fishersam import FisherSAM
from .mongesam import MongeSAM
from .gsam import GSAM
from .bayesiansam import BayesianSAM
from utility.args import Args

Args.add_argument("--optimizer", type=str, help="optimizer name")
Args.add_argument("--weightDecay", type=float, help="L2 weight decay.")
Args.add_argument("--momentum", type=float, help="Momentum.")
Args.add_argument("--rho", type=float, help="Global rho fallback used when method-specific rho is disabled.")
Args.add_argument("--condsamRho", type=float, help="CondSAM rho; negative falls back to rho.")
Args.add_argument("--condsamBeta", type=float, help="CondSAM factor EMA decay.")
Args.add_argument("--condsamEps", type=float, help="CondSAM numerical stabilizer.")
Args.add_argument("--condsamPowerIters", type=int, help="Positive number of power iterations per factor per step.")
Args.add_argument("--samRho", type=float, help="rho for SAM. Use negative value to fall back to --rho.")
Args.add_argument("--asamRho", type=float, help="rho for ASAM. Use negative value to fall back to --rho.")
Args.add_argument("--esamRho", type=float, help="rho for ESAM. Use negative value to fall back to --rho.")
Args.add_argument("--looksamRho", type=float, help="rho for lookSAM. Use negative value to fall back to --rho.")
Args.add_argument("--friendlysamRho", type=float, help="rho for FriendlySAM. Use negative value to fall back to --rho.")
Args.add_argument("--fishersamRho", type=float, help="rho for FisherSAM. Use negative value to fall back to --rho.")
Args.add_argument("--mongesamRho", type=float, help="rho for MongeSAM. Use negative value to fall back to --rho.")
Args.add_argument("--gsamRho", type=float, help="rho for GSAM. Use negative value to fall back to --rho.")
Args.add_argument("--bayesiansamRho", type=float, help="rho for BayesianSAM. Use negative value to fall back to --rho.")
Args.add_argument("--msamRho", type=float, help="rho for MSAM. Use negative value to fall back to --rho.")
Args.add_argument("--adamwSamRho", type=float, help="rho for AdamW_SAM. Use negative value to fall back to --rho.")
Args.add_argument("--adamSamRho", type=float, help="rho for Adam_SAM. Use negative value to fall back to --rho.")
Args.add_argument("--adamwMsamRho", type=float, help="rho for AdamW_MSAM. Use negative value to fall back to --rho.")
Args.add_argument("--adamwFriendlysamRho", type=float, help="rho for AdamW_FriendlySAM. Use negative value to fall back to --rho.")
Args.add_argument("--sigma", type=float, help="FriendlySAM stochastic-gradient-noise scaling.")
Args.add_argument("--lmbda", type=float, help="FriendlySAM momentum decay factor.")
Args.add_argument("--fisherEta", type=float, help="FisherSAM anti-regularizer eta for inverse Fisher approximation.")
Args.add_argument("--gsamAlpha", type=float, help="GSAM gradient decomposition coefficient.")
Args.add_argument("--gsamAdaptive", type=bool, help="Use GSAM adaptive perturbation scaling.")
Args.add_argument("--bayesMSharpness", type=int, help="BayesianSAM number of noise-perturbation splits per batch.")
Args.add_argument("--bayesDelta", type=float, help="BayesianSAM prior regularization coefficient.")
Args.add_argument("--bayesBeta1", type=float, help="BayesianSAM first moment decay.")
Args.add_argument("--bayesBeta2", type=float, help="BayesianSAM second moment decay.")
Args.add_argument("--bayesSInit", type=float, help="BayesianSAM initial scale state.")
Args.add_argument("--bayesDamping", type=float, help="BayesianSAM damping constant.")
Args.add_argument("--bayesEps", type=float, help="BayesianSAM numerical epsilon.")
Args.add_argument("--nesterov", type=bool, help="use normal nesterov momentum for sgd/sam")

Args.add_argument("--grad_clip_norm", type=float, help="")


def _resolve_rho(method_rho: float) -> float:
    return method_rho if method_rho >= 0 else Args.rho


def getOptimizer(params, n_data: int = 0) -> torch.nn.Module:
    # optimizer names and additional args which will be passed from Args.XX
    optimizerDict = {
        "SGD":        (SGD,        {"momentum": Args.momentum, "nesterov": Args.nesterov}),
        "CondSAM": (CondSAM, {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": _resolve_rho(Args.condsamRho), "beta": Args.condsamBeta, "eps": Args.condsamEps, "power_iters": Args.condsamPowerIters}),
        "CondSam1": (CondSam1, {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": _resolve_rho(Args.condsamRho), "beta": Args.condsamBeta, "eps": Args.condsamEps, "power_iters": Args.condsamPowerIters}),
        "CondSam2": (CondSam2, {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": _resolve_rho(Args.condsamRho), "beta": Args.condsamBeta, "eps": Args.condsamEps, "power_iters": Args.condsamPowerIters}),
        "CondSAM_adam": (CondSAM_adam, {"rho": _resolve_rho(Args.condsamRho), "beta": Args.condsamBeta, "eps": Args.condsamEps, "power_iters": Args.condsamPowerIters}),
        "SAM":        (SAM,        {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": _resolve_rho(Args.samRho)}),
        "ASAM":       (ASAM,       {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": _resolve_rho(Args.asamRho)}),
        "FisherSAM":  (FisherSAM,  {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": _resolve_rho(Args.fishersamRho), "eta": Args.fisherEta}),
        "MongeSAM":   (MongeSAM,   {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": _resolve_rho(Args.mongesamRho)}),
        "GSAM":       (GSAM,       {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": _resolve_rho(Args.gsamRho), "alpha": Args.gsamAlpha, "adaptive": Args.gsamAdaptive}),
        "BayesianSAM":(BayesianSAM,{"rho": _resolve_rho(Args.bayesiansamRho), "msharpness": Args.bayesMSharpness, "delta": Args.bayesDelta, "beta1": Args.bayesBeta1, "beta2": Args.bayesBeta2, "s_init": Args.bayesSInit, "damping": Args.bayesDamping, "eps": Args.bayesEps, "n_data": n_data}),
        "ESAM":       (ESAM,       {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": _resolve_rho(Args.esamRho), "beta": Args.ESAMbeta, "gamma": Args.ESAMgamma}),
        "lookSAM":    (LookSAM,    {"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": _resolve_rho(Args.looksamRho), "alpha": Args.lookSAMalpha, "k": Args.lookSAMk, "layerwise": Args.lookSAMLayerwise}),
        "FriendlySAM":(FriendlySAM,{"momentum": Args.momentum, "nesterov": Args.nesterov, "rho": _resolve_rho(Args.friendlysamRho), "sigma": Args.sigma, "lmbda": Args.lmbda}),
        "MSAM":       (MSAM,       {"momentum": Args.momentum, "rho": _resolve_rho(Args.msamRho)}),
        # PyTorch Adam (L2-style weight decay), matches Tiny-ViT-from-Scratch / many ViT-from-scratch CIFAR recipes.
        "Adam":       (torch.optim.Adam, {}),
        "AdamW":      (AdamW,      {}),
        "AdamW_MSAM": (AdamW_MSAM, {"rho": _resolve_rho(Args.adamwMsamRho)}),
        "AdamW_SAM":  (AdamW_SAM,  {"rho": _resolve_rho(Args.adamwSamRho)}),
        "Adam_SAM":   (Adam_SAM,   {"rho": _resolve_rho(Args.adamSamRho)}),
        "AdamW_FriendlySAM": (AdamW_FriendlySAM, {"rho": _resolve_rho(Args.adamwFriendlysamRho), "sigma": Args.sigma, "lmbda": Args.lmbda}),
        "Adam_FriendlySAM": (Adam_FriendlySAM, {"rho": _resolve_rho(Args.adamwFriendlysamRho), "sigma": Args.sigma, "lmbda": Args.lmbda}),
    }
    
    if Args.optimizer in optimizerDict:
        optimizer, additionalArgs = optimizerDict[Args.optimizer]
        return optimizer(params, lr = Args.learningRate, weight_decay=Args.weightDecay, **additionalArgs)
    else:
        raise RuntimeError(f"Optimizer '{Args.optimizer}' not found. Available optimizers: {', '.join(optimizerDict.keys())}")
