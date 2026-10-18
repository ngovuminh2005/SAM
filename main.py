import os
import torch
import json
import time

from models import getModel
from optimizer import getOptimizer
from utility.loss import smooth_crossentropy
from utility.data import DataLoader
from utility.dataLogger import DataLogger
from utility.utils import initialize
from utility.LRScheduler import getLRScheduler, _LRScheduler
from utility.rhoScheduler import getRhoScheduler, _rhoScheduler
from utility.modelSaver import ModelSaver
from utility.args import Args
from utility import tracking
from utility.evaluation import should_evaluate


"""
run:
    python -m torch.distributed.run main.py
"""

Args.add_argument("--logDir", type=str, help="main directory to store logs")
Args.add_argument("--logSubDir", type=str, help="subdir in logDir to store logs for this run")
Args.add_argument("--evalEvery", type=int, help="Evaluate every N epochs and at the final epoch.")
Args.add_argument("--epochs", type=int, help="Total number of epochs")
Args.add_argument("--contin", type=bool, help="Whether to continue from checkpoint. In continue mode parameters are read from params.json file, input file is ignored.")


def _raise_cuda_configuration_error(message: str):
    raise RuntimeError(
        f"{message}\n"
        f"Detected environment:\n"
        f"  torch.__version__ = {torch.__version__}\n"
        f"  torch.version.cuda = {torch.version.cuda}\n"
        f"  torch.cuda.is_available() = {torch.cuda.is_available()}\n"
        f"  torch.cuda.device_count() = {torch.cuda.device_count()}\n"
        "This training entrypoint is GPU-only (NCCL) and does not support CPU fallback.\n"
        "Install compatible wheels for Driver 535/CUDA 12.2:\n"
        "  pip install --index-url https://download.pytorch.org/whl/cu124 "
        "torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1"
    )


def _format_run_value(value) -> str:
    if isinstance(value, str):
        return value.replace(" ", "")
    return f"{float(value):.6g}"


def _resolve_optimizer_rho() -> str:
    optimizer_rho = {
        "SAM": Args.samRho,
        "CondSAM": Args.condsamRho,
        "OrthoSAM": Args.samRho,
        "ASAM": Args.asamRho,
        "ESAM": Args.esamRho,
        "lookSAM": Args.looksamRho,
        "FriendlySAM": Args.friendlysamRho,
        "FisherSAM": Args.fishersamRho,
        "MongeSAM": Args.mongesamRho,
        "GSAM": Args.gsamRho,
        "BayesianSAM": Args.bayesiansamRho,
        "MSAM": Args.msamRho,
        "AdamW_SAM": Args.adamwSamRho,
        "Adam_SAM": Args.adamSamRho,
        "AdamW_MSAM": Args.adamwMsamRho,
        "AdamW_FriendlySAM": Args.adamwFriendlysamRho,
        "Adam_FriendlySAM": Args.adamwFriendlysamRho,
    }

    if Args.optimizer not in optimizer_rho:
        return "na"

    rho_value = optimizer_rho[Args.optimizer]
    if rho_value < 0:
        rho_value = Args.rho
    return _format_run_value(rho_value)


def _resolve_model_tag() -> str:
    model_name = _format_run_value(Args.model)
    # Distinguish WRN variants in auto-generated run directories.
    if Args.model == "WRN":
        depth_tag = _format_run_value(Args.depth)
        width_tag = _format_run_value(Args.widthFactor)
        return f"{model_name}{depth_tag}_{width_tag}"
    return model_name


def _resolve_log_subdir() -> str:
    dataset_tag = _format_run_value(Args.dataset)
    model_tag = _resolve_model_tag()
    base_subdir = f"{dataset_tag}_{Args.optimizer}_{model_tag}_rho{_resolve_optimizer_rho()}_lr{_format_run_value(Args.learningRate)}"
    if Args.truncate:
        return base_subdir

    candidate = base_subdir
    run_idx = 1
    while os.path.exists(os.path.join(Args.logDir, candidate)):
        candidate = f"{base_subdir}_run{run_idx:02d}"
        run_idx += 1
    return candidate


def _format_seconds(total_seconds: float) -> str:
    total_seconds = max(0.0, float(total_seconds))
    hh = int(total_seconds // 3600)
    mm = int((total_seconds % 3600) // 60)
    ss = total_seconds - (hh * 3600 + mm * 60)
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"

def main():
    Args.parse_args()
    console_eval_only = os.environ.get("SAM_CONSOLE_EVAL_ONLY", "0") == "1"
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if not Args.contin:
        Args.data.logSubDir = _resolve_log_subdir()

    logDir = os.path.join(Args.logDir, Args.logSubDir)

    if Args.contin:
        with open(os.path.join(logDir, "params.json"), "r") as file:
            parameters = json.load(file)
        Args.parse_args_contin(parameters)
        logDir = os.path.join(Args.logDir, Args.logSubDir)

    if Args.evalEvery < 1:
        raise ValueError("evalEvery must be positive")

    rank = int(os.environ["SLURM_PROCID"]) if "SLURM_PROCID" in os.environ else int(os.environ.get("RANK", "0"))

    if os.name == "nt":
        _raise_cuda_configuration_error("Windows runtime is not supported for this NCCL training entrypoint.")

    if not torch.cuda.is_available():
        _raise_cuda_configuration_error("CUDA runtime is unavailable or incompatible with the installed PyTorch build.")

    if torch.cuda.device_count() == 0:
        _raise_cuda_configuration_error("No CUDA device found.")

    backend = "nccl"
    torch.distributed.init_process_group(backend=backend, init_method="env://", rank=rank)

    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % torch.cuda.device_count())))
    try:
        torch.cuda.set_device(local_rank)
    except RuntimeError as error:
        _raise_cuda_configuration_error(f"Failed to set CUDA device for LOCAL_RANK={local_rank}. Original error: {error}")

    run_seed = initialize() # set up seed and cudnn

    if torch.distributed.get_rank() == 0:
        os.makedirs(logDir, exist_ok=True)
        print(f"[RUN_DIR] {logDir}")
        with open(os.path.join(logDir, "params.json"), "w") as file:
            json.dump(vars(Args.data), file, indent = 4)

    tracking.start({**vars(Args.data), "actual_seed": run_seed}, logDir, job_type="vision")
    dataLogger = DataLogger()
    dataset = DataLoader()

    model = getModel()(num_classes=dataset.numClasses)
    model = model.cuda(local_rank)
    # Eager CNNs otherwise repeatedly convert NCHW tensors inside cuDNN.
    # Keep CondSAM's hooks/eager execution while using cuDNN's native layout.
    condsam_channels_last = Args.optimizer == "CondSAM" and any(
        isinstance(module, torch.nn.Conv2d) for module in model.modules())
    if condsam_channels_last:
        model = model.to(memory_format=torch.channels_last)
    # torch.compile + inductor can fail on SimpleViT (sincos PE / einops), e.g. sympy "Invalid NaN comparison".
    if hasattr(torch, "compile") and Args.model != "ViT" and Args.optimizer != "CondSAM":
        try:
            model = torch.compile(model)
        except RuntimeError as error:
            if torch.distributed.get_rank() == 0:
                print(f"[WARN] torch.compile disabled due to runtime error: {error}")
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model)


    train_set_size = len(dataset.train.dataset) if hasattr(dataset.train, "dataset") else 0
    if Args.optimizer == "OrthoSAM":
        optimizer = getOptimizer(model.named_parameters(), n_data=train_set_size)
    else:
        optimizer = getOptimizer(model.parameters(), n_data=train_set_size)
    lrScheduler: _LRScheduler = getLRScheduler(optimizer)
    rhoScheduler: _rhoScheduler = getRhoScheduler(optimizer)
    modelSaver = ModelSaver(model = model, optimizer = optimizer)
    
    startEpoch = 1
    if Args.contin:
        startEpoch = modelSaver.loadModel("checkpoint.model")
        startEpoch += 1
        model = model.cuda(local_rank)
        if startEpoch > Args.epochs:
            raise RuntimeError(f"Can't continue model from epoch {startEpoch} to max epoch {Args.epochs}.")
    else:
        modelSaver(0)

    torch.distributed.barrier() #wait until all workers are done with initialization
    dataLogger.epoch = startEpoch - 1
    dataLogger.printHeader()
    state = {
        "model": model,
        "lrScheduler": lrScheduler,
        "optimizer": optimizer,
    }
    epoch_train_times_sec = []
    total_train_time_sec = 0.0
    for epoch in range(startEpoch, Args.epochs+1):
        eval_epoch = should_evaluate(epoch, Args.epochs, Args.evalEvery)
        dataset.train.sampler.set_epoch(epoch)

        model.train()
        numBatches = len(dataset.train)
        dataLogger.startTrain(trainDataLen = numBatches)
        torch.cuda.synchronize(local_rank)
        epoch_train_start = time.perf_counter()

        if Args.optimizer in ["MSAM","AdamW_MSAM"]:
            optimizer.move_up_to_momentumAscent()
        for i, batch in enumerate(dataset.train):
            lrScheduler.step(epoch-1, (i+1)/numBatches)
            rhoScheduler.step(epoch-1, (i+1)/numBatches)

            inputs, targets = (b.cuda(local_rank) for b in batch)
            if condsam_channels_last and inputs.ndim == 4:
                inputs = inputs.contiguous(memory_format=torch.channels_last)

            # SAM needs to calculate second forward/backward paths, so target, inputs, and model have to be passed to optimizer. 
            # we are not doing this for MSAM so that MSAM can be used as a drop in replacement for SGD/AdamW.
            if Args.optimizer in ["SAM", "CondSAM", "OrthoSAM", "ASAM", "FisherSAM", "MongeSAM", "GSAM", "BayesianSAM", "AdamW_SAM", "Adam_SAM", "AdamW_FriendlySAM", "Adam_FriendlySAM", "ESAM", "lookSAM", "FriendlySAM"]:
                loss, predictions = optimizer.step(model, inputs, targets)
            else:
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                loss.mean().backward()
                if Args.grad_clip_norm != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), Args.grad_clip_norm)
                optimizer.step()

            optimizer.zero_grad()

            with torch.no_grad():
                state["loss"] = loss
                state["predictions"] = predictions
                state["targets"] = targets
                dataLogger(state)

        dataLogger.flush()
        torch.cuda.synchronize(local_rank)
        epoch_train_elapsed = time.perf_counter() - epoch_train_start
        tracking.log({"epoch": epoch, "time/train_seconds": epoch_train_elapsed,
                      "optimizer/rho": optimizer.param_groups[0].get("rho", 0),
                      "optimizer/lr": optimizer.param_groups[0]["lr"]})
        total_train_time_sec += epoch_train_elapsed
        epoch_train_times_sec.append(epoch_train_elapsed)
        if torch.distributed.get_rank() == 0 and (not console_eval_only or eval_epoch):
            print(f"[TIME] epoch {epoch} train_time={epoch_train_elapsed:.3f}s")
        

        if Args.optimizer in ["MSAM","AdamW_MSAM"]:
            optimizer.move_back_from_momentumAscent()
        recalibrate_final = (epoch == Args.epochs and Args.optimizer in ["MSAM", "AdamW_MSAM"] and Args.model != "ViT")
        if eval_epoch and not recalibrate_final:
            dataLogger.startTest()
            model.eval()
            with torch.no_grad():
                for batch in dataset.test:
                    inputs, targets = (b.cuda(local_rank) for b in batch)
                    if condsam_channels_last and inputs.ndim == 4:
                        inputs = inputs.contiguous(memory_format=torch.channels_last)

                    predictions = model(inputs)
                    loss = smooth_crossentropy(predictions, targets)
                    state["loss"] = loss
                    state["predictions"] = predictions
                    state["targets"] = targets
                    dataLogger(state)

                dataLogger.flush()
        modelSaver(epoch)

    if torch.distributed.get_rank() == 0:
        avg_train_time_sec = total_train_time_sec / max(1, len(epoch_train_times_sec))
        with open(os.path.join(logDir, "training_time.txt"), "w", encoding="utf-8") as f:
            f.write(f"dataset: {Args.dataset}\n")
            f.write(f"model: {Args.model}\n")
            f.write(f"optimizer: {Args.optimizer}\n")
            f.write(f"epochs_trained_this_run: {len(epoch_train_times_sec)}\n")
            f.write(f"total_train_time_seconds: {total_train_time_sec:.6f}\n")
            f.write(f"total_train_time_hhmmss: {_format_seconds(total_train_time_sec)}\n")
            f.write(f"avg_train_time_per_epoch_seconds: {avg_train_time_sec:.6f}\n")
            f.write(f"avg_train_time_per_epoch_hhmmss: {_format_seconds(avg_train_time_sec)}\n")
            f.write("epoch_train_times_seconds: " + ", ".join(f"{x:.6f}" for x in epoch_train_times_sec) + "\n")
        print(f"[TIME] wrote training_time.txt -> {os.path.join(logDir, 'training_time.txt')}")

    # run extra epoch (forward path only) to calculate batch norm statistics (not needed for ViT since no BN-layers are used)
    # probably a few iterations would be sufficient until BN-statistics converge, but we didnt test this
    if Args.optimizer in ["MSAM","AdamW_MSAM"] and Args.model != "ViT":
        with torch.no_grad():
            model.train()
            numBatches = len(dataset.train)
            dataLogger.epoch = Args.epochs - 1
            dataLogger.startTrain(trainDataLen = numBatches)

            for i, batch in enumerate(dataset.train):
                inputs, targets = (b.cuda(local_rank) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                state["loss"] = loss
                state["predictions"] = predictions
                state["targets"] = targets
                dataLogger(state)
        
            dataLogger.flush()
            dataLogger.startTest()
            model.eval()
            for batch in dataset.test:
                inputs, targets = (b.cuda(local_rank) for b in batch)
                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                state["loss"] = loss
                state["predictions"] = predictions
                state["targets"] = targets
                dataLogger(state)

            dataLogger.flush()
            modelSaver(epoch)


if __name__ == "__main__":
    tracking.execute(main)
