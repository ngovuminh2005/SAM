import os
import torch

from utility.args import Args

Args.add_argument("--saveCheckpoint", type=bool, help="save model after each epoch (delete after next epoch)")
Args.add_argument("--keepLastCheckpoint", type=bool, help="keep checkpoint after last epoch is done")
Args.add_argument("--checkpointsList", type=int, nargs = "*", help="keep checkpoint at specific epochs")
Args.add_argument("--saveCheckpointInterval", type=int, help="save model between interval")


class ModelSaver():
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            self.model = model.module
        else:
            self.model = model

        self.optimizer = optimizer
        self.dir = os.path.join(Args.logDir, Args.logSubDir)

    def __call__(self, epoch):
        if torch.distributed.get_rank() == 0:
            should_save_checkpoint = (
                epoch in map(int, Args.checkpointsList) or
                Args.saveCheckpoint or
                Args.saveCheckpointInterval > 0 or
                epoch == Args.epochs
            )

            if should_save_checkpoint:
                state = {
                    'epoch': epoch,
                    'modelState': self.model.state_dict(),
                    'optimizerState': self.optimizer.state_dict(),
                }

                if epoch in map(int, Args.checkpointsList):
                    torch.save(state, os.path.join(self.dir, f"epoch_{epoch}.model"))
                if Args.saveCheckpointInterval > 0 and epoch % Args.saveCheckpointInterval == 0:
                    torch.save(state, os.path.join(self.dir, f"epoch_{epoch}.model"))
                if Args.saveCheckpoint:
                    checkpoint_path = os.path.join(self.dir, "checkpoint.model")
                    if epoch == Args.epochs and not Args.keepLastCheckpoint:
                        if os.path.isfile(checkpoint_path):
                            os.remove(checkpoint_path)
                    else:
                        torch.save(state, checkpoint_path)

                if epoch == Args.epochs:
                    torch.save(state, os.path.join(self.dir, "final.model"))

    def loadLast(self):
        lastEpoch = -1
        for file in os.listdir(self.dir):
            if file.startswith("epoch_") and file.endswith(".model"):
                epoch = int(file[6:-6])
                lastEpoch = epoch if epoch > lastEpoch else lastEpoch
        if lastEpoch == -1:
            raise RuntimeError(f"No epoch checkpoint found in dir: {self.dir}")

        return self.loadModel(filename = f"epoch_{lastEpoch}.model")


    def loadModel(self, filename):
        #@TODO save and load rnd state
        state = torch.load(os.path.join(self.dir, filename),  map_location='cpu')

        print(f"Loading from epoch {state['epoch']} (File: {filename}).")

        self.model.load_state_dict(state['modelState'])
        self.optimizer.load_state_dict(state['optimizerState'])

        return state['epoch']
