import h5py
import time
import torch
import os
import numpy as np
from utility import tracking

from utility.args import Args
from utility.metrics import BaseMetric, MetricTopKAccuracy, available_metrics

Args.add_argument("--truncate", type=bool, help="truncate log file")
Args.add_argument("--verbose", type=bool, help="print to terminal")
Args.add_argument("--metrics", type=str, nargs = "*", help="list of metrics")
Args.add_argument("--logEach", type=int, help="Iterations to log during training.")

Args.add_argument("--topK", type=int, nargs="*", help="Additional accuracy metrics, e.g. --topK 1 5 10. Empty disables them.")

class DataLogger:
    available_metrics = {}
    def __init__(self):
        # Shell sweeps print epoch summaries only when evaluation occurs.
        # Collection, HDF5 writes and W&B logging still run every epoch.
        self.verbose = Args.verbose and os.environ.get("SAM_CONSOLE_EVAL_ONLY", "0") != "1"
        self.logEach = Args.logEach

        self.columnLen = 15
        self.state = {}
        self.step = 0
        self.epoch = 0
        self.train = True
        self.trainDataLen = None

        self.metrics: list[BaseMetric] = []
        self.printTrainMetrics: list[BaseMetric] = []
        self.printTestMetrics: list[BaseMetric] = []
        for metricName in Args.metrics:
            if metricName in available_metrics:
                self.metrics.append(available_metrics[metricName]())
                if self.metrics[-1].printTrain:
                    self.printTrainMetrics.append(self.metrics[-1])
                if self.metrics[-1].logTest:
                    self.printTestMetrics.append(self.metrics[-1])
            else:
                raise RuntimeError(f"Metric {metricName} not found. Available metrics: {', '.join(available_metrics.keys())}")

        for k in dict.fromkeys(Args.topK):
            metric = MetricTopKAccuracy(k)
            self.metrics.append(metric)
            self.printTrainMetrics.append(metric)
            self.printTestMetrics.append(metric)

        self.loading_bar = LoadingBar(length=(self.columnLen+1)*len(self.printTestMetrics)-3)

        self.filePath = os.path.join(Args.logDir, Args.logSubDir, "data.hdf5")
        self.terminalLogPath = os.path.join(Args.logDir, Args.logSubDir, "terminal.log")
        if torch.distributed.get_rank() == 0:
            if not Args.contin:
                try:
                    with h5py.File(self.filePath, "w" if Args.truncate else "w-") as f:
                        f.create_group("train")
                        f.create_group("test")
                        
                        for metric in self.metrics:
                            metric.createDatasets(f)
                except FileExistsError as e:
                    raise FileExistsError("To overwrite existing logfile use '--truncate' option. ") from e

                terminal_log_mode = "w" if Args.truncate else "x"
                with open(self.terminalLogPath, terminal_log_mode, encoding="utf-8") as _:
                    pass
            else:
                if not os.path.isfile(self.filePath):
                    raise RuntimeError(f"Running in continue mode but log file not found. Path: {self.filePath}")
                # Allow new top-k metrics when resuming an older run.
                with h5py.File(self.filePath, "r+") as file:
                    for metric in self.metrics:
                        if not isinstance(metric, MetricTopKAccuracy):
                            continue
                        for mode in ("train", "test"):
                            group = file[mode]
                            if metric.name not in group:
                                count = len(group["epoch"]) if "epoch" in group else len(group.get("accuracy", []))
                                group.create_dataset(metric.name, data=np.full(count, np.nan), maxshape=(None,), chunks=True)
                with open(self.terminalLogPath, "a", encoding="utf-8") as _:
                    pass


    def _print_and_store(self, text: str, end: str = "\n", flush: bool = False) -> None:
        print(text, end=end, flush=flush)
        if torch.distributed.get_rank() == 0:
            with open(self.terminalLogPath, "a", encoding="utf-8") as terminal_file:
                terminal_file.write(text)
                terminal_file.write(end)


    def __call__(self, state: dict) -> None:
        for metric in self.metrics:
            if self.train and metric.logTrain:
                metric.fetchMetric(state)
            if not self.train and metric.logTest:
                metric.fetchMetric(state)

        self.step += 1
        if torch.distributed.get_rank() == 0:
            if self.verbose and self.step % self.logEach == self.logEach - 1:
                self.printTerminal()

    def startTrain(self, trainDataLen) -> None:
        self.epoch += 1
        self.trainDataLen = trainDataLen
        self.step = 0
        self.train = True
        self.start_time = time.time()
        

    def startTest(self) -> None:
        self.step = 0
        self.train = False

    def flush(self) -> None:
        if torch.distributed.get_rank() == 0:
            self.printTerminal()
            if not self.train:
                self._print_and_store("")
            mode = "train" if self.train else "test"
            values = {f"{mode}/{metric.name}": float(np.mean(np.concatenate(metric.buffer)))
                      for metric in self.metrics if metric.buffer and
                      (metric.logTrain if self.train else metric.logTest)}
            values["epoch"] = self.epoch
            tracking.log(values)
            if not self.train and "test/accuracy" in values:
                self.best_accuracy = max(getattr(self, "best_accuracy", 0), values["test/accuracy"])
                tracking.summary({"best_test_accuracy": self.best_accuracy})
            with h5py.File(self.filePath, "r+") as file:
                group = file[mode]
                if "epoch" not in group:
                    group.create_dataset("epoch", shape=(0,), maxshape=(None,), dtype="i8")
                epochs = group["epoch"]
                epochs.resize(len(epochs) + 1, axis=0)
                epochs[-1] = self.epoch
                for metric in self.metrics:
                    if self.train and metric.logTrain:
                        metric.flushData(file, mode = "train" if self.train else "test")
                    if not self.train and metric.logTest:
                        metric.flushData(file, mode = "train" if self.train else "test")
            

    def printTerminal(self) -> None:
        """ print to terminal """
        if self.train:
            self.trainString = f"{'│'.join([metric.getDisplayStr().center(self.columnLen) for metric in self.printTrainMetrics])}"
            
            if self.verbose:
                self._print_and_store(
                    f"\r┃{str(self.epoch).center(self.columnLen-1)}│{self._time().center(self.columnLen-1)}┃{self.trainString}{self.loading_bar(self.step / self.trainDataLen)}",
                    end="",
                    flush=True,
                )
        else:
            start = '\r' if self.verbose else ''
            self._print_and_store(
                f"{start}┃{str(self.epoch).center(self.columnLen-1)}│{self._time().center(self.columnLen-1)}┃{self.trainString}┃{'│'.join([metric.getDisplayStr().center(self.columnLen) for metric in self.printTestMetrics])}┃",
                end="",
            )

    def _time(self) -> str:
        time_seconds = int(time.time() - self.start_time)
        return f"{time_seconds // 60:02d}:{time_seconds % 60:02d} min"

    def printHeader(self) -> None:
        if torch.distributed.get_rank() == 0:
            self._print_and_store(f"┏━━━━━━━╸S╺╸T╺╸A╺╸T╺╸S╺━━━━━━━┳{'T╺╸R╺╸A╺╸I╺╸N '.center((self.columnLen+1)*len(self.printTrainMetrics)-1,'━')}┳{'T╺╸E╺╸S╺╸T '.center((self.columnLen+1)*len(self.printTestMetrics)-1,'━')}┓")
            self._print_and_store(f"┃                             ┃{' '*((self.columnLen+1)*len(self.printTrainMetrics)-1)}┃{' '*((self.columnLen+1)*len(self.printTestMetrics)-1)}┃")
            self._print_and_store(f"┃    epoch     │     time     ┃{'│'.join([metric.name[:self.columnLen].center(self.columnLen) for metric in self.printTrainMetrics])}┃{'│'.join([metric.name[:self.columnLen].center(self.columnLen) for metric in self.printTestMetrics])}┃")
            self._print_and_store(f"┠──────────────┼──────────────╂{'┼'.join(['─'*self.columnLen]*len(self.printTrainMetrics))}╂{'┼'.join(['─'*self.columnLen]*len(self.printTestMetrics))}┨")

class LoadingBar:
    def __init__(self, length: int = 40):
        self.length = length
        self.symbols = ['┈', '░', '▒', '▓']

    def __call__(self, progress: float) -> str:
        p = int(progress * self.length*4 + 0.5)
        d, r = p // 4, p % 4
        return '┠┈' + d * '█' + ((self.symbols[r]) + max(0, self.length-1-d) * '┈' if p < self.length*4 else '') + "┈┨"
