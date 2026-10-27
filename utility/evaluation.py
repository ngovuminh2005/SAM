"""Evaluation schedule shared by training entrypoints."""


def should_evaluate(epoch: int, total_epochs: int, interval: int) -> bool:
    if interval < 1:
        raise ValueError('evalEvery must be a positive number of epochs')
    return epoch % interval == 0 or epoch == total_epochs
