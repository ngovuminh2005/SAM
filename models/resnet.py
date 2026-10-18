import torchvision.models as models
import torch.nn as nn

from utility.args import Args


def _get_resnet_blocks_layers(depth: int):
    blocks_layers = {
        18: (models.resnet.BasicBlock, [2, 2, 2, 2]),
        34: (models.resnet.BasicBlock, [3, 4, 6, 3]),
        50: (models.resnet.Bottleneck, [3, 4, 6, 3]),
        101: (models.resnet.Bottleneck, [3, 4, 23, 3]),
        152: (models.resnet.Bottleneck, [3, 8, 36, 3]),
    }
    if depth not in blocks_layers:
        raise RuntimeError(
            f"Model depth {depth} not supported for ResNet. Supported: {', '.join(map(str, blocks_layers.keys()))}."
        )
    return blocks_layers[depth]


class ResNet(models.ResNet):
    def __init__(self, num_classes=10, cifar_stem=None):
        block, layers = _get_resnet_blocks_layers(Args.depth)
        super().__init__(block=block, layers=layers, num_classes=num_classes)

        # Single implementation for both ImageNet and CIFAR:
        # CIFAR uses a smaller stem and no maxpool.
        if cifar_stem is None:
            cifar_stem = Args.dataset in ["CIFAR10", "CIFAR100"]

        if cifar_stem:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()

    def setBatchNormTracking(self, track_running_stats: bool):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.track_running_stats = track_running_stats


class ResNetCIFAR(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(num_classes=num_classes, cifar_stem=True)
