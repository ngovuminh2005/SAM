"""DenseNet-BC in PyTorch (CIFAR + ImageNet stems).

[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
    Densely Connected Convolutional Networks — https://arxiv.org/abs/1608.06993
"""

import torch
import torch.nn as nn

from utility.args import Args


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channel = 4 * growth_rate
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.AvgPool2d(2, stride=2),
        )

    def forward(self, x):
        return self.down_sample(x)


class _DenseNetCore(nn.Module):
    """DenseNet-BC core: B = bottleneck, C = compression ``reduction``."""

    def __init__(
        self,
        block,
        nblocks,
        growth_rate=12,
        reduction=0.5,
        num_classes=100,
        imagenet_stem=False,
    ):
        super().__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate

        if imagenet_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, inner_channels, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(inner_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.features.add_module(
                "dense_block_layer_{}".format(index),
                self._make_dense_layers(block, inner_channels, nblocks[index]),
            )
            inner_channels += growth_rate * nblocks[index]
            out_channels = int(reduction * inner_channels)
            self.features.add_module(
                "transition_layer_{}".format(index),
                Transition(inner_channels, out_channels),
            )
            inner_channels = out_channels

        self.features.add_module(
            "dense_block{}".format(len(nblocks) - 1),
            self._make_dense_layers(block, inner_channels, nblocks[len(nblocks) - 1]),
        )
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module("bn", nn.BatchNorm2d(inner_channels))
        self.features.add_module("relu", nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(inner_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for index in range(nblocks):
            dense_block.add_module("bottle_neck_layer_{}".format(index), block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block


# depth -> (block_type, per-block layer counts, growth_rate)
_DENSENET_PROFILES = {
    40: (Bottleneck, (6, 6, 6), 12),
    100: (Bottleneck, (16, 16, 16), 12),
    121: (Bottleneck, (6, 12, 24, 16), 32),
}


class DenseNet(nn.Module):
    """Same calling convention as WRN / PyramidNet: ``__init__(num_classes=...)``."""

    def __init__(self, num_classes=10):
        super().__init__()
        d = Args.depth
        if d not in _DENSENET_PROFILES:
            raise RuntimeError(
                "DenseNet: Args.depth={} not supported. Choose one of: {}.".format(
                    d, sorted(_DENSENET_PROFILES.keys())
                )
            )
        block, nblocks, growth = _DENSENET_PROFILES[d]
        imagenet = Args.dataset == "ImageNet"
        self.net = _DenseNetCore(
            block,
            list(nblocks),
            growth_rate=growth,
            reduction=0.5,
            num_classes=num_classes,
            imagenet_stem=imagenet,
        )

    def setBatchNormTracking(self, track_running_stats: bool):
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.track_running_stats = track_running_stats

    def forward(self, x):
        return self.net(x)
