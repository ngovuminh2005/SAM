import os
from pathlib import Path

from .common import ImageFolderWithPaths
from .imagenet import ImageNet


class ImageNetV2DatasetWithPaths(ImageFolderWithPaths):
    def __init__(self, transform=None, location=os.path.expanduser('~/data')):
        path = os.path.join(location, 'imagenetv2-matched-frequency-format-val')
        super().__init__(path, transform=transform)
        # Folder names are ImageNet class indices, not lexicographic indices.
        self.class_to_idx = {name: int(name) for name in self.classes}
        self.samples = [(filename, int(Path(filename).parent.name)) for filename, _ in self.samples]
        self.imgs = self.samples
        self.targets = [label for _, label in self.samples]


class ImageNetV2(ImageNet):
    def populate_train(self):
        pass

    def get_test_dataset(self):
        return ImageNetV2DatasetWithPaths(transform=self.preprocess, location=self.location)
