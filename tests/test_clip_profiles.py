import unittest
from unittest.mock import patch

import torch
from PIL import Image

import clip.clip as clip
from clip.model import CLIP
from src.args import parse_arguments
from src.models.utils import cosine_lr


class CLIPProfileTests(unittest.TestCase):
    def test_large_model_names_use_checkpoint_resolution(self):
        for name, resolution, checksum, filename in [
            ('ViT-L/14', 224,
             'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836', 'ViT-L-14.pt'),
            ('ViT-L/14@336px', 336,
             '3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02', 'ViT-L-14-336px.pt'),
        ]:
            with self.subTest(model=name):
                # A small checkpoint exercises the real architecture inference
                # and preprocessing without downloading large pretrained weights.
                source = CLIP(embed_dim=32, image_resolution=resolution, vision_layers=1,
                              vision_width=64, vision_patch_size=14, context_length=16,
                              vocab_size=100, transformer_width=64, transformer_heads=1,
                              transformer_layers=1)
                self.assertIn(name, clip.available_models())
                self.assertTrue(clip._MODELS[name].endswith(checksum + '/' + filename))
                with patch('clip.clip._download', return_value='mock.pt') as download, \
                        patch('clip.clip.torch.jit.load', return_value=source):
                    model, train, val = clip.load(name, device='cpu', jit=False)
                download.assert_called_once_with(clip._MODELS[name])
                self.assertEqual(model.visual.input_resolution, resolution)
                self.assertEqual(model.visual.conv1.kernel_size, (14, 14))
                self.assertEqual(model.visual.proj.shape[1], 32)
                image = Image.new('RGB', (400, 400))
                self.assertEqual(tuple(train(image).shape), (3, resolution, resolution))
                sample = val(image)
                self.assertEqual(tuple(sample.shape), (3, resolution, resolution))
                with torch.no_grad():
                    features = model.encode_image(sample.unsqueeze(0))
                self.assertEqual(tuple(features.shape), (1, 32))
                self.assertTrue(torch.isfinite(features).all())

    def test_end_to_end_argument_recipe(self):
        args = parse_arguments([])
        self.assertEqual(args.lr, 3e-5)
        self.assertEqual(args.batch_size, 512)
        self.assertEqual(args.warmup_length, 500)
        self.assertEqual(args.epochs, 10)

    def test_linear_warmup_then_cosine_decay(self):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=3e-5)
        scheduler = cosine_lr(optimizer, 3e-5, 2, 10)
        values = []
        for step in range(11):
            scheduler(step)
            values.append(optimizer.param_groups[0]['lr'])
        self.assertAlmostEqual(values[0], 1.5e-5)
        self.assertAlmostEqual(values[1], 3e-5)
        self.assertAlmostEqual(values[2], 3e-5)
        self.assertAlmostEqual(values[6], 1.5e-5)
        self.assertAlmostEqual(values[10], 0)
        self.assertTrue(all(a >= b for a, b in zip(values[2:], values[3:])))


if __name__ == '__main__':
    unittest.main()
