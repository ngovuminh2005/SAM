#https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/simple_vit.py
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from utility.args import Args

Args.add_argument("--ViTPatchsize", type=int, help="")
Args.add_argument("--ViTSize", type=str, help="")


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SimpleViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Blog-aligned branch: ViT-B/16 or ViT-B/32 from timm with ImageNet pretrained weights.
        # Keep TinyImageNet pipeline unchanged by resizing on-the-fly to 384.
        self._use_timm_vit = False
        self._timm_input_size = 384
        if Args.ViTSize == "B":
            if Args.ViTPatchsize not in [16, 32]:
                raise RuntimeError(f"ViT-B supports ViTPatchsize 16 or 32, got {Args.ViTPatchsize}")
            try:
                import timm
            except ImportError as error:
                raise RuntimeError(
                    "timm is required for ViT-B architecture. Install with: pip install timm"
                ) from error

            timm_model_name = f"vit_base_patch{Args.ViTPatchsize}_384"
            self.timm_vit = timm.create_model(
                timm_model_name,
                pretrained=True,
                num_classes=num_classes,
            )
            self._use_timm_vit = True
            return

        image_size = 32 if Args.imageSize == 0 else Args.imageSize 
        patch_size = Args.ViTPatchsize
        
        # ViT-S/B widths from https://arxiv.org/pdf/2106.04560.pdf Table 2.
        # Ti: common ViT-Tiny / DeiT-Tiny-style width (embed 192, 12 layers, mlp 4×dim, 3 heads ≈5–6M params on CIFAR).
        architecture_parameters = {
            "Ti": (192, 12, 768, 3),
            "S": (384, 12, 1536, 6),
            "B": (768, 12, 3072, 12),
        }
        if not Args.ViTSize in architecture_parameters:
            raise RuntimeError(f"ViTSize must be in {', '.join(architecture_parameters.keys())} but is {Args.ViTSize}")
        dim, depth, mlp_dim, heads = architecture_parameters[Args.ViTSize]
        

        channels = 3
        dim_head = 64
        
        
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        self.linear_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        if self._use_timm_vit:
            if img.shape[-1] != self._timm_input_size or img.shape[-2] != self._timm_input_size:
                img = F.interpolate(
                    img,
                    size=(self._timm_input_size, self._timm_input_size),
                    mode="bicubic",
                    align_corners=False,
                )
            return self.timm_vit(img)

        *_, h, w, dtype = *img.shape, img.dtype

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x)
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return self.linear_head(x)
    
    
    def setBatchNormTracking(self, track_running_stats: bool):
        if self._use_timm_vit:
            return
        for m in self.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.track_running_stats = track_running_stats