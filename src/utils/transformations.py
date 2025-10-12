import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from scipy.ndimage import zoom

TARGET_SHAPE = (128, 128, 64)   # [W, H, D]. Will be transposed to (D, H, W)

# numpy-based bounding box crop
def bbox_crop(img):
    """Crop image to the bounding box of the nonzero region of the image."""
    coords = np.array(np.nonzero(img))
    if coords.size == 0:
        # No nonzero region, return original(s)
        return img
    top_left = coords.min(axis=1)
    bottom_right = coords.max(axis=1) + 1  # slices are exclusive at the top
    slices = tuple(slice(top, bottom) for top, bottom in zip(top_left, bottom_right))
    img[slices]
    

def resize_volume(volume, target_shape=TARGET_SHAPE, order=1):
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=order)

def pad_and_resize(volume, target_shape=TARGET_SHAPE, pad_xy=10, pad_z=5, order=1) -> np.ndarray:

    faux_target_shape = (target_shape[0] - 2 * pad_xy,
                         target_shape[1] - 2 * pad_xy,
                         target_shape[2] - 2 * pad_z)
    
    resized = resize_volume(volume, target_shape=faux_target_shape, order=order)

    # Create final padded volume
    final_volume = np.pad(resized, ((pad_xy, pad_xy), (pad_xy, pad_xy), (pad_z, pad_z)), mode='constant', constant_values=0)

    return final_volume


# Torch-Based Augmentations

def create_identity_grid(volume: torch.Tensor) -> torch.Tensor:
    """
    Create a normalized identity grid for spatial transformations.
    Accepts volume of shape [C, D, H, W] and returns grid of shape (1, D, H, W, 3).
    """
    _, D, H, W = volume.shape
    z = torch.linspace(-1, 1, D, device=volume.device, dtype=volume.dtype)
    y = torch.linspace(-1, 1, H, device=volume.device, dtype=volume.dtype)
    x = torch.linspace(-1, 1, W, device=volume.device, dtype=volume.dtype)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    grid = torch.stack([xx, yy, zz], dim=-1)  # (D, H, W, 3)
    return grid.unsqueeze(0)  # (1, D, H, W, 3)

# Base class for 3D augmentations
class Augmentation3D(nn.Module):
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p

    def forward(self, img: torch.Tensor):
        if not self.training or torch.rand(1) > self.p:
            return img
        return self.forward_img(img)

    def forward_img(self, img):
        raise NotImplementedError

class RandomFlip3D(Augmentation3D):
    def __init__(self, p=0.5, dims=('H', 'W')):
        super().__init__()
        self.p = p
        self.dims = dims

    def forward_img(self, img):

        do_flip = torch.rand(1) < self.p
        if do_flip:
            if 'W' in self.dims and torch.rand(1) < self.p:
                img = torch.flip(img, dims=[3])
            if 'H' in self.dims and torch.rand(1) < self.p:
                img = torch.flip(img, dims=[2])
        return img

class RandomTranslation3D(Augmentation3D):
    def __init__(self, max_shift: Tuple[int, int, int] = (5, 10, 10), p=1.0):
        """
        max_shift: (max_shift_D, max_shift_H, max_shift_W)
        """
        super().__init__(p)
        self.max_shift = max_shift

    def forward_img(self, img):
        if not self.training:
            return img
        C, D, H, W = img.shape
        shifts = [torch.randint(-m, m + 1, (1,)).item() for m in self.max_shift]
        pad = (self.max_shift[2],)*2 + (self.max_shift[1],)*2 + (self.max_shift[0],)*2
        img_padded = F.pad(img, pad)
        d0 = self.max_shift[0] + shifts[0]
        h0 = self.max_shift[1] + shifts[1]
        w0 = self.max_shift[2] + shifts[2]
        img_out = img_padded[:, d0:d0 + D, h0:h0 + H, w0:w0 + W]
        return img_out

class ElasticDeformation3D(Augmentation3D):
    def __init__(self, alpha=2.0, sigma=8.0, p=1.0):
        super().__init__(p)
        self.alpha = alpha
        self.sigma = sigma

    def forward_img(self, img):

        C, D, H, W = img.shape
        grid = create_identity_grid(img)  # [1, D, H, W, 3]
        disp = torch.randn(D, H, W, 3, device=img.device, dtype=img.dtype)
        # Efficient 3D Gaussian blur using conv3d
        def gaussian_blur3d_torch(vol, sigma, kernel_size=5):
            x = torch.arange(kernel_size, device=vol.device, dtype=vol.dtype) - kernel_size // 2
            kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel_1d = kernel_1d / kernel_1d.sum()
            kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
            kernel_3d = kernel_3d.unsqueeze(0).unsqueeze(0)  # [1,1,K,K,K]
            padding = kernel_size // 2
            vol = vol.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
            blurred = torch.nn.functional.conv3d(vol, kernel_3d, padding=padding)
            return blurred.squeeze(0).squeeze(0)  # [D, H, W]
        for i in range(3):
            disp[..., i] = gaussian_blur3d_torch(disp[..., i], self.sigma)
        disp = disp * self.alpha
        warped_grid = grid + disp.unsqueeze(0) / torch.tensor([W/2, H/2, D/2], device=img.device, dtype=img.dtype)
        # grid_sample expects [N, C, D, H, W] and grid [N, D, H, W, 3]
        img_b = img.unsqueeze(0)
        img_out = F.grid_sample(img_b, warped_grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        return img_out.squeeze(0)

class RandomRotation3D(Augmentation3D):
    def __init__(self, max_angle=10.0, p=1.0):
        super().__init__(p)
        self.max_angle = max_angle

    def forward_img(self, img):

        # img: [C, D, H, W]
        C, D, H, W = img.shape
        # Random rotation angles (in radians)
        angles = (torch.rand(3, device=img.device, dtype=img.dtype) * 2 - 1) * self.max_angle * (torch.pi / 180)
        cx, cy, cz = torch.cos(angles)
        sx, sy, sz = torch.sin(angles)
        Rx = torch.tensor([[1, 0, 0],
                           [0, cx, -sx],
                           [0, sx, cx]], device=img.device, dtype=img.dtype)
        Ry = torch.tensor([[cy, 0, sy],
                           [0, 1, 0],
                           [-sy, 0, cy]], device=img.device, dtype=img.dtype)
        Rz = torch.tensor([[cz, -sz, 0],
                           [sz, cz, 0],
                           [0, 0, 1]], device=img.device, dtype=img.dtype)
        R = Rz @ Ry @ Rx  # (3, 3)
        affine = torch.zeros((1, 3, 4), device=img.device, dtype=img.dtype)
        affine[0, :, :3] = R

        # Add batch dimension for affine_grid and grid_sample
        img_b = img.unsqueeze(0)   # [1, C, D, H, W]
        grid = F.affine_grid(affine, img_b.size(), align_corners=True)
        img_out = F.grid_sample(img_b, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return img_out.squeeze(0)

class RandomGamma3D(Augmentation3D):
    def __init__(self, gamma_range=(0.8, 1.2), p=1.0):
        super().__init__(p)
        self.gamma_range = gamma_range

    def forward_img(self, img):
        if not self.training:
            return img
        gamma = torch.empty(1).uniform_(*self.gamma_range).item()
        return torch.clamp(img ** gamma, min=0.0)

class RandomGaussianBlur3D(Augmentation3D):
    def __init__(self, sigma_range=(0.0, 0.5), p=1.0):
        super().__init__(p)
        self.sigma_range = sigma_range

    def forward_img(self, img):
        if not self.training:
            return img
        sigma = torch.empty(1, device=img.device, dtype=img.dtype).uniform_(*self.sigma_range).item()
        def get_gaussian_kernel1d(kernel_size, sigma):
            x = torch.arange(kernel_size, device=img.device, dtype=img.dtype) - kernel_size // 2
            kernel = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel = kernel / kernel.sum()
            return kernel

        kernel_size = 5
        if sigma < 1e-3:
            return img

        kernel_1d = get_gaussian_kernel1d(kernel_size, sigma)
        kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
        kernel_3d = kernel_3d.expand(img.shape[0], 1, kernel_size, kernel_size, kernel_size).contiguous()
        kernel_3d = kernel_3d / kernel_3d.sum()

        padding = kernel_size // 2
        # Add batch dimension: [1, C, D, H, W]
        img_b = img.unsqueeze(0)
        img_blur = torch.nn.functional.conv3d(
            img_b, kernel_3d, padding=padding, groups=img.shape[0]
        )
        return img_blur.squeeze(0)

class MRIAugmentationPipeline(nn.Module):
    def __init__(self, transforms=[
        RandomFlip3D(p=0.5),    # probability for both axes
        RandomTranslation3D(max_shift=(5, 10, 10), p=0.8),
        ElasticDeformation3D(alpha=2.0, sigma=8.0, p=0.2),
        RandomRotation3D(max_angle=8.0, p=0.3),
        RandomGamma3D(gamma_range=(0.8, 1.2), p=1.0),
        RandomGaussianBlur3D(sigma_range=(0, 0.5), p=0.3)
    ]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    def forward(self, img):
        for t in self.transforms:
            img = t(img)
        return img
    
MRI_AUGMENTATION_PIPELINE = MRIAugmentationPipeline()

# Example usage:
# pipeline = MRIAugmentationPipeline()
# img_aug = pipeline(img_tensor)