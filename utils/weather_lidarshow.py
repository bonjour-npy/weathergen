from pathlib import Path
from typing import Literal
# import utils.inference
import utils.render
import einops
import numpy as np
from torchvision.utils import make_grid, save_image
import torch
import torch.nn.functional as F
from torch import nn
import datasets as ds
from PIL import Image
import matplotlib.cm as cm
import numba
import numpy as np
import matplotlib.pyplot as plt

def get_hdl64e_linear_ray_angles(
    HH: int = 32, WW: int = 1024, device: torch.device = "cpu"
):
    h_up, h_down = 3, -25
    w_left, w_right = 180, -180
    elevation = 1 - torch.arange(HH, device=device) / HH  # [0, 1]
    elevation = elevation * (h_up - h_down) + h_down  # [-25, 3]
    azimuth = 1 - torch.arange(WW, device=device) / WW  # [0, 1]
    azimuth = azimuth * (w_left - w_right) + w_right  # [-180, 180]
    [elevation, azimuth] = torch.meshgrid([elevation, azimuth], indexing="ij")
    angles = torch.stack([elevation, azimuth])[None].deg2rad()
    return angles

class LiDARUtility(nn.Module):
    def __init__(
        self,
        resolution: tuple[int],
        image_format: Literal["log_depth", "inverse_depth", "depth"],
        min_depth: float,
        max_depth: float,
        ray_angles: torch.Tensor = None,
    ):
        super().__init__()
        assert image_format in ("log_depth", "inverse_depth", "depth")
        self.resolution = resolution
        self.image_format = image_format
        self.min_depth = min_depth
        self.max_depth = max_depth
        if ray_angles is None:
            ray_angles = get_hdl64e_linear_ray_angles(*resolution)
        else:
            assert ray_angles.ndim == 4 and ray_angles.shape[1] == 2
        ray_angles = F.interpolate(
            ray_angles,
            size=self.resolution,
            mode="nearest-exact",
        )
        self.register_buffer("ray_angles", ray_angles)

    @staticmethod
    def denormalize(x):
        """Scale from [-1, +1] to [0, 1]"""
        return (x + 1) / 2

    @staticmethod
    def normalize(x):
        """Scale from [0, 1] to [-1, +1]"""
        return x * 2 - 1

    @torch.no_grad()
    def to_xyz(self, metric):
        assert metric.dim() == 4
        mask = (metric > self.min_depth) & (metric < self.max_depth)
        phi = self.ray_angles[:, [0]]
        theta = self.ray_angles[:, [1]]
        grid_x = metric * phi.cos() * theta.cos()
        grid_y = metric * phi.cos() * theta.sin()
        grid_z = metric * phi.sin()
        xyz = torch.cat((grid_x, grid_y, grid_z), dim=1)
        xyz = xyz * mask.float()
        return xyz

    @torch.no_grad()
    def convert_depth(
        self,
        metric: torch.Tensor,
        mask: torch.Tensor | None = None,
        image_format: str = None,
    ) -> torch.Tensor:
        """
        Convert metric depth in [0, `max_depth`] to normalized depth in [0, 1].
        """
        if image_format is None:
            image_format = self.image_format
        if mask is None:
            mask = self.get_mask(metric)
        if image_format == "log_depth":
            normalized = torch.log2(metric + 1) / np.log2(self.max_depth + 1)
        elif image_format == "inverse_depth":
            normalized = self.min_depth / metric.add(1e-8)
        elif image_format == "depth":
            normalized = metric.div(self.max_depth)
        else:
            raise ValueError
        normalized = normalized.clamp(0, 1) * mask
        return normalized

    @torch.no_grad()
    def revert_depth(
        self,
        normalized: torch.Tensor,
        image_format: str = None,
    ) -> torch.Tensor:
        """
        Revert normalized depth in [0, 1] back to metric depth in [0, `max_depth`].
        """
        if image_format is None:
            image_format = self.image_format
        if image_format == "log_depth":
            metric = torch.exp2(normalized * np.log2(self.max_depth + 1)) - 1
        elif image_format == "inverse_depth":
            metric = self.min_depth / normalized.add(1e-8)
        elif image_format == "depth":
            metric = normalized.mul(self.max_depth)
        else:
            raise ValueError
        return metric * self.get_mask(metric)

    def get_mask(self, metric):
        mask = (metric > self.min_depth) & (metric < self.max_depth)
        return mask.float()

lidar_utils = LiDARUtility(
    resolution=(64, 1024),
    image_format="log_depth",
    min_depth=1.45,
    max_depth=80.0,
    ray_angles=None,
)

def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array
def preprocess(xyzrdm):
    x = []
    x += [lidar_utils.convert_depth(xyzrdm[[4]])]
    x += [lidar_utils.convert_depth(xyzrdm[[3]])]
    x = torch.cat(x, dim=0)
    x = lidar_utils.normalize(x)
    return x

def render(x):
    xyz = lidar_utils.to_xyz(x[[0]].unsqueeze(dim=0) * lidar_utils.max_depth)
    xyz /= lidar_utils.max_depth
    z_min, z_max = -2 / lidar_utils.max_depth, 0.5 / lidar_utils.max_depth
    z = (xyz[:, [2]] - z_min) / (z_max - z_min)
    colors = utils.render.colorize(z.clamp(0, 1), cm.viridis) / 255
    R, t = utils.render.make_Rt(pitch=torch.pi / 3, yaw=torch.pi / 4, z=0.8)
    # R, t = utils.render.make_Rt(pitch=0, yaw=0, z=0.8)
    bev = 1 - utils.render.render_point_clouds(
        points = einops.rearrange(xyz, "B C H W -> B (H W) C"),
        colors = 1 - einops.rearrange(colors, "B C H W -> B (H W) C"),
        R=R.to(xyz),
        t=t.to(xyz),
    )
    return bev

def load_points_as_images_test(
    point_path: str,
    scan_unfolding: bool = False,
    H: int = 64,
    W: int = 1024,
    min_depth: float = 1.45, # 1.45
    max_depth: float = 80, # 80.0
):
    # load xyz & intensity and add depth & mask
    # points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 4))
    points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 5))
    points = points[:, :4]
    xyz = points[:, :3]  # xyz
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    mask = (depth >= min_depth) & (depth <= max_depth)
    points = np.concatenate([points, depth, mask], axis=1)

    if scan_unfolding:
        # the i-th quadrant
        # suppose the points are ordered counterclockwise
        quads = np.zeros_like(x, dtype=np.int32)
        quads[(x >= 0) & (y >= 0)] = 0  # 1st
        quads[(x < 0) & (y >= 0)] = 1  # 2nd
        quads[(x < 0) & (y < 0)] = 2  # 3rd
        quads[(x >= 0) & (y < 0)] = 3  # 4th

        # split between the 3rd and 1st quadrants
        diff = np.roll(quads, shift=1, axis=0) - quads
        delim_inds, _ = np.where(diff == 3)  # number of lines
        inds = list(delim_inds) + [len(points)]  # add the last index

        # vertical grid
        grid_h = np.zeros_like(x, dtype=np.int32)
        cur_ring_idx = H - 1  # ...0
        for i in reversed(range(len(delim_inds))):
            grid_h[inds[i] : inds[i + 1]] = cur_ring_idx
            if cur_ring_idx >= 0:
                cur_ring_idx -= 1
            else:
                break
    else:
        h_up, h_down = np.deg2rad(3), np.deg2rad(-25)
        elevation = np.arcsin(z / depth) + abs(h_down)
        grid_h = 1 - elevation / (h_up - h_down)
        grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)

    # horizontal grid
    azimuth = -np.arctan2(y, x)  # [-pi,pi]
    grid_w = (azimuth / np.pi + 1) / 2 % 1  # [0,1]
    grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)

    grid = np.concatenate((grid_h, grid_w), axis=1)

    # projection
    order = np.argsort(-depth.squeeze(1))
    proj_points = np.zeros((H, W, 4 + 2), dtype=points.dtype)
    proj_points = scatter(proj_points, grid[order], points[order]).astype(np.float32)
    xyzrdm = proj_points.transpose(2, 0, 1)
    xyzrdm *= xyzrdm[[5]]
    xyzrdm = torch.from_numpy(xyzrdm)
    x = preprocess(xyzrdm)

    xs = lidar_utils.denormalize(x)
    xs[[0]] = lidar_utils.revert_depth(xs[[0]]) / lidar_utils.max_depth

    bev = render(xs)
    # save_image(bev, "./samples_bev.png", nrow=4)

    img = x[[0]]
    # out = out.squeeze()
    # plt.imshow(out, cmap='jet')
    # plt.savefig('./depth.png', dpi=300, bbox_inches='tight', pad_inches=0)

    return bev, img

# load_points_as_images("./000003.bin")