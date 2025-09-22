from pathlib import Path
from typing import Literal

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
from rich import print
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=42)


def get_hdl64e_linear_ray_angles(HH: int = 64, WW: int = 1024, device: torch.device = "cpu"):
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
        device = metric.device
        mask = (metric > self.min_depth) & (metric < self.max_depth)
        phi = self.ray_angles[:, [0]].to(device=device)
        theta = self.ray_angles[:, [1]].to(device=device)
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


def r2p(x):
    xyz = lidar_utils.to_xyz(x[[0]].unsqueeze(dim=0) * lidar_utils.max_depth)
    return xyz


def preprocess_weather(xyzrdm):
    x = []
    x += [lidar_utils.convert_depth(xyzrdm[[4]])]  # depth [0, max_depth] -> [0, 1]
    x += [lidar_utils.convert_depth(xyzrdm[[3]])]  # reflectance [0, max_depth] -> [0, 1]
    x = torch.cat(x, dim=0)
    x = lidar_utils.normalize(x)  # [0, 1] -> [-1, 1]
    return x


def weather_process(
    x_normal: torch.Tensor, weather_flag: str, xyz_normal: torch.Tensor, depth_normal: torch.Tensor
):
    device = x_normal.device
    B, C, H, W = x_normal.shape
    x_0 = torch.empty(B, C, H, W).to(device=device)
    if weather_flag == "snow":
        for i in range(B):
            xs = lidar_utils.denormalize(x_normal[i])
            xs[[0]] = lidar_utils.revert_depth(xs[[0]]) / lidar_utils.max_depth
            new_xyz = r2p(xs)  # new_xyz [1, 3, 64, 1024]
            new_xyz = (
                einops.rearrange(new_xyz, "B C H W -> B (H W) C").squeeze().cpu().numpy()
            )  # new_xyz (65536, 3)

            x_multi = np.random.random(1800)
            y_multi = np.random.random(1800)
            z_multi = np.random.random(1800)
            x_array = np.round((rng.integers(low=-15, high=15, size=1800) * x_multi), 5).reshape(1800, 1)
            y_array = np.round((rng.integers(low=-15, high=15, size=1800) * y_multi), 5).reshape(1800, 1)
            z_array = np.round((rng.integers(low=-10, high=0.8, size=1800) * z_multi), 5).reshape(1800, 1)
            x = new_xyz[:, [0]]
            x = np.vstack((x, x_array))
            y = new_xyz[:, [1]]
            y = np.vstack((y, y_array))
            z = new_xyz[:, [2]]
            z = np.vstack((z, z_array))
            xyz = np.concatenate([x, y, z], axis=1)
            depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True) + 0.0001
            ref = depth
            mask = (depth >= 1.45) & (depth <= 80.0)
            points = np.concatenate([x, y, z, ref, depth, mask], axis=1)

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
            x = preprocess_weather(xyzrdm).to(device=device)

            height = xyzrdm[[2]].squeeze()
            depth = depth_normal[i].squeeze() / lidar_utils.max_depth
            zeros = torch.zeros(H, W).to(device=device)
            ones = torch.ones(H, W).to(device=device)
            # mask_mask_noise = torch.empty(H, W).bernoulli_(0.8).to(device=device)
            mask_mask_1 = torch.empty(H, W).bernoulli_(0.8).to(device=device)
            mask_1 = torch.empty(H, 1).bernoulli_(0.9).to(device=device) * mask_mask_1
            noise = ones * 0.005
            noise = torch.where(depth < 0.13, noise, zeros) * mask_mask_1
            mask = mask_1
            x[[0]] = noise + x[[0]]
            xs = mask * x + (1 - mask) * -1
            x_0[i] = xs

    if weather_flag == "fog":
        for i in range(B):
            xs = lidar_utils.denormalize(x_normal[i])
            xs[[0]] = lidar_utils.revert_depth(xs[[0]]) / lidar_utils.max_depth
            new_xyz = r2p(xs)  # new_xyz [1, 3, 64, 1024]
            new_xyz = (
                einops.rearrange(new_xyz, "B C H W -> B (H W) C").squeeze().cpu().numpy()
            )  # new_xyz (65536, 3)

            x_multi = np.random.random(5500)
            y_multi = np.random.random(5500)
            z_multi = np.random.random(5500)
            x_array = np.round((rng.integers(low=-15, high=15, size=5500) * x_multi), 5).reshape(5500, 1)
            y_array = np.round((rng.integers(low=-15, high=15, size=5500) * y_multi), 5).reshape(5500, 1)
            z_array = np.round((rng.integers(low=-10, high=0.8, size=5500) * z_multi), 5).reshape(5500, 1)
            x = new_xyz[:, [0]]
            x = np.vstack((x, x_array))
            y = new_xyz[:, [1]]
            y = np.vstack((y, y_array))
            z = new_xyz[:, [2]]
            z = np.vstack((z, z_array))
            xyz = np.concatenate([x, y, z], axis=1)
            depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True) + 0.0001
            ref = depth
            mask = (depth >= 1.45) & (depth <= 80.0)
            points = np.concatenate([x, y, z, ref, depth, mask], axis=1)

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
            x = preprocess_weather(xyzrdm).to(device=device)

            height = xyzrdm[[2]].squeeze()
            depth = depth_normal[i].squeeze() / lidar_utils.max_depth
            zeros = torch.zeros(H, W).to(device=device)
            ones = torch.ones(H, W).to(device=device)
            mask_mask_noise = torch.empty(H, W).bernoulli_(0.8).to(device=device)
            mask_mask_2 = torch.empty(H, W).bernoulli_(0.3).to(device=device)
            mask_mask_3 = torch.empty(H, 1).bernoulli_(0.55).to(device=device)
            noise = ones * 0.005
            noise = torch.where(depth < 0.13, noise, zeros) * mask_mask_noise

            mask_mask_atten = torch.empty(H, W).bernoulli_(0.8).to(device=device)
            fog_atten_out = torch.where(depth > 0.03, ones, zeros) * mask_mask_atten
            fog_atten_in = torch.where(depth < 0.03, ones, zeros)
            mask = fog_atten_out + fog_atten_in
            xs = mask * x + (1 - mask) * -1

            mask_1 = torch.where(depth > 0.14, zeros, ones)
            mask_2 = torch.where(depth > 0.14, ones, zeros) * mask_mask_2 * mask_mask_3
            mask = mask_1 + mask_2
            xs[[0]] = noise + xs[[0]]
            xs = mask * xs + (1 - mask) * -1
            x_0[i] = xs

    if weather_flag == "rain":
        for i in range(B):
            height = xyz_normal[i][2]
            depth = depth_normal[i].squeeze() / lidar_utils.max_depth
            zeros = torch.zeros(H, W).to(device=device)
            ones = torch.ones(H, W).to(device=device)
            mask_mask_1 = torch.empty(H, 1).bernoulli_(0.3).to(device=device)
            mask_mask_2 = torch.empty(H, 1).bernoulli_(0.2).to(device=device)
            mask_mask_3 = torch.empty(H, 1).bernoulli_(0.5).to(device=device)
            mask_rain = torch.where(depth < 0.2, ones, zeros)
            mask_1 = torch.where(height > -1.2, ones, zeros) * mask_mask_1 * mask_rain
            # mask for depth
            mask_2 = torch.where(depth > 0.08, ones, zeros) * mask_mask_3 * mask_mask_2
            mask_3 = torch.where(depth < 0.08, ones, zeros) * mask_mask_1
            mask = mask_1 + mask_3 + mask_2
            xs = mask * x_normal[i] + (1 - mask) * -1
            x_0[i] = xs

    if weather_flag == "wet_ground":
        for i in range(B):
            height = xyz_normal[i][2]
            depth = depth_normal[i].squeeze() / lidar_utils.max_depth
            zeros = torch.zeros(H, W).to(device=device)
            ones = torch.ones(H, W).to(device=device)
            mask_mask_1 = torch.empty(H, 1).bernoulli_(0.5).to(device=device)
            mask_mask_2 = torch.empty(H, W).bernoulli_(0.2).to(device=device)
            mask_1 = torch.where(height > -1.2, ones, zeros)
            mask_2 = torch.where(depth > 0.06, zeros, ones) * mask_mask_1 * mask_mask_2
            mask = mask_1 + mask_2
            xs = mask * x_normal[i] + (1 - mask) * -1
            x_0[i] = xs

    if weather_flag == "normal":
        x_0 = x_normal

    return x_0


def stf_process(weather_flag: str, x_0: torch.Tensor):
    device = x_0.device
    B, C, H, W = x_0.shape
    x_weather = torch.empty(B, C, H, W).to(device=device)
    if weather_flag == "snow":
        all_point_path = np.genfromtxt("./seeingthroughfog/snow_day.txt", dtype="U", delimiter="\n")
        selected_path = np.random.choice(all_point_path, size=B, replace=False)
        for i in range(B):
            point_path = Path("./seeingthroughfog/lidar_hdl64_strongest/") / (selected_path[i] + ".bin")
            points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 5))
            points = points[:, :4]
            xyz = points[:, :3]  # xyz
            x = xyz[:, [0]]
            y = xyz[:, [1]]
            z = xyz[:, [2]]
            depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
            mask = (depth >= 1.45) & (depth <= 80)
            points = np.concatenate([points, depth, mask], axis=1)

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
            x = preprocess_weather(xyzrdm)
            x_weather[i] = x

    if weather_flag == "rain":
        all_point_path = np.genfromtxt("./seeingthroughfog/rain.txt", dtype="U", delimiter="\n")
        selected_path = np.random.choice(all_point_path, size=B, replace=False)
        for i in range(B):
            point_path = Path("./seeingthroughfog/lidar_hdl64_strongest/") / (selected_path[i] + ".bin")
            points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 5))
            points = points[:, :4]
            xyz = points[:, :3]  # xyz
            x = xyz[:, [0]]
            y = xyz[:, [1]]
            z = xyz[:, [2]]
            depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
            mask = (depth >= 1.45) & (depth <= 80)
            points = np.concatenate([points, depth, mask], axis=1)

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
            x = preprocess_weather(xyzrdm)
            x_weather[i] = x

    if weather_flag == "fog":
        all_point_path = np.genfromtxt("./seeingthroughfog/dense_fog_day.txt", dtype="U", delimiter="\n")
        selected_path = np.random.choice(all_point_path, size=B, replace=False)
        for i in range(B):
            point_path = Path("./seeingthroughfog/lidar_hdl64_strongest/") / (selected_path[i] + ".bin")
            points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 5))
            points = points[:, :4]
            xyz = points[:, :3]  # xyz
            x = xyz[:, [0]]
            y = xyz[:, [1]]
            z = xyz[:, [2]]
            depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
            mask = (depth >= 1.45) & (depth <= 80)
            points = np.concatenate([points, depth, mask], axis=1)

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
            x = preprocess_weather(xyzrdm)
            x_weather[i] = x

    return x_weather


class StatisticsBasedWeatherGenerator(nn.Module):
    """
    替代原始的 weather_process 函数，使用基于统计和分布的来模拟天气进行数据增强
    部分增强函数不是在 range image 上进行的, 因此需要 lidar_utils 中的 to_xyz 函数进行反投影

    Args:

    Returns:

    """

    def __init__(self, lidar_utils: LiDARUtility = lidar_utils):
        super(StatisticsBasedWeatherGenerator(), self).__init__()
        self.lidar_utils = lidar_utils

    def density_matching_vectorized():
        """
        将 clear weather 点云的密度匹配到 adverse weather 点云上 (在 3D 空间, 而不是 range image)

        Args:

        """
        pass

    def histogram_matching_vectorized():
        pass

    def range_depth_scale():
        pass

    def forward():
        pass
