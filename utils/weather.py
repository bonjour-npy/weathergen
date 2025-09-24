from pathlib import Path
from typing import Literal

import utils.render
import einops
import numpy as np
import functools
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
    min_depth=0.0,
    max_depth=120.0,
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


# 若考虑反射强度，反射强度已经是 [0, 1]
def preprocess_weather_with_rflct(xyzrdm):
    x = []
    x += [lidar_utils.convert_depth(xyzrdm[[4]])]  # depth [0, max_depth] -> [0, 1]
    x += [xyzrdm[[3]]]  # reflectance 已经是 [0, 1]
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
            azimuth = -np.arctan2(y, x)  # [-pi, pi]
            grid_w = (azimuth / np.pi + 1) / 2 % 1  # [0, 1]
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
            azimuth = -np.arctan2(y, x)  # [-pi, pi]
            grid_w = (azimuth / np.pi + 1) / 2 % 1  # [0, 1]
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
            points = np.fromfile(point_path, dtype=np.float32).reshape((-1, 5))  # five channels
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
            grid_w = (azimuth / np.pi + 1) / 2 % 1  # [0, 1]
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
            azimuth = -np.arctan2(y, x)  # [-pi, pi]
            grid_w = (azimuth / np.pi + 1) / 2 % 1  # [0, 1]
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
            azimuth = -np.arctan2(y, x)  # [-pi, pi]
            grid_w = (azimuth / np.pi + 1) / 2 % 1  # [0, 1]
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
        super().__init__()
        self.lidar_utils = lidar_utils
        self.H, self.W = self.lidar_utils.resolution
        self.min_depth = float(self.lidar_utils.min_depth)
        self.max_depth = float(self.lidar_utils.max_depth)

        # STF 根目录与 splits
        self._stf_root = Path("./data/SeeingThroughFog")
        self._split_files = {
            "fog": self._stf_root / "splits/dense_fog_day.txt",
            "snow": self._stf_root / "splits/snow_day.txt",
            "rain": self._stf_root / "splits/rain.txt",
        }

    @functools.lru_cache(maxsize=4)
    def _load_split_names(self, weather: str) -> list[str]:
        split_path = self._split_files[weather]
        names = np.genfromtxt(str(split_path), dtype="U", delimiter="\n").tolist()
        return names

    @staticmethod
    def _name_to_bin(root: Path, name: str) -> Path:
        return (
            root
            / "SeeingThroughFogCompressedExtracted"
            / "lidar_hdl64_strongest"
            / (name.strip().replace(",", "_") + ".bin")
        )

    @functools.lru_cache(maxsize=1024)
    def _load_stf_projected(self, bin_path: str) -> torch.Tensor:
        import numpy as _np

        pts = _np.fromfile(bin_path, dtype=_np.float32).reshape((-1, 5))
        pts = pts[:, :4]  # xyz + reflectance
        xyz = pts[:, :3]
        x = xyz[:, [0]]
        y = xyz[:, [1]]
        z = xyz[:, [2]]
        rflct = pts[:, [3]]
        depth = _np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
        mask = (depth >= self.min_depth) & (depth <= self.max_depth)
        points = _np.concatenate([x, y, z, rflct, depth, mask], axis=1)

        H, W = self.H, self.W
        h_up, h_down = _np.deg2rad(3), _np.deg2rad(-25)
        elevation = _np.arcsin(z / (depth + 1e-8)) + abs(h_down)
        grid_h = 1 - elevation / (h_up - h_down)
        grid_h = _np.floor(grid_h * H).clip(0, H - 1).astype(_np.int32)

        azimuth = -_np.arctan2(y, x)
        grid_w = (azimuth / _np.pi + 1) / 2 % 1
        grid_w = _np.floor(grid_w * W).clip(0, W - 1).astype(_np.int32)
        grid = _np.concatenate((grid_h, grid_w), axis=1)

        order = _np.argsort(-depth.squeeze(1))
        proj = _np.zeros((H, W, 6), dtype=points.dtype)
        for (hh, ww), v in zip(grid[order], points[order]):
            proj[hh, ww] = v
        xyzrdm = proj.transpose(2, 0, 1)
        xyzrdm *= xyzrdm[[5]]
        # STF 中真实的反射强度
        x2 = preprocess_weather_with_rflct(torch.from_numpy(xyzrdm))  # 2xHxW, [-1, 1]
        return x2.float()

    def _sample_target_batch(self, weather: str, batch_size: int) -> torch.Tensor:
        names = self._load_split_names(weather)
        idxs = np.random.randint(0, len(names), size=batch_size)
        xs = []
        for i in idxs:
            bin_path = self._name_to_bin(self._stf_root, names[i])
            xs.append(self._load_stf_projected(str(bin_path)))
        return torch.stack(xs, dim=0)  # [B, 2, H, W]

    @staticmethod
    def _blur(depth: torch.Tensor, kernel_size: int = 3, iters: int = 2) -> torch.Tensor:
        device = depth.device
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=depth.dtype) / float(
            kernel_size * kernel_size
        )
        pad = kernel_size // 2
        x = depth.clone()
        for _ in range(iters):
            # 使用 reflect 边界，更接近 OpenCV filter2D 的默认边界行为（reflect-101）
            x = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel, padding=0)
        return x

    @staticmethod
    def _proj_fill(
        repeat: int,
        proj_w: int,
        px_norm: torch.Tensor,  # 归一化的横坐标
        proj_y: torch.Tensor,
        depth_vals: torch.Tensor,
        proj_range: torch.Tensor,
    ) -> torch.Tensor:
        """
        - 根据给定的低分辨率宽度 proj_w，将归一化横向坐标 px_norm 投影到 [0, proj_w-1]
        - 在形状 [H, proj_w] 的画布上散点赋值（其余位置填 -1）
        - 沿宽度维重复 repeat 次以回到原始宽度
        - 用该重复后的低分辨率图，填充原始 proj_range 中无效（<0）但低分辨率为有效（>0）的位置
        """
        H, W = proj_range.shape
        # 将归一化横坐标映射到低分辨率格子并离散化
        px = torch.floor(px_norm * proj_w).clamp(0, proj_w - 1).to(dtype=torch.long)
        # 低分辨率画布，-1 表示无效
        canvas = torch.full((H, proj_w), -1.0, device=proj_range.device, dtype=proj_range.dtype)
        canvas[proj_y.long(), px] = depth_vals
        # 沿宽度重复
        canvas_rep = canvas.repeat_interleave(repeat, dim=1)
        # 仅在原始无效但低分辨率有效处进行填充
        mask = (proj_range < 0) & (canvas_rep > 0)
        return torch.where(mask, canvas_rep, proj_range)

    def range_depth_scale_deprecated(self, depth_t: torch.Tensor, xyz_s: torch.Tensor, depth_s: torch.Tensor):
        """
        Range-View-Based Point Drop Pattern Transfer

        Procedures:
            1. 以目标域深度图 depth_t 作为 proj_range，-1 表示无效
            2. 进行两次低分辨率投影回填（proj_fill）：
                - 第一次：proj_w = W//2, repeat = 2
                - 第二次：proj_w = W//4, repeat = 4
            3. 对回填后的图做两次 3x3 均值滤波得到 smoothed
            4. 计算 scale = smoothed / filled；对无效处先置 0，然后对 <0.9 或 >1.1 的位置还原为 1.0
            5. 将该缩放场应用到源点云 xyz_s / depth_s 上，并对 |Δd|>0.2m 的像素恢复缩放为 1.0

        Returns:
            Locally Scaled Source Points (xyz coords): B, 3, H, W
        """
        B, _, H, W = depth_t.shape
        device = depth_t.device

        scales = []
        for b in range(B):
            # 原始目标深度图：将无效（<=0）标记为 -1，与 RND 保持一致
            depth_img = depth_t[b, 0].clone()
            # 目标深度图中无效点标记为 -1
            proj_range = torch.where(depth_img > 0, depth_img, torch.full_like(depth_img, -1.0))

            # 从 target depth image 中构造“点集”以用于投影
            valid = proj_range > 0
            if valid.any():  # 至少有一个有效点
                idx = valid.nonzero(as_tuple=False)
                # 这里进行了筛选，因此 idx 的长度 N 可能小于 H*W
                proj_y = idx[:, 0]  # (N,)
                proj_x = idx[:, 1]  # (N,)
                px_norm = proj_x.to(torch.float32) / float(W)
                depth_vals = proj_range[valid]  # 存储有效深度值 (N,)

                # 第一次回填：proj_w = W//2, repeat = 2
                proj_range_filled = self._proj_fill(
                    2, W // 2, px_norm, proj_y, depth_vals, proj_range.clone()
                )
                # 第二次回填：proj_w = W//4, repeat = 4
                proj_range_filled = self._proj_fill(4, W // 4, px_norm, proj_y, depth_vals, proj_range_filled)
            else:
                # 整幅无效，直接保持 -1
                proj_range_filled = proj_range

            # 无效掩码
            mask_invalid = proj_range_filled == -1

            # 两次 3x3 均值滤波
            img = proj_range_filled.unsqueeze(0).unsqueeze(0)  # 1, 1, H, W
            smoothed = self._blur(img, kernel_size=3, iters=1)
            smoothed = self._blur(smoothed, kernel_size=3, iters=1)
            smoothed = smoothed.squeeze(0).squeeze(0)  # H, W

            filled = proj_range_filled
            # 比例
            scale = smoothed / filled
            # 无效处恢复 0，然后对 <0.9 与 >1.1 的位置都置回 1.0 不改变
            scale = torch.where(mask_invalid, torch.zeros_like(scale), scale)
            scale = torch.where(scale < 0.9, torch.ones_like(scale), scale)
            scale = torch.where(scale > 1.1, torch.ones_like(scale), scale)

            scales.append(scale.unsqueeze(0))

        scale_img = torch.stack(scales, dim=0)  # B, 1, H, W

        # 应用到源：先根据深度差约束回退到 1.0
        scaled_depth = depth_s * scale_img
        ok = (scaled_depth - depth_s).abs() <= 0.2
        scale_img = torch.where(ok, scale_img, torch.ones_like(scale_img))

        scaled_xyz = xyz_s * scale_img
        return scaled_xyz, scale_img

    def range_drop_pattern_transfer(self, d_t: torch.Tensor) -> torch.Tensor:
        """
        Range-View-Based Point Drop Pattern Transfer

        Given a target range image depth d_t (B, 1, H, W) in metric units, build a
        drop/keep mask on range view by:
          1) Building a boolean invalid mask from target projection (proj_mask=False)
             here defined as depth in [min_depth, max_depth].
          2) Mean-blurring the invalid mask with a 3x3 kernel (reflect padding),
             approximating OpenCV's filter2D behavior used in RND.
          3) Following RND rule: keep_mask = ~((blurred < 1.0) & target_invalid)
             This maps to selecting points to keep on the source view when sampling
             with (proj_y, proj_x). Since our representation is a dense HxW grid,
             we can directly use keep_mask as (B, 1, H, W) to mask pixels.

        Returns:
            keep_mask (torch.Tensor): (B, 1, H, W) with values in {0., 1.}
        """
        assert d_t.ndim == 4 and d_t.shape[1] == 1, "d_t must be (B, 1, H, W)"
        B, _, H, W = d_t.shape
        device = d_t.device

        keep_masks = []
        for b in range(B):
            # Select depth channel
            depth_img = d_t[b, 0]
            # proj_mask: valid where depth in [min_depth, max_depth]
            proj_mask = (depth_img >= self.min_depth) & (depth_img <= self.max_depth)
            # invalid mask: True where invalid (bool)
            target_invalid = ~proj_mask  # bool tensor

            # Mean blur (3x3) with reflect padding; output in [0, 1]
            blurred = self._blur(target_invalid.float().unsqueeze(0).unsqueeze(0), kernel_size=3, iters=1)
            blurred = blurred.squeeze(0).squeeze(0)

            # Choose isolate invalid points which are caused by adverse weather
            cond = (blurred < 1.0) & target_invalid  # both bool
            # Drop them
            keep_bool = ~cond
            keep_masks.append(keep_bool.to(dtype=torch.float32).unsqueeze(0))  # (1, H, W)

        keep_mask = torch.stack(keep_masks, dim=0)  # (B, 1, H, W)
        return keep_mask

    @staticmethod
    def histogram_matching_vectorized(
        source_ref: torch.Tensor,
        target_ref: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Reflectance Intensity Distribution Transfer
        通过 CDF 匹配恶劣天气的反射强度分布

        Returns:
            Matched Reflectance: B, 1, H, W
        """
        B = source_ref.shape[0]
        out = []
        for b in range(B):
            # 展平
            s = source_ref[b].flatten().cpu().numpy()
            if target_mask is not None:
                tm = target_mask[b].flatten().to(dtype=torch.bool, device=target_ref.device)
                t = target_ref[b].flatten()[tm].cpu().numpy()
            else:
                t = target_ref[b].flatten().cpu().numpy()

            if t.size == 0:
                # 如果目标没有有效值，直接返回源（不做匹配）
                out.append(source_ref[b].clone())
                continue
            # rflct in [0, 1] -> [0, 255]
            s = np.floor(np.clip(s, 0, None) * 255.0)
            t = np.floor(np.clip(t, 0, None) * 255.0)
            # 计算直方图
            # sc: counts, sv: values
            # tc: counts, tv: values
            sv, sc = np.unique(s, return_counts=True)
            tv, tc = np.unique(t, return_counts=True)
            # 计算累积分布函数 CDF
            scdf = np.cumsum(sc).astype(np.float64)
            # 归一化
            scdf /= scdf[-1]
            tcdf = np.cumsum(tc).astype(np.float64)
            tcdf /= tcdf[-1]
            # 计算源域点云每个点的 rflct 值在源域 CDF 上的值
            s_val2cdf = np.interp(s, sv, scdf, left=scdf[0], right=scdf[-1])
            # 反查在目标域 CDF 上对应的 rflct 值
            t_val = np.interp(s_val2cdf, tcdf, tv, left=tv[0], right=tv[-1])
            # 先在 numpy 侧根据源的形状重塑，再转回 torch，并放回原 device
            arr = (t_val / 255.0).astype(np.float32).reshape(tuple(source_ref[b].shape))
            matched = torch.from_numpy(arr).to(device=source_ref.device)
            out.append(matched)
        return torch.stack(out, dim=0)

    def density_matching_vectorized(self, xyz_s: torch.Tensor, xyz_t: torch.Tensor) -> torch.Tensor:
        """
        Distance-Aware Density Distribution Transfer

        Returns:
            Drop Mask: B, 1, H, W
        """
        B = xyz_s.shape[0]
        keep_masks = []
        bins = 100
        # 仅在有效深度范围内进行统计，避免无效像素挤占第一个 bin 造成整圈伪影
        min_d = float(self.min_depth)
        max_d = float(self.max_depth)
        H, W = self.H, self.W
        for b in range(B):
            # 计算源/目标的欧氏深度并拆分出有效像素
            ds_all = torch.linalg.norm(xyz_s[b], dim=0)  # H, W
            dt_all = torch.linalg.norm(xyz_t[b], dim=0)  # H, W

            valid_s = (ds_all >= min_d) & (ds_all <= max_d)
            valid_t = (dt_all >= min_d) & (dt_all <= max_d)

            ds = ds_all[valid_s].flatten().cpu().numpy()
            dt = dt_all[valid_t].flatten().cpu().numpy()

            if ds.size == 0:
                # 没有有效源像素，直接置零掩码
                keep_masks.append(torch.zeros((1, H, W), dtype=torch.float32, device=xyz_s.device))
                continue

            # 计算直方图（在有效范围内）
            hist_s, edges = np.histogram(ds, bins=bins, range=(min_d, max_d))
            hist_t, _ = np.histogram(dt, bins=edges)
            hist_s = hist_s.astype(float) + 1e-8
            hist_t = hist_t.astype(float) + 1e-8

            # 计算每个 bin 的保留概率
            ratio = np.where(hist_s > hist_t, hist_t / hist_s, 1.0)

            # 为每个源像素计算保留概率（无效像素直接置 0 不参与）
            idx = np.digitize(ds_all.cpu().numpy(), edges) - 1  # HxW -> bin id
            idx = np.clip(idx, 0, len(ratio) - 1)
            probs = ratio[idx]

            # 采样得到保留掩码，仅对有效处采样，其余无效处仍为 0
            rnd = np.random.rand(H, W)
            keep = (rnd < probs).astype(np.float32)
            keep = np.where(valid_s.cpu().numpy(), keep, 0.0)

            keep_masks.append(torch.from_numpy(keep).unsqueeze(0))

        # 保持设备一致
        out = torch.stack(keep_masks, dim=0).to(device=xyz_s.device, dtype=torch.float32)
        return out

    def forward(
        self, x_normal: torch.Tensor, weather_flag: str, xyz_normal: torch.Tensor, depth_normal: torch.Tensor
    ) -> torch.Tensor:
        """
        SWG forward 函数，应用 3 个天气模拟方法

        Args:
            x_normal (torch.Tensor): B, 2, 64, 1024 [-1, 1]
            weather_flag (str): "fog", "snow", "rain", "wet_ground", "normal"
            xyz_normal (torch.Tensor): B, 3, 64, 1024
            depth_normal (torch.Tensor): B, 1, 64, 1024

        Returns:
            torch.Tensor: B, 2, 64, 1024
        """
        device = x_normal.device
        B, C, H, W = x_normal.shape  # B, 2 (d, rflct), 64, 1024
        assert (H, W) == (self.H, self.W)

        # 源域反归一化
        x_denorm = self.lidar_utils.denormalize(x_normal)  # d, rflct in [0, 1]
        d_s_metric = self.lidar_utils.revert_depth(x_denorm[:, [0]])  # d in [0, max_depth]
        r_s = x_denorm[:, [1]] if C > 1 else None  # rflct in [0, 1]

        # 目标域反归一化
        x_t = self._sample_target_batch(weather_flag, B).to(device=device)
        x_t_denorm = self.lidar_utils.denormalize(x_t)  # d, rflct in [0, 1]
        d_t_metric = self.lidar_utils.revert_depth(x_t_denorm[:, [0]])  # d in [0, max_depth]
        r_t = x_t_denorm[:, [1]] if C > 1 else None  # rflct in [0, 1]

        # 源域 xyz
        xyz_src = xyz_normal.to(device)
        d_src_from_xyz = torch.linalg.norm(xyz_src, dim=1, keepdim=True)

        # 1. Reflectance Intensity Distribution Transfer
        if (r_s is not None) and (r_t is not None):
            dist_mask = (d_t_metric > 1.6) & (d_t_metric <= self.max_depth)
            r_s2 = self.histogram_matching_vectorized(r_s, r_t, target_mask=dist_mask)
        else:
            r_s2 = None

        # 2. Range-View-Based Point Drop Pattern Transfer
        mask_rv = self.range_drop_pattern_transfer(d_t_metric)  # (B, 1, H, W) in {0, 1}
        keep_mask = mask_rv

        # 3. Distance-Aware Density Distribution Transfer
        mask_density = self.density_matching_vectorized(
            xyz_src, self.lidar_utils.to_xyz(d_t_metric)
        )  # (B, 1, H, W)
        keep_mask = keep_mask * mask_density

        # Apply mask
        d_out = d_src_from_xyz * keep_mask
        if r_s2 is not None:
            r_out = r_s2 * keep_mask

        # Postprocess
        x_depth = self.lidar_utils.normalize(self.lidar_utils.convert_depth(d_out))
        outs = [x_depth]
        if r_s2 is not None:
            outs.append(self.lidar_utils.normalize(r_out))
        x_0 = torch.cat(outs, dim=1)

        # Final depth validity mask
        valid_depth_mask = (d_out >= self.min_depth) & (d_out <= self.max_depth)
        x_0 = x_0 * valid_depth_mask.to(dtype=x_0.dtype, device=x_0.device)
        return x_0


def weather_process_swg(
    x_normal: torch.Tensor, weather_flag: str, xyz_normal: torch.Tensor, depth_normal: torch.Tensor
) -> torch.Tensor:
    """
    Replace the original weather_process function in train.py directly
    """
    if weather_flag == "normal":
        return x_normal

    # Lazily initialize and cache the generator to avoid frequent construction
    if not hasattr(weather_process_swg, "_generator"):
        weather_process_swg._generator = StatisticsBasedWeatherGenerator(lidar_utils)
    gen: StatisticsBasedWeatherGenerator = weather_process_swg._generator
    return gen(x_normal, weather_flag, xyz_normal, depth_normal)


# =========================
# Raw-point version of SWG
# =========================


def _read_points_bin(bin_path: str) -> np.ndarray:
    """
    Read .bin file as Nx{4 | 5} float32 array.
    Supported:
      - Nx4: [x, y, z, reflectance]
      - Nx5: [x, y, z, reflectance, extra] (e.g., STF)
    Returns Nx4 with reflectance normalized to [0, 1].
    """
    arr = np.fromfile(bin_path, dtype=np.float32)
    if arr.size % 5 == 0:
        pts = arr.reshape((-1, 5))[:, :4]
    elif arr.size % 4 == 0:
        pts = arr.reshape((-1, 4))
    else:
        raise ValueError(f"Unexpected array size {arr.size} for {bin_path}; not divisible by 4 or 5.")

    return pts


def _project_raw_to_image(raw: np.ndarray, H: int, W: int, min_depth: float, max_depth: float):
    """
    Project raw points to range image with near-point overwrite.

    Returns:
      depth_img (H, W): metric depth; 0 means invalid
      rflct_img (H, W): reflectance in [0, 1] approx (clipped later)
      proj_y (N,), proj_x (N,), unproj_depth (N,) for the input points
    """
    xyz = raw[:, :3]
    r = raw[:, 3:4]
    x = xyz[:, 0:1]
    y = xyz[:, 1:2]
    z = xyz[:, 2:3]
    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    valid = (depth >= min_depth) & (depth <= max_depth)

    # Vertical index
    h_up, h_down = np.deg2rad(3), np.deg2rad(-25)
    elevation = np.arcsin(z / (depth + 1e-8)) + abs(h_down)
    # Compute y coords for all (N,) points
    grid_h = 1 - elevation / (h_up - h_down)
    grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)

    # Horizontal index
    azimuth = -np.arctan2(y, x)  # [-pi, pi]
    # Compute x coords for all (N,) points
    grid_w = (azimuth / np.pi + 1) / 2 % 1
    grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)

    # Prepare depth & reflectance images
    depth_img = np.zeros((H, W), dtype=np.float32)
    rflct_img = np.zeros((H, W), dtype=np.float32)

    # Order by -depth so nearer points overwrite farther ones
    order = np.argsort(-depth.squeeze(1))
    gh = grid_h[order].squeeze(1)
    gw = grid_w[order].squeeze(1)
    dep_ord = depth[order].squeeze(1)
    r_ord = r[order].squeeze(1)
    val_ord = valid[order].squeeze(1)

    # Assign only valid
    ghv = gh[val_ord]
    gwv = gw[val_ord]
    depth_img[ghv, gwv] = dep_ord[val_ord]
    rflct_img[ghv, gwv] = r_ord[val_ord]
    # Ensure reflectance within [0, 1]
    if rflct_img.size > 0:
        rflct_img = np.clip(rflct_img, 0.0, 1.0)

    # proj_y: grid_h, proj_x: grid_w, unproj_depth: depth
    return depth_img, rflct_img, grid_h.squeeze(1), grid_w.squeeze(1), depth.squeeze(1)


# Reflectance Intensity Distribution Transfer
def _histogram_reflectance_matching(src_r: np.ndarray, tgt_r: np.ndarray) -> np.ndarray:
    """
    Match histogram of source reflectance to target reflectance using 256 bins.
    src_r, tgt_r: (N,), real-valued; returns matched (N,) in [0,1].
    """
    if tgt_r.size == 0:
        return np.clip(src_r, 0.0, 1.0)
    s = np.floor(np.clip(src_r, 0.0, None) * 255.0)
    t = np.floor(np.clip(tgt_r, 0.0, None) * 255.0)
    sv, sc = np.unique(s, return_counts=True)
    tv, tc = np.unique(t, return_counts=True)
    scdf = np.cumsum(sc).astype(np.float64)
    scdf /= scdf[-1]
    tcdf = np.cumsum(tc).astype(np.float64)
    tcdf /= tcdf[-1]
    # map source value -> cdf -> target value
    s_val2cdf = np.interp(s, sv, scdf, left=scdf[0], right=scdf[-1])
    t_val = np.interp(s_val2cdf, tcdf, tv, left=tv[0], right=tv[-1])
    return (t_val / 255.0).astype(np.float32)


# Range-View-Based Point Drop Pattern Transfer
def _range_view_drop_mask(depth_img_t: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """
    Build range view drop mask (H, W) from target depth image (metric depth, 0 invalid).
    keep = ~((blurred_invalid < 1.0) & target_invalid)
    """
    H, W = depth_img_t.shape
    proj_mask = (depth_img_t >= min_depth) & (depth_img_t <= max_depth)
    # Choose
    target_invalid = ~proj_mask
    inv_float = target_invalid.astype(np.float32)
    # 3x3 mean blur with reflect-like padding replicated by cv2 equivalent using np.pad + conv
    k = np.ones((3, 3), dtype=np.float32) / 9.0
    pad = 1
    inv_pad = np.pad(inv_float, ((pad, pad), (pad, pad)), mode="reflect")
    # convolution
    blurred = (
        inv_pad[:-2, :-2]
        + inv_pad[:-2, 1:-1]
        + inv_pad[:-2, 2:]
        + inv_pad[1:-1, :-2]
        + inv_pad[1:-1, 1:-1]
        + inv_pad[1:-1, 2:]
        + inv_pad[2:, :-2]
        + inv_pad[2:, 1:-1]
        + inv_pad[2:, 2:]
    ) / 9.0
    # Choose invalid points caused by adverse weather, where surrounding is not all invalid (blurred < 1.0)
    cond = (blurred < 1.0) & target_invalid
    keep = ~cond
    return keep.astype(np.bool_)


# Distance-aware Density Distribution Transfer
def _density_matching(
    source_xyz: np.ndarray, target_xyz: np.ndarray, min_depth: float, max_depth: float
) -> np.ndarray:
    """
    Compute density mask for source points to match target depth distribution.
    Returns boolean mask (N_src,).
    """
    s_depth = np.linalg.norm(source_xyz[:, :3], 2, axis=1)
    t_depth = np.linalg.norm(target_xyz[:, :3], 2, axis=1)
    bins = 100
    hist_s, edges = np.histogram(
        np.clip(s_depth, min_depth, max_depth), bins=bins, range=(min_depth, max_depth)
    )
    hist_t, _ = np.histogram(np.clip(t_depth, min_depth, max_depth), bins=edges)
    hist_s = hist_s.astype(float) + 1e-8
    hist_t = hist_t.astype(float) + 1e-8
    ratio = np.where(hist_s > hist_t, (hist_t / hist_s), 1.0)
    bin_indices = np.digitize(s_depth, edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(ratio) - 1)
    out_of = (s_depth < min_depth) | (s_depth > max_depth)
    probs = np.where(out_of, 1.0, ratio[bin_indices])
    keep = np.random.rand(len(s_depth)) < probs
    return keep


def weather_process_swg_raw(
    src_bin_paths: list[str],
    weather_flag: str,
    lidar_utils: LiDARUtility,
    include_reflectance: bool = True,
    resolution: tuple[int, int] = (64, 1024),
) -> torch.Tensor:
    """
    SWG operating on raw point clouds, then re-project to range image.

    Returns: torch.Tensor [B, C, H, W] in [-1, 1]
    """
    assert weather_flag in {"fog", "snow", "rain"}, "Only adverse weather is expected here"
    H, W = int(resolution[0]), int(resolution[1])
    min_d = float(lidar_utils.min_depth)
    max_d = float(lidar_utils.max_depth)

    # Sample B target files from STF splits
    stf_root = Path("./data/SeeingThroughFog")
    split_map = {
        "fog": "splits/dense_fog_day.txt",
        "snow": "splits/snow_day.txt",
        "rain": "splits/rain.txt",
    }
    names = np.genfromtxt(str(stf_root / split_map[weather_flag]), dtype="U", delimiter="\n").tolist()
    tgt_paths = [
        str(
            stf_root
            / "SeeingThroughFogCompressedExtracted/lidar_hdl64_strongest"
            / (names[np.random.randint(0, len(names))].strip().replace(",", "_") + ".bin")
        )
        for _ in src_bin_paths
    ]

    outs = []
    for src_path, tgt_path in zip(src_bin_paths, tgt_paths):
        # Load raw clouds
        src_pts = _read_points_bin(src_path)
        tgt_pts = _read_points_bin(tgt_path)

        # 1. Reflectance Intensity Distribution Transfer (raw points)
        if include_reflectance:
            src_r = src_pts[:, 3]
            tgt_r = tgt_pts[:, 3]
            src_pts[:, 3] = _histogram_reflectance_matching(src_r, tgt_r)

        # Re-project source after reflectance change (depth unchanged)
        depth_s_img, r_s_img, proj_y_s, proj_x_s, unproj_d_s = _project_raw_to_image(
            src_pts, H, W, min_d, max_d
        )

        # 2. Range-View-Based Point Drop Pattern Transfer
        depth_t_img, _, _, _, _ = _project_raw_to_image(tgt_pts, H, W, min_d, max_d)
        keep_mask = _range_view_drop_mask(depth_t_img, min_d, max_d)
        # Apply drop mask to source points
        keep_src_rv = keep_mask[proj_y_s, proj_x_s]
        src_pts = src_pts[keep_src_rv]

        # Re-project after range view drop
        depth_s_img, r_s_img, proj_y_s, proj_x_s, unproj_d_s = _project_raw_to_image(
            src_pts, H, W, min_d, max_d
        )

        # 3. Distance-Aware Density Distribution Transfer
        keep_density_mask = _density_matching(src_pts[:, :3], tgt_pts[:, :3], 0.0, 100.0)
        src_pts = src_pts[keep_density_mask]

        # Final projection to image
        depth_s_img, r_s_img, _, _, _ = _project_raw_to_image(src_pts, H, W, min_d, max_d)

        # Build tensor output channels
        depth_img_metric = torch.from_numpy(depth_s_img).unsqueeze(0)  # 1, H, W
        depth_mask = (depth_img_metric >= min_d) & (depth_img_metric <= max_d)  # bool: 1, H, W
        x_depth = lidar_utils.normalize(lidar_utils.convert_depth(depth_img_metric))  # [-1, 1]
        outs_ch = [x_depth]
        if include_reflectance:
            r_img = torch.from_numpy(np.clip(r_s_img, 0.0, 1.0)).unsqueeze(0)
            r_img = lidar_utils.normalize(r_img)  # [0, 1] -> [-1, 1]
            outs_ch.append(r_img)
        x_hw = torch.cat(outs_ch, dim=0)  # C, H, W
        x_hw = x_hw * depth_mask.to(dtype=x_hw.dtype, device=x_hw.device)
        outs.append(x_hw)

    x_0 = torch.stack(outs, dim=0)  # B, C, H, W
    return x_0
