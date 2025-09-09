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


# def weather_process(
#     x_normal: torch.Tensor, weather_flag: str, xyz_normal: torch.Tensor, depth_normal: torch.Tensor
# ):
#     """
#     向量化 & GPU 加速版 MDP，不改变原有掩码/噪声语义，但移除了 Python/NumPy 循环与重投影。
#     输入均为张量：
#     - x_normal: (B, 2, H, W), 已标准化 [-1,1]
#     - xyz_normal: (B, 3, H, W), metric XYZ
#     - depth_normal: (B, 1, H, W), metric depth
#     返回：x_0 (B, 2, H, W)
#     """
#     device = x_normal.device
#     B, C, H, W = x_normal.shape
#     ones = torch.ones(B, 1, H, W, device=device)
#     zeros = torch.zeros(B, 1, H, W, device=device)

#     # 归一化深度到 [0,1]，保持与原实现的比较阈值一致
#     depth01 = (depth_normal / lidar_utils.max_depth).clamp(0, 1)

#     if weather_flag == "normal":
#         return x_normal

#     if weather_flag == "rain":
#         height = xyz_normal[:, [2]]  # (B,1,H,W)
#         # 原实现：每个样本独立，每行独立采样 (H,1) 维度的伯努利掩码
#         mask_mask_1 = torch.empty(B, 1, H, 1, device=device).bernoulli_(0.3)
#         mask_mask_2 = torch.empty(B, 1, H, 1, device=device).bernoulli_(0.2)
#         mask_mask_3 = torch.empty(B, 1, H, 1, device=device).bernoulli_(0.5)
#         mask_rain = torch.where(depth01 < 0.2, ones, zeros)
#         mask_1 = torch.where(height > -1.2, ones, zeros) * mask_mask_1 * mask_rain
#         mask_2 = torch.where(depth01 > 0.08, ones, zeros) * mask_mask_3 * mask_mask_2
#         mask_3 = torch.where(depth01 < 0.08, ones, zeros) * mask_mask_1
#         mask = (mask_1 + mask_2 + mask_3).clamp_max_(1.0)
#         return mask * x_normal + (1 - mask) * -1

#     if weather_flag == "wet_ground":
#         height = xyz_normal[:, [2]]
#         # 原实现：按行采样 (H,1) 和全像素采样 (H,W)
#         mask_mask_1 = torch.empty(B, 1, H, 1, device=device).bernoulli_(0.5)
#         mask_mask_2 = torch.empty(B, 1, H, W, device=device).bernoulli_(0.2)
#         mask_1 = torch.where(height > -1.2, ones, zeros)
#         mask_2 = torch.where(depth01 > 0.06, zeros, ones) * mask_mask_1 * mask_mask_2
#         mask = (mask_1 + mask_2).clamp_max_(1.0)
#         return mask * x_normal + (1 - mask) * -1

#     # 下面两个天气在原实现中包含更多随机噪声与衰减；保持阈值/概率与通道操作一致
#     if weather_flag == "fog":
#         # 随机噪声强度 0.005，深度<0.13 区域根据 0.8 伯努利掩码添加
#         mask_mask_noise = torch.empty(B, 1, H, W, device=device).bernoulli_(0.8)
#         noise = (depth01 < 0.13).float() * 0.005 * mask_mask_noise

#         # 衰减：深度>0.03 外层以 0.8 伯努利保留，<=0.03 全保留
#         mask_mask_atten = torch.empty(B, 1, H, W, device=device).bernoulli_(0.8)
#         fog_atten_out = torch.where(depth01 > 0.03, ones, zeros) * mask_mask_atten
#         fog_atten_in = torch.where(depth01 < 0.03, ones, zeros)
#         mask_atten = (fog_atten_out + fog_atten_in).clamp_max_(1.0)
#         xs = mask_atten * x_normal + (1 - mask_atten) * -1

#         # 深度<=0.14 全保留，其余区域以 0.3 & 0.55 的伯努利随机保留
#         mask_mask_2 = torch.empty(B, 1, H, W, device=device).bernoulli_(0.3)
#         mask_mask_3 = torch.empty(B, 1, H, 1, device=device).bernoulli_(0.55)
#         mask_1 = torch.where(depth01 > 0.14, zeros, ones)
#         mask_2 = torch.where(depth01 > 0.14, ones, zeros) * mask_mask_2 * mask_mask_3
#         mask_keep = (mask_1 + mask_2).clamp_max_(1.0)

#         # 仅对 depth 通道加噪
#         xs = xs.clone()
#         xs[:, [0]] = xs[:, [0]] + noise
#         xs = mask_keep * xs + (1 - mask_keep) * -1
#         return xs

#     if weather_flag == "snow":
#         # 原实现：mask_mask_1 为 (H,W)，mask_1 为 (H,1) 然后相乘
#         mask_mask_1 = torch.empty(B, 1, H, W, device=device).bernoulli_(0.8)
#         mask_1_col = torch.empty(B, 1, H, 1, device=device).bernoulli_(0.9)
#         noise = (depth01 < 0.13).float() * 0.005 * mask_mask_1
#         xs = x_normal.clone()
#         xs[:, [0]] = xs[:, [0]] + noise
#         mask = (mask_1_col * mask_mask_1).clamp_max_(1.0)
#         xs = mask * xs + (1 - mask) * -1
#         return xs

#     # 兜底：未知天气直接返回原图
#     return x_normal


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
        all_point_path = np.genfromtxt("/./seeingthroughfog/rain.txt", dtype="U", delimiter="\n")
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
