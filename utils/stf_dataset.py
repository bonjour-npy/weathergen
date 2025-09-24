from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import numba

# 依赖现有的预处理逻辑与 LiDAR 参数，保持与 utils/weather.py 一致
from .weather import preprocess_weather


def _name_to_bin(root: Path, name: str) -> Path:
    # 原始 split 中的名称形如 '2018-02-03_20-58-04,00400'，真实文件名用下划线连接
    return (
        root
        / "SeeingThroughFogCompressedExtracted"
        / "lidar_hdl64_strongest"
        / (name.strip().replace(",", "_") + ".bin")
    )


@lru_cache(maxsize=512)
def _load_and_project_cached(bin_path: str, H: int, W: int) -> np.ndarray:
    """
    读取一个 STF .bin 点云并投影为 (6, H, W) 的 xyzrdm, 再经 preprocess_weather
    返回标准训练用的 2xHxW (depth、reflectance 归一化后再 [-1,1]) 的 numpy 数组
    说明: 缓存作用域为进程内 (DataLoader worker 内部) 避免重复 I/O 与投影
    """
    path = Path(bin_path)
    points = np.fromfile(path, dtype=np.float32).reshape((-1, 5))
    points = points[:, :4]  # xyz + reflectance
    xyz = points[:, :3]
    x = xyz[:, [0]]
    y = xyz[:, [1]]
    z = xyz[:, [2]]

    depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
    mask = (depth >= 1.45) & (depth <= 80.0)
    pts = np.concatenate([x, y, z, points[:, [3]], depth, mask], axis=1)  # [x, y, z, ref, depth, mask]

    # elevation/azimuth 到 range image 网格
    h_up, h_down = np.deg2rad(3), np.deg2rad(-25)
    elevation = np.arcsin(z / (depth + 1e-12)) + abs(h_down)
    grid_h = 1 - elevation / (h_up - h_down)
    grid_h = np.floor(grid_h * H).clip(0, H - 1).astype(np.int32)

    azimuth = -np.arctan2(y, x)  # [-pi, pi]
    grid_w = (azimuth / np.pi + 1) / 2 % 1
    grid_w = np.floor(grid_w * W).clip(0, W - 1).astype(np.int32)

    grid = np.concatenate((grid_h, grid_w), axis=1)  # (N, 2)

    # 近点覆盖远点：按 -depth 排序，让近点后写覆盖
    order = np.argsort(-depth.squeeze(1))
    proj = np.zeros((H, W, 6), dtype=pts.dtype)
    proj = _scatter(proj, grid[order], pts[order])

    xyzrdm = proj.transpose(2, 0, 1)  # (6, H, W)
    xyzrdm *= xyzrdm[[5]]  # 应用深度掩码

    # 转换为训练输入 depth / reflectance -> 2 x H x W, [-1, 1]
    x = preprocess_weather(torch.from_numpy(xyzrdm))  # torch.Tensor, CPU
    return x.numpy()


@numba.jit(nopython=True, parallel=False)
def _scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array


class STFDataset(Dataset):
    """
    Seeing Through Fog 投影数据集 (按需在线投影 + 进程内 LRU 缓存)

    返回: Tensor[2, H, W], 为 depth 与 reflectance 归一化后再标准化到 [-1, 1]
    """

    def __init__(
        self,
        root: str | Path = "./data/SeeingThroughFog",
        weather: Literal["fog", "snow", "rain"] = "fog",
        resolution: Sequence[int] = (64, 1024),
        split_file_map: dict[str, str] | None = None,
        return_dict: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.weather = weather
        self.H, self.W = int(resolution[0]), int(resolution[1])
        self.return_dict = return_dict

        # 默认 splits 文件路径
        if split_file_map is None:
            split_file_map = {
                "fog": "splits/dense_fog_day.txt",
                "snow": "splits/snow_day.txt",
                "rain": "splits/rain.txt",
            }

        split_path = self.root / split_file_map[weather]
        # np.genfromtxt: 只在构造时读取一次
        self.names: list[str] = np.genfromtxt(str(split_path), dtype="U", delimiter="\n").tolist()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> torch.Tensor:
        name = self.names[index]
        bin_path = _name_to_bin(self.root, name)
        x_np = _load_and_project_cached(str(bin_path), self.H, self.W)
        x = torch.from_numpy(x_np).float()  # [2, H, W]
        if self.return_dict:
            return {
                "x": x,  # [-1, 1], 2xHxW
                "file_path": str(bin_path),
                "name": name,
            }
        return x


def build_stf_loader(
    weather: Literal["fog", "snow", "rain"],
    batch_size: int,
    num_workers: int,
    resolution: Sequence[int] = (64, 1024),
    root: str | Path = "./data/SeeingThroughFog",
    drop_last: bool = True,
):
    from torch.utils.data import DataLoader

    ds = STFDataset(root=root, weather=weather, resolution=resolution)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    # 仅在 num_workers>0 时可设置 prefetch_factor
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def build_stf_loader_dict(
    weather: Literal["fog", "snow", "rain"],
    batch_size: int,
    num_workers: int,
    resolution: Sequence[int] = (64, 1024),
    root: str | Path = "./data/SeeingThroughFog",
    drop_last: bool = True,
):
    from torch.utils.data import DataLoader

    ds = STFDataset(root=root, weather=weather, resolution=resolution, return_dict=True)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)
