"""
由于 stf 数据集加载的问题修改较多
"""

import argparse
import datetime
import json
import pickle
import random
from pathlib import Path
from torch import nn
from typing import Literal

import numpy as np
import datasets as ds
import einops
import torch
import torch.nn.functional as F
from rich import print
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.stf_dataset import build_stf_loader

import utils.inference
from metrics import bev, distribution
from metrics.extractor import pointnet, rangenet

# from LiDARGen
MAX_DEPTH = 63.0
MIN_DEPTH = 0.5
DATASET_MAX_DEPTH = 80.0


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
        # return mask


lidar_utils = LiDARUtility(
    resolution=(64, 1024),
    image_format="log_depth",
    min_depth=1.45,
    max_depth=80.0,
    ray_angles=None,
)


def preprocess(xyzrdm):
    x = []
    x += [xyzrdm[4].unsqueeze(0)]
    x += [xyzrdm[0].unsqueeze(0)]
    x += [xyzrdm[1].unsqueeze(0)]
    x += [xyzrdm[2].unsqueeze(0)]
    x += [lidar_utils.convert_depth(xyzrdm[[3]])]
    x = torch.cat(x, dim=0)
    return x


def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array


def resize(x, size):
    return F.interpolate(x, size=size, mode="nearest-exact")


class Samples(torch.utils.data.Dataset):
    def __init__(self, root, helper):
        self.sample_path_list = sorted(Path(root).glob("*.pth"))[:200]
        self.helper = helper

    def __getitem__(self, index):
        img = torch.load(self.sample_path_list[index], map_location="cpu")
        assert img.shape[0] == 5, img.shape
        depth = img[[0]]
        mask = torch.logical_and(depth > MIN_DEPTH, depth < MAX_DEPTH).float()
        img = img * mask
        return img.float(), mask.float()

    def __len__(self):
        return len(self.sample_path_list)


def stf_batch_to_5ch(batch_2ch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """将 STFDataset 返回的 2xHxW 批次转换为评估所需的 5xHxW（depth, x, y, z, reflectance）。"""
    batch_2ch = batch_2ch.to(device)
    # [-1,1] -> [0,1]
    depth_norm01 = lidar_utils.denormalize(batch_2ch[:, [0]])
    # 还原为 metric depth
    depth_metric = lidar_utils.revert_depth(depth_norm01)
    # 从 metric depth 恢复 xyz
    xyz = lidar_utils.to_xyz(depth_metric)
    # reflectance 已在 STFDataset 中经 [0,1] 归一化，并缩放到 [-1,1]
    reflect01 = lidar_utils.denormalize(batch_2ch[:, [1]])
    imgs_frd = torch.cat([depth_metric, xyz, reflect01], dim=1)
    return imgs_frd


@torch.no_grad()
def evaluate(args):
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, lidar_utils, cfg = utils.inference.setup_model(args.ckpt, device=device)
    lidar_utils.to(device)

    H, W = lidar_utils.resolution
    extract_img_feats, preprocess_img = rangenet.rangenet53(
        weights=f"SemanticKITTI_{H}x{W}",
        device=device,
        compile=True,
    )
    extract_pts_feats = pointnet.pretrained_pointnet(
        dataset="shapenet",
        device=device,
        compile=True,
    )

    results = dict(img=dict(), pts=dict(), bev=dict(), info=dict())
    results["info"]["phase"] = args.dataset
    results["info"]["directory"] = args.sample_dir

    # =====================================================
    # real set
    # =====================================================

    real_set = dict(img_feats=list(), pts_feats=list(), bev_hists=list())

    # 使用 STFDataset 加载 STF 实际样本
    stf_loader = build_stf_loader(
        weather="snow", batch_size=args.batch_size, num_workers=args.num_workers, resolution=(64, 1024)
    )
    stf_iter = iter(stf_loader)
    for _ in range(12):
        try:
            batch_2ch = next(stf_iter)
        except StopIteration:
            stf_iter = iter(stf_loader)
            batch_2ch = next(stf_iter)
        imgs_frd = stf_batch_to_5ch(batch_2ch, device)
        x = imgs_frd[:, 1].unsqueeze(1)
        y = imgs_frd[:, 2].unsqueeze(1)
        z = imgs_frd[:, 3].unsqueeze(1)
        xyz = torch.cat([x, y, z], dim=1)
        depth = imgs_frd[:, 0].unsqueeze(1)
        mask = torch.logical_and(depth > MIN_DEPTH, depth < MAX_DEPTH)

        with torch.inference_mode():
            feats_img = extract_img_feats(preprocess_img(imgs_frd, mask), feature="lidargen")
        real_set["img_feats"].append(feats_img.cpu())

        point_clouds = einops.rearrange(xyz * mask, "B C H W -> B C (H W)")
        for point_cloud in point_clouds:
            point_cloud = einops.rearrange(point_cloud, "C N -> N C")
            hist = bev.point_cloud_to_histogram(point_cloud)
            real_set["bev_hists"].append(hist.cpu())

        with torch.inference_mode():
            feats_pts = extract_pts_feats(point_clouds / DATASET_MAX_DEPTH)
        real_set["pts_feats"].append(feats_pts.cpu())

    real_set["img_feats"] = torch.cat(real_set["img_feats"], dim=0).numpy()
    real_set["pts_feats"] = torch.cat(real_set["pts_feats"], dim=0).numpy()
    real_set["bev_hists"] = torch.stack(real_set["bev_hists"], dim=0).numpy()
    # pickle.dump(real_set, open(cache_file_path, "wb"))
    results["info"]["#real"] = len(real_set["pts_feats"])

    # =====================================================
    # gen set
    # =====================================================

    gen_loader = DataLoader(
        dataset=Samples(args.sample_dir, helper=lidar_utils.cpu()),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    gen_set = dict(img_feats=list(), pts_feats=list(), bev_hists=list())

    for imgs_frd, mask in tqdm(gen_loader, desc="gen"):
        imgs_frd, mask = imgs_frd.to(device), mask.to(device)
        if cfg.data["train_reflectance"]:
            with torch.inference_mode():
                feats_img = extract_img_feats(preprocess_img(imgs_frd, mask), feature="lidargen")
            gen_set["img_feats"].append(feats_img.cpu())

        xyz = imgs_frd[:, 1:4]
        point_clouds = einops.rearrange(xyz * mask, "B C H W -> B C (H W)")
        for point_cloud in point_clouds:
            point_cloud = einops.rearrange(point_cloud, "C N -> N C")
            hist = bev.point_cloud_to_histogram(point_cloud)
            gen_set["bev_hists"].append(hist.cpu())

        with torch.inference_mode():
            feats_pts = extract_pts_feats(point_clouds / DATASET_MAX_DEPTH)
        gen_set["pts_feats"].append(feats_pts.cpu())

    if cfg.data["train_reflectance"]:
        gen_set["img_feats"] = torch.cat(gen_set["img_feats"], dim=0).numpy()
    gen_set["pts_feats"] = torch.cat(gen_set["pts_feats"], dim=0).numpy()
    gen_set["bev_hists"] = torch.stack(gen_set["bev_hists"], dim=0).numpy()

    results["info"]["#fake"] = len(gen_set["pts_feats"])

    # =====================================================
    # evaluation
    # =====================================================
    torch.cuda.empty_cache()

    if cfg.data["train_reflectance"]:
        results["img"]["frechet_distance"] = distribution.compute_frechet_distance(
            real_set["img_feats"], gen_set["img_feats"]
        )
        results["img"]["squared_mmd"] = distribution.compute_squared_mmd(
            real_set["img_feats"], gen_set["img_feats"]
        )

    results["pts"]["frechet_distance"] = distribution.compute_frechet_distance(
        real_set["pts_feats"], gen_set["pts_feats"]
    )
    results["pts"]["squared_mmd"] = distribution.compute_squared_mmd(
        real_set["pts_feats"], gen_set["pts_feats"]
    )

    perm = list(range(len(real_set["bev_hists"])))
    random.Random(0).shuffle(perm)
    perm = perm[:200]

    results["bev"]["jsd"] = bev.compute_jsd_2d(
        torch.from_numpy(real_set["bev_hists"][perm]).to(device).float(),
        torch.from_numpy(gen_set["bev_hists"]).to(device).float(),
    )

    results["bev"]["mmd"] = bev.compute_mmd_2d(
        torch.from_numpy(real_set["bev_hists"][perm]).to(device).float(),
        torch.from_numpy(gen_set["bev_hists"]).to(device).float(),
    )

    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default="/path_to/diffusion_steps.pth")
    parser.add_argument("--sample_dir", type=str, default="/path_to/results")
    parser.add_argument("--dataset", choices=["train", "test", "all"], default="all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    evaluate(args)
