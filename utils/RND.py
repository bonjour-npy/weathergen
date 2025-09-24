import os
import glob
import json
import cv2
import torch
from scipy import interpolate
import numpy as np
import random
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict

from .builder import DATASETS, build_dataset
from .transform import Compose, TRANSFORMS


def index_operator(data_dict, index, duplicate=False):
    # index selection operator for keys in "index_valid_keys"
    # custom these keys by "Update" transform in config
    if "index_valid_keys" not in data_dict:
        data_dict["index_valid_keys"] = [
            "coord",
            "color",
            "normal",
            "strength",
            "segment",
            "instance",
        ]
    if not duplicate:
        for key in data_dict["index_valid_keys"]:
            if key in data_dict:
                data_dict[key] = data_dict[key][index]
        return data_dict
    else:
        data_dict_ = dict()
        for key in data_dict.keys():
            if key in data_dict["index_valid_keys"]:
                data_dict_[key] = data_dict[key][index]
            else:
                data_dict_[key] = data_dict[key]
        return data_dict_


@DATASETS.register_module("SimDataset")
class SimDataset(Dataset):
    def __init__(
        self,
        source_dataset,
        target_dataset,
        output_dict="target",
        few_shot=None,
        sim_noise=True,
        sim_noise_prob=0.5,
        sim_ref=True,
        sim_ref_prob=0.5,
        sim_density=True,
        sim_density_prob=0.5,
        depth_filter=False,
        transform=None,
        transform_sim=None,
        loop=1,
        sim_jitter=True,
        sim_jitter_prob=0.5,
    ):
        super(SimDataset, self).__init__()
        self.source_dataset = build_dataset(source_dataset)
        self.target_dataset = build_dataset(target_dataset)
        self.sim_noise = sim_noise
        self.sim_ref = sim_ref
        self.sim_density = sim_density
        self.sim_noise_prob = sim_noise_prob
        self.sim_ref_prob = sim_ref_prob
        self.sim_density_prob = sim_density_prob
        self.transform = Compose(transform)
        self.transform_sim = Compose(transform_sim)
        self.loop = loop
        self.output_dict = output_dict
        self.few_shot = few_shot
        self.depth_filter = depth_filter
        self.sim_jitter = sim_jitter
        self.sim_jitter_prob = sim_jitter_prob
        if self.few_shot:
            self.select_samples = random.sample(range(len(self.target_dataset)), len(self.target_dataset))
            select_samples = np.array(self.select_samples) % (
                len(self.target_dataset) / self.target_dataset.loop
            )
            unique_values, indices = np.unique(select_samples, return_index=True)
            sorted_indices = np.sort(indices)
            unique_values_in_order = np.array(self.select_samples)[sorted_indices]
            self.select_samples = unique_values_in_order[: self.few_shot].tolist()
        logger = get_root_logger()
        logger.info("simulate target noise {}, ref {}, density {}.".format(sim_noise, sim_ref, sim_density))
        logger.info("Totally {} x {} samples in the source set.".format(len(self.source_dataset), self.loop))
        logger.info("Totally {} x {} samples in the target set.".format(len(self.target_dataset), self.loop))
        if self.few_shot:
            logger.info(
                "Totally {} samples in the target set for few-shot: {}.".format(
                    len(self.select_samples), self.select_samples
                )
            )

    def get_data(self, idx):
        source_idx = np.random.randint(0, len(self.source_dataset))
        source_data = self.source_dataset[source_idx]
        if self.few_shot:
            target_idx = idx % len(self.select_samples)
            target_idx = self.select_samples[target_idx]
        else:
            target_idx = idx
        target_data = self.target_dataset[target_idx]
        min_coord = source_data["coord"].min(axis=0, keepdims=True)

        source_data_ = deepcopy(source_data)
        source_data_["coord"] = np.concatenate((source_data_["coord"], min_coord), axis=0)
        if "strength" in source_data_:
            min_strength = np.zeros(
                (1, source_data_["strength"].shape[1]), dtype=source_data_["strength"].dtype
            )
            source_data_["strength"] = np.concatenate((source_data_["strength"], min_strength), axis=0)
        if "segment" in source_data_:
            min_segment = np.array([self.source_dataset.ignore_index], dtype=source_data_["segment"].dtype)
            source_data_["segment"] = np.concatenate((source_data_["segment"], min_segment), axis=0)
        source_data_ = self.transform(source_data_)
        if self.output_dict == "source":
            data_dict = deepcopy(source_data_)
        source_data_ = {f"source_{key}": value for key, value in source_data_.items()}

        if self.depth_filter:
            # filter target data by depth
            target_mask = np.linalg.norm(target_data["coord"], 2, axis=1) > 1.6
            target_data = index_operator(target_data, target_mask)

        # Deprecated
        if self.sim_jitter and np.random.rand() < self.sim_jitter_prob:
            source_data = self.range_depth_scale(target_data, source_data)

        # Reflectance Intensity Distribution Transfer
        # simulate source_strength to match target_strength
        p = np.random.rand()
        if self.sim_ref and p < self.sim_ref_prob:
            source_strength = source_data["strength"]
            target_strength = target_data["strength"]
            dist_mask = np.linalg.norm(target_data["coord"], 2, axis=1) > 1.6
            simulate_strength = self.histogram_matching_vectorized(
                source_strength, target_strength[dist_mask]
            )
            source_data["strength"] = simulate_strength

        # Range-View-Based Point Drop Pattern Transfer
        p = np.random.rand()
        # simulate drop noise by range view
        if self.sim_noise and p < self.sim_noise_prob:
            # 无效点掩码
            target_mask = ~target_data["proj_mask"].astype(bool)
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            convolved_mask = cv2.filter2D(target_mask.astype(np.float32), -1, kernel)
            # 选中孤立无效点并设置为 0 用于后续丢弃
            convolved_mask = ~((convolved_mask < 1.0) & target_mask)
            unproj_mask = convolved_mask[source_data["proj_y"], source_data["proj_x"]]
            source_data = index_operator(source_data, unproj_mask)

        # Distance-Aware Density Distribution Transfer
        p = np.random.rand()
        # simulate source_data density to match target_data
        if self.sim_density and p < self.sim_density_prob:
            source_coord = source_data["coord"]
            target_coord = target_data["coord"]
            density_mask = self.density_matching_vectorized(source_coord, target_coord)
            source_data = index_operator(source_data, density_mask)

        source_data = self.transform_sim(source_data)

        # keep sim and source in same voxel coord
        source_data["coord"] = np.concatenate((source_data["coord"], min_coord), axis=0)
        if "strength" in source_data:
            min_strength = np.zeros(
                (1, source_data["strength"].shape[1]), dtype=source_data["strength"].dtype
            )
            source_data["strength"] = np.concatenate((source_data["strength"], min_strength), axis=0)
        if "segment" in source_data:
            min_segment = np.array([self.source_dataset.ignore_index], dtype=source_data["segment"].dtype)
            source_data["segment"] = np.concatenate((source_data["segment"], min_segment), axis=0)

        source_data = self.transform(source_data)
        sim_data = {f"sim_{key}": value for key, value in source_data.items()}
        if self.output_dict == "sim":
            data_dict = deepcopy(sim_data)
        target_data = self.transform(target_data)
        if self.output_dict == "target":
            data_dict = deepcopy(target_data)
        target_data = {f"target_{key}": value for key, value in target_data.items()}

        # merge all data
        data_dict.update(sim_data)
        data_dict.update(source_data_)
        data_dict.update(target_data)

        return data_dict

    def get_data_name(self, idx):
        return self.target_dataset.get_data_name(idx)

    def __getitem__(self, idx):
        idx = idx % len(self.target_dataset)
        return self.get_data(idx)

    def __len__(self):
        return len(self.target_dataset) * self.loop

    def density_matching_vectorized(self, source, template):
        """
        将 source coord 的密度匹配到 template coord 的密度 (不是在 range image 上匹配)

        参数:
            source: (N1, 3) 形状的源坐标数组 xyz
            template: (N2, 3) 形状的模板坐标数组 xyz

        返回:
            匹配后的源 mask (N1,) 形状 值为, 0 或 1, 表示是否保留该点
        """
        # L2 范式计算每个点的深度（到原点的距离）
        source_depth = np.linalg.norm(source[:, :3], 2, axis=1)
        template_depth = np.linalg.norm(template[:, :3], 2, axis=1)

        # 固定深度范围为 [0, 100]
        min_depth = 0.0
        max_depth = 100.0

        # 计算直方图 - 使用固定的 bin 边界
        hist_source, bin_edges = np.histogram(
            np.clip(source_depth, min_depth, max_depth), bins=100, range=(min_depth, max_depth)
        )
        hist_template, _ = np.histogram(np.clip(template_depth, min_depth, max_depth), bins=bin_edges)

        # 避免除以零
        hist_source = hist_source.astype(float) + 1e-10
        hist_template = hist_template.astype(float) + 1e-10

        # 计算每个 bin 的保留概率
        ratio = np.where(hist_source > hist_template, (hist_template / hist_source), 1.0)

        # 为每个 source 点找到对应的 bin 和概率
        bin_indices = np.digitize(source_depth, bin_edges) - 1

        # 处理边界情况 (直接截断)
        # 1. 小于 min_depth 的点(分配到第一个 bin)
        # 2. 大于 max_depth 的点(分配到最后一个 bin)
        # 3. 正常范围内的点
        bin_indices = np.clip(bin_indices, 0, len(ratio) - 1)

        # 深度在 [0, 100] 之外的点概率设为 1
        out_of_range_mask = (source_depth < min_depth) | (source_depth > max_depth)
        probs = np.where(out_of_range_mask, 1.0, ratio[bin_indices])

        # 根据概率随机采样决定保留哪些点
        mask = np.random.rand(len(source_depth)) < probs

        return mask

    def histogram_matching_vectorized(self, source, template):
        """
        将 source 的直方图匹配到 template 数组的直方图

        参数:
            source: (N1, 1) 形状的源数组
            template: (N2, 1) 形状的模板数组

        返回:
            匹配后的源数组
        """
        # 获取源数组和模板数组的唯一值及其对应的计数
        # random_perturbation = np.random.rand(*source.shape) / 255. * 3
        # source = source + random_perturbation
        source = np.floor(source[:, 0] * 255)
        template = np.floor(template[:, 0] * 255)
        source_values, source_counts = np.unique(source, return_counts=True)
        template_values, template_counts = np.unique(template, return_counts=True)

        # 计算累积分布函数(CDF)
        source_cdf = np.cumsum(source_counts).astype(np.float64)
        source_cdf /= source_cdf[-1]  # 归一化

        template_cdf = np.cumsum(template_counts).astype(np.float64)
        template_cdf /= template_cdf[-1]  # 归一化

        # 创建插值函数（源CDF -> 源值）
        source_cdf_to_value = interpolate.interp1d(
            source_cdf, source_values, bounds_error=False, fill_value=(source_values[0], source_values[-1])
        )

        # 创建插值函数（源值 -> 源CDF）
        value_to_source_cdf = interpolate.interp1d(
            source_values, source_cdf, bounds_error=False, fill_value=(source_cdf[0], source_cdf[-1])
        )

        # 创建插值函数（模板CDF -> 模板值）
        template_cdf_to_value = interpolate.interp1d(
            template_cdf,
            template_values,
            bounds_error=False,
            fill_value=(template_values[0], template_values[-1]),
        )

        # 对于源数组中的每个值：
        # 1. 找到对应的源CDF
        source_values_cdf = value_to_source_cdf(source)

        # 2. 在模板CDF中找到最接近的值
        # 使用searchsorted找到插入位置
        idx = np.searchsorted(template_cdf, source_values_cdf)
        idx = np.clip(idx, 0, len(template_cdf) - 1)

        # 3. 获取对应的模板值
        matched_values = template_cdf_to_value(template_cdf[idx])

        return matched_values[:, np.newaxis] / 255.0

    def range_depth_scale(self, data_dict, source_dict):
        """
        平滑深度图，使用范围深度平滑方法
        data_dict: 目标域点云
        source_dict: 源域点云
        """
        depth_img = data_dict["proj_range"]
        depth = data_dict["unproj_range"]
        # 用低分辨率复制再填充回原始高分辨率的方法来实现孤立无效点抑制
        depth_img_filled = self.proj_fill(
            2, 1024, data_dict["proj_x"] / 2048, data_dict["proj_y"], depth, depth_img.copy()
        )
        # 填充两次
        depth_img_filled = self.proj_fill(
            4, 512, data_dict["proj_x"] / 2048, data_dict["proj_y"], depth, depth_img_filled
        )
        # 填充后仍为无效点的掩码
        mask = depth_img_filled == -1
        # 平滑处理
        kernel_size = 3
        # 均值滤波 保证卷积核加权平均后权值和为1
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        # 过滤两次
        smoothed_depth_img = cv2.filter2D(depth_img_filled, -1, kernel)
        smoothed_depth_img = cv2.filter2D(smoothed_depth_img, -1, kernel)
        # 计算尺度
        scale = smoothed_depth_img / depth_img_filled
        # 恢复无效点
        scale[mask] = 0
        mask_1 = scale < 0.9
        scale[mask_1] = 1.0  # 保持小于 0.9 的区域不变
        mask_2 = scale > 1.1
        scale[mask_2] = 1.0  # 保持大于 1.1 的区域不变

        # scale 作为最终的目标域缩放掩码，引入源域点云进行对齐
        # 形状说明：
        # - scale: (H, W)
        # - source_dict["proj_y"]: (N_src,) int
        # - source_dict["proj_x"]: (N_src,) int
        # 使用配对的一维索引数组进行高级索引，相当于逐点从 (H, W) 的 scale 图上
        # 取每个源点投影到的像素处的缩放值，结果为一维数组 (N_src,)
        scale = scale[source_dict["proj_y"], source_dict["proj_x"]]  # (N_src,)

        # source_dict["unproj_range"]: (N_src,)
        source_depth_scaled = source_dict["unproj_range"] * scale  # (N_src,)
        # 大于 0.2m 的保持不变（将对应 scale 置回 1.0）
        mask_3 = np.abs(source_depth_scaled - source_dict["unproj_range"]) > 0.2
        scale[mask_3] = 1.0

        # source_dict["coord"]: (N_src, 3)；scale[:, None]: (N_src, 1) -> 广播到 (N_src, 3)
        source_dict["coord"] = source_dict["coord"] * scale[:, None]

        return source_dict

    def proj_fill(self, repeat, proj_w, px, proj_y, depth, proj_range):
        """
        低分辨率复制再回填对 range image 进行无效点填补

        args:
            - repeat: int
                沿宽度方向的重复倍数（例如 2 或 4），满足 proj_w * repeat == W
            - proj_w: int
                低分辨率的宽度（例如 1024 或 512）
            - px: np.ndarray, shape (N,)
                归一化的横向坐标，典型来源为原始投影横坐标除以基准宽度（如 proj_x / 2048），范围约在 [0, 1)
            - proj_y: np.ndarray, shape (N,)
                垂直方向像素索引，取值范围 [0, H-1]，整型
            - depth: np.ndarray, shape (N,)
                与 (proj_y, px) 对应的深度值，> 0
            - proj_range: np.ndarray, shape (H, W)
                原始分辨率的深度图，使用 -1 表示无效像素；要求 W == proj_w * repeat
            - N 是有效投影点数（或用于回填的采样点数），不等于也不必等于 H * W；通常 N <= H * W

        returns:
            - np.ndarray, shape (H, W)：回填后的 range image（仅在“原始无效且低分辨率复制后有效”的位置被填补）
        """
        proj_H = 64
        # px: (N,) -> 缩放到低分辨率网格并离散化
        px = px * proj_w  # (N,)
        px = np.floor(px)  # (N,)
        # 等价于 clamp 操作，确保在 [0, proj_w-1]
        px = np.minimum(proj_w - 1, px)  # (N,)
        px = np.maximum(0, px).astype(np.int32)  # (N,)

        # 低分辨率画布，使用 -1 表示无效
        proj_range512 = np.full((proj_H, proj_w), -1, dtype=np.float32)  # (H, proj_w)

        # 将深度值投影到 range image 上
        # 两个索引数组 proj_y 和 px 都是一维且等长 (N,)
        # 赋值后 proj_range512 仍是 (H, proj_w)
        # 若同一像素位置被多次索引赋值，则会覆盖
        proj_range512[proj_y, px] = depth
        # 沿宽度方向重复 -> (H, proj_w * repeat) == (H, W)
        proj_range512 = np.repeat(proj_range512, repeats=repeat, axis=1)
        # 原始分辨率无效但是低分辨率复制后有效的位置
        mask512_1024 = (proj_range < 0) & (proj_range512 > 0)

        # 将低分辨率复制后有效的值填补到原始分辨率无效点中
        proj_range[mask512_1024] = proj_range512[mask512_1024]  # 抑制无效点
        return proj_range
