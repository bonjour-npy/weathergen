import argparse
from pathlib import Path

# from models.CLIP.clip import clip
import einops
import imageio
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from utils.stf_dataset import build_stf_loader
from utils.weather_generate import load_points_as_images, stf_process
import utils.inference
import utils.render
import numpy as np
import datetime


def main(args):
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    # clip_model = clip.load("ViT-B/32",device=args.device)
    # =================================================================================
    # Load pre-trained model
    # =================================================================================

    ddpm, lidar_utils, _ = utils.inference.setup_model(args.ckpt, device=args.device)

    # =================================================================================
    # Sampling (reverse diffusion)
    # =================================================================================
    # 这里是错的，不因该用 stf 数据集提供天气条件
    # 以 STFDataset 提供 weather 条件（示例选用 rain）
    loader = build_stf_loader(
        weather="rain", batch_size=args.batch_size, num_workers=4, resolution=(64, 1024)
    )
    batch_2ch = next(iter(loader)).to(args.device)
    weather = batch_2ch  # ddpm.sample 期望 [-1, 1] 归一化的 2 通道输入

    # 加载 kitti_360 数据集之后通过 weather_flag 进行 MDP 增强
    # weather = load_points_as_images(
    #     point_path="./KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000018.bin",
    #     weather_fla="rain",
    # )
    # weather = weather.repeat_interleave(args.batch_size, dim=0).to(device=args.device)

    xs = ddpm.sample(
        batch_size=args.batch_size,
        num_steps=args.sampling_steps,
        return_all=True,
        weather=weather,
        # weather=text_features,
        train_model="finetune",
        # train_model='train',
    ).clamp(-1, 1)

    # =================================================================================
    # Save as image or video
    # =================================================================================

    xs = lidar_utils.denormalize(xs)
    xs[:, :, [0]] = lidar_utils.revert_depth(xs[:, :, [0]]) / lidar_utils.max_depth
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")

    def render(x):
        img = einops.rearrange(x, "B C H W -> B 1 (C H) W")
        img = utils.render.colorize(img) / 255
        xyz = lidar_utils.to_xyz(x[:, [0]] * lidar_utils.max_depth)
        xyz /= lidar_utils.max_depth
        z_min, z_max = -2 / lidar_utils.max_depth, 0.5 / lidar_utils.max_depth
        z = (xyz[:, [2]] - z_min) / (z_max - z_min)
        colors = utils.render.colorize(z.clamp(0, 1), cm.viridis) / 255
        R, t = utils.render.make_Rt(pitch=torch.pi / 3, yaw=torch.pi / 4, z=0.8)
        bev = 1 - utils.render.render_point_clouds(
            points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
            colors=1 - einops.rearrange(colors, "B C H W -> B (H W) C"),
            R=R.to(xyz),
            t=t.to(xyz),
        )
        return img, bev

    img, bev = render(xs[-1])
    save_image(img, "./samples_img.png", nrow=1)
    save_image(bev, "./samples_bev.png", nrow=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=Path,
        default="logs/diffusion/kitti_360/spherical-1024/20250910T125905/models/diffusion_0000300000.pth",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sampling_steps", type=int, default=256)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)
