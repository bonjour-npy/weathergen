import warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
from accelerate import Accelerator
from rich import print
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from utils.weather_generate import load_points_as_images
import utils.inference

warnings.filterwarnings("ignore", category=UserWarning)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.automatic_dynamic_shapes = False


def sample(args):
    torch.set_grad_enabled(False)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    ddpm, lidar_utils, cfg = utils.inference.setup_model(args.ckpt)

    accelerator = Accelerator(
        mixed_precision=cfg.training["mixed_precision"],
        dynamo_backend=cfg.training["dynamo_backend"],
        split_batches=True,
        # even_batches=False,
        step_scheduler_with_optimizer=True,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print(cfg)

    dataloader = DataLoader(
        TensorDataset(torch.arange(args.num_samples).long()),
        batch_size=args.batch_size,
        num_workers=cfg.training["num_workers"],
        drop_last=False,
    )

    ddpm.to(device)
    # sample_fn = torch.compile(ddpm.sample)
    lidar_utils, dataloader = accelerator.prepare(lidar_utils, dataloader)

    # redefine output directory
    if args.output_dir is None:
        # logs/diffusion/kitti_360/spherical-1024/20250909T154744/models/diffusion_0000060000.pth
        save_dir = Path(
            Path(args.ckpt).parent.parent
            / f"results"
            / f"{args.ckpt.split('/')[-1].split('.')[0]}"
            / args.weather_flag
        )
    else:
        save_dir = Path(args.output_dir)

    with accelerator.main_process_first():
        save_dir.mkdir(parents=True, exist_ok=True)

    def postprocess(sample):
        sample = lidar_utils.denormalize(sample)
        depth, rflct = sample[:, [0]], sample[:, [1]]
        depth = lidar_utils.revert_depth(depth)
        xyz = lidar_utils.to_xyz(depth)
        return torch.cat([depth, xyz, rflct], dim=1)

    for seeds in tqdm(
        dataloader,
        desc="saving...",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
    ):
        if seeds is None:
            break
        else:
            (seeds,) = seeds

        local_batch_size = len(seeds)

        # 缓存 MDP 增强的天气条件，避免重复 IO
        if "weather_cache" not in globals() or weather_cache.get("base") is None:
            base_weather = load_points_as_images(
                point_path="data/kitti_360/dataset/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000018.bin",
                weather_fla=args.weather_flag,
            )  # shape: (1, C, H, W)
            weather_cache = {"base": base_weather}
        else:
            base_weather = weather_cache["base"]

        # 复制到当前本地 batch 尺寸；使用 repeat_interleave 生成显式副本，避免 inplace 造成潜在问题
        weather = base_weather.repeat_interleave(local_batch_size, dim=0).to(device=device)
        samples = ddpm.sample(
            batch_size=local_batch_size,  # 传入的是本地 batch，而不是全局 batch
            num_steps=args.num_steps,
            rng=utils.inference.setup_rng(seeds.cpu().tolist(), device=device),
            progress=False,
            weather=weather,
            train_model="train",  # train 模式来应用 MDP Learnable Mask
        ).clamp(-1, 1)

        samples = postprocess(samples)

        for i in range(len(samples)):
            sample = samples[i]
            torch.save(sample.clone(), save_dir / f"samples_{seeds[i]:010d}.pth")

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=f"logs/diffusion/kitti_360/spherical-1024/20250909T154744/models/diffusion_0000060000.pth",
    )
    parser.add_argument("--weather_flag", type=str, default=f"snow")  # rain, fog, snow
    # 默认保存在 ckpt 同级目录的 results-{ckpt}/{args.weather_flag} 下
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=256)
    args = parser.parse_args()
    sample(args)
