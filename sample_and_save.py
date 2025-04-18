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
        mixed_precision=cfg.training['mixed_precision'],
        dynamo_backend=cfg.training['dynamo_backend'],
        split_batches=True,
        even_batches=False,
        step_scheduler_with_optimizer=True,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print(cfg)

    dataloader = DataLoader(
        TensorDataset(torch.arange(args.num_samples).long()),
        batch_size=args.batch_size,
        num_workers=cfg.training['num_workers'],
        drop_last=False,
    )

    ddpm.to(device)
    # sample_fn = torch.compile(ddpm.sample)
    lidar_utils, dataloader = accelerator.prepare(lidar_utils, dataloader)

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

        weather = load_points_as_images(point_path="./KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000018.bin", weather_fla="snow") # or random select
        weather = weather.repeat_interleave(args.batch_size, dim=0).to(device=device)
        samples = ddpm.sample(
                batch_size=len(seeds),
                num_steps=args.num_steps,
                rng=utils.inference.setup_rng(seeds.cpu().tolist(), device=device),
                progress=False,
                weather=weather,
                train_model='train',
        ).clamp(-1, 1)


        samples = postprocess(samples)

        for i in range(len(samples)):
            sample = samples[i]
            torch.save(sample.clone(), save_dir / f"samples_{seeds[i]:010d}.pth")

        accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, default='./project/diffusion_steps.pth')
    parser.add_argument("--output_dir", type=str, default='./project/results')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--num_steps", type=int, default=256)
    args = parser.parse_args()
    sample(args)
