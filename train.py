import dataclasses
import datetime
import json
import os
import warnings
from pathlib import Path

from models.CLIP.clip import clip
import datasets as ds
import einops
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from ema_pytorch import EMA
from rich import print
from simple_parsing import ArgumentParser
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.weather import weather_process
from utils.stf_dataset import build_stf_loader
import utils.inference
import utils.option
import utils.render
import utils.training
from models.diffusion import (
    ContinuousTimeGaussianDiffusion,
    DiscreteTimeGaussianDiffusion,
)
from models.efficient_unet import EfficientUNet  # With Mamba
from models.refinenet import LiDARGenRefineNet
from utils.lidar import LiDARUtility, get_hdl64e_linear_ray_angles

from rich import print

# debug
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.automatic_dynamic_shapes = False
rng = np.random.default_rng(seed=42)


def train(cfg: utils.option.Config):
    torch.backends.cudnn.benchmark = True
    project_dir = Path(cfg.training.output_dir) / cfg.data.dataset / cfg.data.projection

    # =================================================================================
    # Initialize accelerator
    # =================================================================================

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_with=["tensorboard"],
        project_dir=project_dir,
        dynamo_backend=cfg.training.dynamo_backend,
        split_batches=True,
        step_scheduler_with_optimizer=True,
        # sync_bn=True,
    )
    if accelerator.is_main_process:
        print(cfg)
        os.makedirs(project_dir, exist_ok=True)
        project_name = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        accelerator.init_trackers(project_name=project_name)
        tracker = accelerator.get_tracker("tensorboard")
        json.dump(
            dataclasses.asdict(cfg),
            open(Path(tracker.logging_dir) / "training_config.json", "w"),
            indent=4,
        )
    device = accelerator.device

    # =================================================================================
    # Setup models
    # =================================================================================

    channels = [
        1 if cfg.data.train_depth else 0,
        1 if cfg.data.train_reflectance else 0,
    ]

    if cfg.model.architecture == "efficient_unet":
        model = EfficientUNet(
            in_channels=sum(channels),
            resolution=cfg.data.resolution,  # 64, 1024
            base_channels=cfg.model.base_channels,  # 64
            temb_channels=cfg.model.temb_channels,
            channel_multiplier=cfg.model.channel_multiplier,
            num_residual_blocks=cfg.model.num_residual_blocks,
            gn_num_groups=cfg.model.gn_num_groups,
            gn_eps=cfg.model.gn_eps,
            attn_num_heads=cfg.model.attn_num_heads,
            coords_encoding=cfg.model.coords_encoding,
            ring=True,
        )
    elif cfg.model.architecture == "refinenet":
        model = LiDARGenRefineNet(
            in_channels=sum(channels),
            resolution=cfg.data.resolution,
            base_channels=cfg.model.base_channels,
            channel_multiplier=cfg.model.channel_multiplier,
        )
    else:
        raise ValueError(f"Unknown: {cfg.model.architecture}")

    if "spherical" in cfg.data.projection:
        model.coords = get_hdl64e_linear_ray_angles(*cfg.data.resolution)
    elif "unfolding" in cfg.data.projection:
        model.coords = F.interpolate(
            torch.load(f"data/{cfg.data.dataset}/unfolding_angles.pth"),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
    else:
        raise ValueError(f"Unknown: {cfg.data.projection}")

    if accelerator.is_main_process:
        print(f"number of parameters: {utils.inference.count_parameters(model):,}")

    if cfg.diffusion.timestep_type == "discrete":
        ddpm = DiscreteTimeGaussianDiffusion(
            model=model,
            prediction_type=cfg.diffusion.prediction_type,
            loss_type=cfg.diffusion.loss_type,
            noise_schedule=cfg.diffusion.noise_schedule,
            num_training_steps=cfg.diffusion.num_training_steps,
        )
    # r2dm adopts continuous
    # num_training_steps equals to 1000 in base.py
    elif cfg.diffusion.timestep_type == "continuous":
        ddpm = ContinuousTimeGaussianDiffusion(
            model=model,
            prediction_type=cfg.diffusion.prediction_type,
            loss_type=cfg.diffusion.loss_type,  # l2
            noise_schedule=cfg.diffusion.noise_schedule,
        )
    else:
        raise ValueError(f"Unknown: {cfg.diffusion.timestep_type}")

    # Load pretrained weights for fine-tuning
    if cfg.training.train_model == "finetune":
        checkpoint_path = (
            "logs/diffusion/kitti_360/spherical-1024/20250910T125905/models/diffusion_0000300000.pth"
        )

        with accelerator.main_process_first():
            if accelerator.is_main_process:
                print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        ddpm.load_state_dict(checkpoint["weights"])

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print("Checkpoint loaded successfully across all processes")

    # ------------------------------------------------------------------------------------------------ #
    # Load pretrained weights for fine-tuning
    # if cfg.training.train_model == "finetune":
    #     ddpm = utils.inference.load_model(
    #         "logs/diffusion/kitti_360/spherical-1024/20250910T125905/models/diffusion_0000300000.pth",
    #         device=device,
    #     )

    #     # 重新设置 coords 与 LFA 冻结状态
    #     # if "spherical" in cfg.data.projection:
    #     #     ddpm.model.coords = get_hdl64e_linear_ray_angles(*cfg.data.resolution)
    #     # elif "unfolding" in cfg.data.projection:
    #     #     ddpm.model.coords = F.interpolate(
    #     #         torch.load(f"data/{cfg.data.dataset}/unfolding_angles.pth"),
    #     #         size=cfg.data.resolution,
    #     #         mode="nearest-exact",
    #     #     )
    # ------------------------------------------------------------------------------------------------ #

    # 冻结 / 解冻 LFA (VAE) 支路，避免 DDP 未用参数错误
    if hasattr(ddpm.model, "set_lfa_mode"):
        ddpm.model.set_lfa_mode(bool(cfg.model.lfa))

    ddpm.train()
    ddpm.to(device)

    clip_model = clip.load("ViT-B/32", device=device)

    if accelerator.is_main_process:
        ddpm_ema = EMA(
            ddpm,
            beta=cfg.training.ema_decay,
            update_every=cfg.training.ema_update_every,
            update_after_step=cfg.training.lr_warmup_steps * cfg.training.gradient_accumulation_steps,
        )
        ddpm_ema.to(device)

    # CPU 版用于 DataLoader 的 collate_fn，避免在 worker 进程中触碰 CUDA 张量
    lidar_utils_cpu = LiDARUtility(
        resolution=cfg.data.resolution,
        depth_format=cfg.data.depth_format,
        min_depth=cfg.data.min_depth,
        max_depth=cfg.data.max_depth,
        ray_angles=ddpm.model.coords,
    )
    # GPU 版用于训练阶段的渲染和日志
    lidar_utils = LiDARUtility(
        resolution=cfg.data.resolution,
        depth_format=cfg.data.depth_format,
        min_depth=cfg.data.min_depth,
        max_depth=cfg.data.max_depth,
        ray_angles=ddpm.model.coords,
    ).to(device)

    # =================================================================================
    # Setup optimizer & dataloader
    # =================================================================================

    optimizer = torch.optim.AdamW(
        (p for p in ddpm.parameters() if p.requires_grad),
        lr=cfg.training.lr,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        weight_decay=cfg.training.adam_weight_decay,
        eps=cfg.training.adam_epsilon,
    )

    dataset = ds.load_dataset(
        path=f"data/{cfg.data.dataset}",
        name=cfg.data.projection,
        split=ds.Split.TRAIN,
        num_proc=cfg.training.num_workers,
        trust_remote_code=True,
    ).with_format("torch")

    if accelerator.is_main_process:
        print(dataset)

    # 将预处理与 MDP 封装成 DataLoader 的 collate_fn，避免在训练循环中逐 batch 处理
    def _kitti360_collate(samples):
        # stack 原始字段（仅保留必要字段）
        depth = torch.stack([s["depth"] for s in samples], dim=0)  # (B, 1, H, W)
        reflectance = None
        if cfg.data.train_reflectance:
            reflectance = torch.stack([s["reflectance"] for s in samples], dim=0)  # (B, 1, H, W)
        xyz = torch.stack([s["xyz"] for s in samples], dim=0)  # (B, 3, H, W)

        # 在 CPU 上完成与 preprocess 等价的操作
        x_parts = []
        if cfg.data.train_depth:
            x_parts.append(lidar_utils_cpu.convert_depth(depth))
        if cfg.data.train_reflectance and reflectance is not None:
            x_parts.append(reflectance)
        x = torch.cat(x_parts, dim=1)
        x = lidar_utils_cpu.normalize(x)
        x = F.interpolate(x, size=cfg.data.resolution, mode="nearest-exact")

        # 为整个 batch 随机选择同一种天气（与原训练循环一致）
        q = np.random.randint(0, 4)
        if q == 0:
            weather_flag = "normal"
        elif q == 1:
            weather_flag = "fog"
        elif q == 2:
            weather_flag = "snow"
        else:
            weather_flag = "rain"

        # 在 CPU 上应用天气增强（保持与原 weather_process 语义一致）
        x_0 = weather_process(x, weather_flag, xyz, depth)

        return {
            "x_0": x_0,  # (B,C,H,W)
            "weather_flag": weather_flag,
        }

    def _worker_init_fn(worker_id: int):
        try:
            base_seed = torch.initial_seed() % 2**32
        except Exception:
            base_seed = np.random.SeedSequence().entropy
        np.random.seed((base_seed + worker_id) % (2**32 - 1))

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size_train,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        # num_workers=0,
        drop_last=True,
        pin_memory=True,
        collate_fn=_kitti360_collate,
        worker_init_fn=_worker_init_fn,
    )

    lr_scheduler = utils.training.get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps * cfg.training.gradient_accumulation_steps,
        num_training_steps=cfg.training.num_steps * cfg.training.gradient_accumulation_steps,
    )

    # STF weather loaders
    need_stf = (cfg.training.train_model == "finetune") or bool(cfg.model.lfa)
    if need_stf:
        stf_fog_loader = build_stf_loader(
            weather="fog",
            batch_size=cfg.training.batch_size_train,
            num_workers=cfg.training.num_workers,
            resolution=cfg.data.resolution,
        )
        stf_snow_loader = build_stf_loader(
            weather="snow",
            batch_size=cfg.training.batch_size_train,
            num_workers=cfg.training.num_workers,
            resolution=cfg.data.resolution,
        )
        stf_rain_loader = build_stf_loader(
            weather="rain",
            batch_size=cfg.training.batch_size_train,
            num_workers=cfg.training.num_workers,
            resolution=cfg.data.resolution,
        )
        ddpm, optimizer, dataloader, lr_scheduler, stf_fog_loader, stf_snow_loader, stf_rain_loader = (
            accelerator.prepare(
                ddpm, optimizer, dataloader, lr_scheduler, stf_fog_loader, stf_snow_loader, stf_rain_loader
            )
        )

        def _infinite(loader):
            while True:
                for b in loader:
                    yield b

        if accelerator.is_main_process:
            print(f"STF loaders ready. Training Mode = {cfg.training.train_model}. LFA = {cfg.model.lfa}.")
        stf_iters = {
            "fog": iter(_infinite(stf_fog_loader)),
            "snow": iter(_infinite(stf_snow_loader)),
            "rain": iter(_infinite(stf_rain_loader)),
        }
    else:
        ddpm, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            ddpm, optimizer, dataloader, lr_scheduler
        )
        if accelerator.is_main_process:
            print(f"STF loaders off. Training Mode = {cfg.training.train_model}. LFA = {cfg.model.lfa}.")

    # =================================================================================
    # Utility
    # =================================================================================

    def preprocess(batch):
        x = []
        if cfg.data.train_depth:
            x += [lidar_utils.convert_depth(batch["depth"])]
        if cfg.data.train_reflectance:
            x += [batch["reflectance"]]
        x = torch.cat(x, dim=1)
        x = lidar_utils.normalize(x)
        x = F.interpolate(
            x.to(device),
            size=cfg.data.resolution,
            mode="nearest-exact",
        )
        return x

    def split_channels(image: torch.Tensor):
        depth, rflct = torch.split(image, channels, dim=1)
        return depth, rflct

    @torch.inference_mode()
    def log_images(image, tag: str = "name", global_step: int = 0):
        image = lidar_utils.denormalize(image)
        out = dict()
        depth, rflct = split_channels(image)
        if depth.numel() > 0:
            out[f"{tag}/depth"] = utils.render.colorize(depth)
            metric = lidar_utils.revert_depth(depth)
            mask = (metric > lidar_utils.min_depth) & (metric < lidar_utils.max_depth)
            out[f"{tag}/depth/orig"] = utils.render.colorize(metric / lidar_utils.max_depth)
            xyz = lidar_utils.to_xyz(metric) / lidar_utils.max_depth * mask
            normal = -utils.render.estimate_surface_normal(xyz)
            normal = lidar_utils.denormalize(normal)
            R, t = utils.render.make_Rt(pitch=torch.pi / 4, yaw=torch.pi / 4, z=0.6)
            bev = utils.render.render_point_clouds(
                points=einops.rearrange(xyz, "B C H W -> B (H W) C"),
                colors=einops.rearrange(normal, "B C H W -> B (H W) C"),
                R=R.to(xyz),
                t=t.to(xyz),
            )
            out[f"{tag}/bev"] = bev.mul(255).clamp(0, 255).byte()
        if rflct.numel() > 0:
            out[f"{tag}/reflectance"] = utils.render.colorize(rflct, cm.plasma)
        if mask.numel() > 0:
            out[f"{tag}/mask"] = utils.render.colorize(mask, cm.binary_r)
        tracker.log_images(out, step=global_step)

    # =================================================================================
    # Training loop
    # =================================================================================

    progress_bar = tqdm(
        range(cfg.training.num_steps),
        desc="training",
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
    )

    text = {
        "This is the LiDAR range map for a sunny day.",
        "This is the LiDAR range map for a foggy day.",
        "This is the LiDAR range map for a snowy day.",
        "This is the LiDAR range map for a rainy day.",
    }
    text_emb = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_emb)  # B, 512

    global_step = 0
    while global_step < cfg.training.num_steps:
        ddpm.train()

        # for batch in dataloader:  # 每个 batch 输入时单独进行天气处理
        #     x_0 = preprocess(batch)  # B, 2, 64, 1024
        #     weather_flag = random_select()
        #     x_0 = weather_process(x_0, weather_flag, batch["xyz"], batch["depth"])
        for batch in dataloader:
            # 重写 kitti_360 的 dataloader 的 collate_fn 方法，完成 preprocess + MDP 模拟
            x_0 = batch["x_0"].to(device, non_blocking=True)
            weather_flag = batch["weather_flag"]

            if weather_flag == "normal":
                text_features = torch.cat([text_features, text_features[0].unsqueeze(0)], dim=0)
                if need_stf:
                    x_weather = x_0

            if weather_flag == "fog":
                text_features = torch.cat([text_features, text_features[1].unsqueeze(0)], dim=0)
                if need_stf:
                    # 由 DataLoader 预取，避免在线 I/O
                    x_weather = next(stf_iters["fog"]).to(device, non_blocking=True)

            if weather_flag == "snow":
                text_features = torch.cat([text_features, text_features[2].unsqueeze(0)], dim=0)
                if need_stf:
                    x_weather = next(stf_iters["snow"]).to(device, non_blocking=True)

            if weather_flag == "rain":
                text_features = torch.cat([text_features, text_features[3].unsqueeze(0)], dim=0)
                if need_stf:
                    x_weather = next(stf_iters["rain"]).to(device, non_blocking=True)

            with accelerator.accumulate(ddpm):
                """
                # from original code
                loss = ddpm(x_0=x_0, x_condition=x_0, text=text_features, weather=x_weather, train_model='train')
                # fine-tune
                # loss = ddpm(x_0=x_weather, x_condition=x_0, text=text_features, weather=x_weather, train_model='finetune')
                accelerator.backward(loss)
                """
                # loss path
                if cfg.training.train_model == "finetune":
                    loss = ddpm(
                        x_0=x_weather,
                        x_condition=x_0,
                        text=text_features,
                        weather=x_weather,  # fine-tune 要加载 stf，但是用不用由 cfg.model.lfa 控制
                        train_model="finetune",
                        train_lfa=bool(cfg.model.lfa),
                    )
                elif cfg.training.train_model == "train":
                    if cfg.model.lfa:
                        loss = ddpm(
                            x_0=x_0,
                            x_condition=x_0,
                            text=text_features,
                            weather=x_weather,
                            train_model="train",
                            train_lfa=True,
                        )
                    else:
                        loss = ddpm(
                            x_0=x_0,
                            x_condition=x_0,
                            text=text_features,
                            weather=None,  # 传入 x_0 占位
                            train_model="train",
                            train_lfa=False,
                        )
                # fine-tune
                # loss = ddpm(x_0=x_weather, x_condition=x_0, text=text_features, weather=x_weather, train_model='finetune')
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            log = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            if accelerator.is_main_process:
                ddpm_ema.update()
                log["ema/decay"] = ddpm_ema.get_current_decay()

                # Update progress bar with loss
                progress_bar.set_postfix(loss=f"{loss.item():.6f}")

                if global_step == 1:
                    log_images(x_0, "image", global_step)

                if global_step % cfg.training.steps_save_image == 0:
                    print(weather_flag)
                    ddpm_ema.ema_model.eval()
                    # 对齐采样条件的 batch 维度
                    weather_for_sampling = (
                        x_0[: cfg.training.batch_size_eval]
                        if x_0.shape[0] >= cfg.training.batch_size_eval
                        else x_0.repeat_interleave(cfg.training.batch_size_eval, dim=0)
                    )
                    sample = ddpm_ema.ema_model.sample(
                        batch_size=cfg.training.batch_size_eval,
                        num_steps=cfg.diffusion.num_sampling_steps,
                        rng=torch.Generator(device=device).manual_seed(42),
                        weather=weather_for_sampling,  # weather condition
                        train_model="train",  # 采样时使用 train 模式以启用 MDP
                    )
                    log_images(sample, "sample", global_step)

                if global_step % cfg.training.steps_save_model == 0:
                    save_dir = Path(tracker.logging_dir) / "models"
                    save_dir.mkdir(exist_ok=True, parents=True)
                    torch.save(
                        {
                            "cfg": dataclasses.asdict(cfg),
                            "weights": ddpm_ema.online_model.state_dict(),
                            "ema_weights": ddpm_ema.ema_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "global_step": global_step,
                        },
                        save_dir / f"diffusion_{global_step:010d}.pth",
                    )

            accelerator.log(log, step=global_step)
            progress_bar.update(1)

            if global_step >= cfg.training.num_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(utils.option.Config, dest="cfg")
    train(parser.parse_args().cfg)
