import argparse
from pathlib import Path
from scipy.stats import entropy

import einops
import imageio
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm
from utils.weather_generate import load_points_as_images
import utils.inference
import utils.render
import numpy as np
from utils.weather_lidarshow import load_points_as_images_test
from torchvision.models import inception_v3

def main(args):
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    # =================================================================================
    # Load pre-trained model
    # =================================================================================

    ddpm, lidar_utils, _ = utils.inference.setup_model(args.ckpt, device=args.device)

    # =================================================================================
    # Sampling (reverse diffusion)
    # =================================================================================
    weather = load_points_as_images(point_path="/opt/data/private/project/points2image/000004.bin", weather_fla="fog")
    weather = weather.repeat_interleave(args.batch_size, dim=0).to(device=args.device)
    xs = ddpm.sample(
        batch_size=args.batch_size,
        num_steps=args.sampling_steps,
        return_all=True,
        weather=weather,
    ).clamp(-1, 1)

    # =================================================================================
    # Save as image or video
    # =================================================================================

    xs = lidar_utils.denormalize(xs)
    xs[:, :, [0]] = lidar_utils.revert_depth(xs[:, :, [0]]) / lidar_utils.max_depth

    def resize(x, size):
        return F.interpolate(x, size=size, mode="nearest-exact")
    
    def gaussian_kernel(x, y, sigma=1):
        """高斯核函数"""
        N = x.size(0)
        M = y.size(0)
        delta = x.unsqueeze(1).expand(N, M, *x.size()[1:]) - y.unsqueeze(0).expand(N, M, *y.size()[1:])
        return torch.exp(-torch.sum(delta ** 2, dim=2) / (2 * (sigma ** 2)))

    def mmd(x, y, sigma=1):
        """计算MMD"""
        K_xx = gaussian_kernel(x, x, sigma)
        K_yy = gaussian_kernel(y, y, sigma)
        K_xy = gaussian_kernel(x, y, sigma)
        
        mmd_value = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return mmd_value
    
    inception_model = inception_v3(pretrained=True).type(torch.DoubleTensor)
    inception_model = inception_model.type(torch.FloatTensor)
    inception_model.eval()
    inception_model.requires_grad_(False)
    def extract_features(images):
        with torch.no_grad():
            return inception_model(images)
        
    def render(x):
        img = einops.rearrange(x, "B C H W -> B 1 (C H) W")
        img = utils.render.colorize(img) / 255
        xyz = lidar_utils.to_xyz(x[:, [0]] * lidar_utils.max_depth)

        depth = torch.tensor(np.linalg.norm(xyz.cpu(), ord=2, axis=1, keepdims=True) + 0.0001)
        mask_mask_atten = torch.empty(4, 1, 64, 1024).bernoulli_(0.6) * torch.empty(4, 1, 64, 1).bernoulli_(0.6)
        mask_mask_atten_2 = torch.empty(4, 1, 64, 1024).bernoulli_(0.8)
        zeros = torch.zeros(4, 1, 64, 1024)
        ones = torch.ones(4, 1, 64, 1024)
        fog_atten_out = torch.where(depth > 18, ones, zeros)
        fog_atten_in = torch.where(depth < 18, ones, zeros)
        mask_1 = (fog_atten_out * mask_mask_atten).to(xyz.device)
        mask_2 = (fog_atten_in * mask_mask_atten_2).to(xyz.device)
        mask = mask_1 + mask_2
        xyz = mask * xyz + (1 - mask) * -1

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
    img = img[:, :, 0:64, :]

    # bevtest, imgtest = load_points_as_images_test("/opt/data/private/seeingthroughfog/lidar_hdl64_strongest/2018-02-12_09-25-30_00700.bin") #snow2
    # bevtest, imgtest = load_points_as_images_test("/opt/data/private/seeingthroughfog/lidar_hdl64_strongest/2019-01-09_13-50-10_00900.bin") #snow1
    # bevtest, imgtest = load_points_as_images_test("/opt/data/private/seeingthroughfog/lidar_hdl64_strongest/2019-01-09_09-18-07_00000.bin") #rain1
    bevtest, imgtest = load_points_as_images_test("/opt/data/private/seeingthroughfog/lidar_hdl64_strongest/2018-10-29_15-32-57_01480.bin") #fog2
    # bevtest, imgtest = load_points_as_images_test("/opt/data/private/seeingthroughfog/lidar_hdl64_strongest/2018-10-08_08-10-40_04200.bin") #fog3
    # bevtest, imgtest = load_points_as_images_test("/opt/data/private/seeingthroughfog/lidar_hdl64_strongest/2019-01-09_10-55-25_00330.bin") #fog4 




    bevtest = bevtest.repeat_interleave(bev.shape[0], dim=0)
    imgtest = imgtest.repeat_interleave(img.shape[0], dim=0).unsqueeze(dim=1)
    imgtest = imgtest.repeat_interleave(img.shape[1], dim=1)

    jsd = (0.5 * entropy(bev.cpu(), bevtest.cpu())).mean()
    print('JSD:', jsd)

    mmd_value = mmd(bev.cpu(), bevtest.cpu())
    print('MMD:', mmd_value)

    bev_fid = resize(bev, (229, 229))
    fortest_fid = resize(bevtest, (229, 229))

    real_features = extract_features(fortest_fid.cpu()) # [4,1000]
    fake_features = extract_features(bev_fid.cpu())

    # 计算均值和协方差
    mu_real = torch.mean(real_features, dim=1) # 4
    mu_fake = torch.mean(fake_features, dim=1)
    sigma_real = torch.cov(real_features)
    sigma_fake = torch.cov(fake_features)

    # 计算 FID
    ssdiff = (mu_real - mu_fake) ** 2
    covmean = torch.sqrt((sigma_real + 1e-6) * (sigma_fake + 1e-6))
    # kkk = np.trace(sigma_real + sigma_fake - 2.0 * torch.sqrt(sigma_real @ sigma_fake))
    fid = torch.sum(ssdiff + torch.abs(sigma_real - sigma_fake) / covmean)

    print('FPD:', fid.item())

    img_fid = resize(img, (229, 229))
    imgtest_fid = resize(imgtest, (229, 229))

    real_features = extract_features(imgtest_fid.cpu()) # [4,1000]
    fake_features = extract_features(img_fid.cpu())

    # 计算均值和协方差
    mu_real = torch.mean(real_features, dim=1) # 4
    mu_fake = torch.mean(fake_features, dim=1)
    sigma_real = torch.cov(real_features)
    sigma_fake = torch.cov(fake_features)

    # 计算 FID
    ssdiff = (mu_real - mu_fake) ** 2
    covmean = torch.sqrt((sigma_real + 1e-6) * (sigma_fake + 1e-6))
    # kkk = np.trace(sigma_real + sigma_fake - 2.0 * torch.sqrt(sigma_real @ sigma_fake))
    fid = torch.sum(ssdiff + torch.abs(sigma_real - sigma_fake) / covmean)

    print('FID:', fid.item())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default="/opt/data/private/project/weather-gen_5.10_ssm/logs/diffusion/kitti_360/spherical-1024/mixssm/models/diffusion_0000300000.pth")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sampling_steps", type=int, default=256)
    args = parser.parse_args()
    args.device = torch.device(args.device)
    main(args)
