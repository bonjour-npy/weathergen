from typing import Iterable, Literal
from mamba_ssm import Mamba
import einops
import numpy as np
import torch
from torch import nn

from . import encoding, ops


def _join(*tensors) -> torch.Tensor:
    return torch.cat(tensors, dim=1)


def _n_tuple(x: Iterable | int, N: int) -> tuple[int]:
    if isinstance(x, Iterable):
        assert len(x) == N
        return x
    else:
        return (x,) * N


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        gn_eps: float = 1e-6,
        gn_num_groups: int = 8,
        scale: float = 1 / np.sqrt(2),
    ):
        super().__init__()
        self.norm = nn.GroupNorm(gn_num_groups, in_channels, gn_eps)
        self.attn = nn.MultiheadAttention(
            embed_dim=in_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.attn.out_proj.apply(ops.zero_out)
        self.register_buffer("scale", torch.tensor(scale).float())

    def residual(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        B, C, H, W = h.shape
        h = einops.rearrange(h, "B C H W -> B (H W) C")
        h, _ = self.attn(query=h, key=h, value=h, need_weights=False)
        h = einops.rearrange(h, "B (H W) C -> B C H W", H=H, W=W)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.residual(x)
        h = h * self.scale
        return h


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int | None,
        gn_num_groups: int = 8,
        gn_eps: float = 1e-6,
        scale: float = 1 / np.sqrt(2),
        dropout: float = 0.0,
        ring: bool = False,
    ):
        super().__init__()
        self.has_emb = emb_channels is not None

        # layer 1
        self.norm1 = nn.GroupNorm(gn_num_groups, in_channels, gn_eps)
        self.silu1 = nn.SiLU()
        self.conv1 = ops.Conv2d(in_channels, out_channels, 3, 1, 1, ring=ring)

        # layer 2
        if self.has_emb:
            self.norm2 = ops.AdaGN(emb_channels, out_channels, gn_num_groups, gn_eps)
        else:
            self.norm2 = nn.GroupNorm(gn_num_groups, out_channels, gn_eps)
        self.silu2 = nn.SiLU()
        self.drop2 = nn.Dropout(dropout)
        self.conv2 = ops.Conv2d(out_channels, out_channels, 3, 1, 1, ring=ring)
        self.conv2.apply(ops.zero_out)

        # skip connection
        self.skip = (
            ops.Conv2d(in_channels, out_channels, 1, 1, 0)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.register_buffer("scale", torch.tensor(scale).float())

    def residual(
        self, x: torch.Tensor, emb: torch.Tensor, h_emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.silu1(h)
        h = self.conv1(h)
        h = self.norm2(h, emb, h_emb) if self.has_emb else self.norm2(h)
        h = self.silu2(h)
        h = self.drop2(h)
        h = self.conv2(h)
        return h

    def forward(self, x: torch.Tensor, emb: torch.Tensor, h_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.skip(x) + self.residual(x, emb, h_emb)
        h = h * self.scale
        return h


class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_residual_blocks: int,
        emb_channels: int,
        gn_num_groups: int = 8,
        gn_eps: float = 1e-6,
        attn: bool = False,
        attn_num_heads: int = 8,
        up: int = 1,
        down: int = 1,
        dropout: float = 0.0,
        ring: bool = False,
    ):
        super().__init__()

        # downsampling
        self.downsample = (
            nn.Sequential(
                ops.Conv2d(in_channels, out_channels, 3, 1, 1, ring=ring),
                ops.Resample(down=down, ring=ring),
            )
            if down > 1
            else nn.Identity()
        )

        # resnet blocks x N
        self.residual_blocks = ops.ConditionalSequential()
        for i in range(num_residual_blocks):
            self.residual_blocks.append(
                ResidualBlock(
                    in_channels=out_channels if i != 0 or down > 1 else in_channels,
                    out_channels=out_channels,
                    emb_channels=emb_channels,
                    gn_num_groups=gn_num_groups,
                    gn_eps=gn_eps,
                    dropout=dropout,
                    ring=ring,
                )
            )

        # self-attention
        self.self_attn_block = (
            SelfAttentionBlock(
                in_channels=out_channels,
                num_heads=attn_num_heads,
                gn_eps=gn_eps,
                gn_num_groups=gn_num_groups,
            )
            if attn
            else nn.Identity()
        )

        # upsampling
        self.upsample = (
            nn.Sequential(
                ops.Resample(up=up, ring=ring),
                ops.Conv2d(out_channels, out_channels, 3, 1, 1, ring=ring),
            )
            if up > 1
            else nn.Identity()
        )

    def forward(
        self, h: torch.Tensor, temb: torch.Tensor, h_emb: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.downsample(h)
        h = self.residual_blocks(h, temb, h_emb)
        h = self.self_attn_block(h)
        h = self.upsample(h)
        return h

class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim//4, # Model dimension d_model 
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C, H, W = x.shape
        assert C == self.input_dim
        x = einops.rearrange(x, "B C H W -> B C (H W)")
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2) # Transform back into B (H W) C
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2,x_mamba3,x_mamba4], dim=2)
        # x_mamba = self.mamba(x_norm) + self.skip_scale * x_norm

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        out = einops.rearrange(out, "B C (H W) -> B C H W", H=H, W=W)
        return out

class EfficientUNet(nn.Module):
    """
    Re-implementation of Efficient U-Net (https://arxiv.org/abs/2205.11487)
    + Our modification for LiDAR domain
    """

    def __init__(
        self,
        in_channels: int,
        resolution: tuple[int, int] | int,
        out_channels: int | None = None,  # == in_channels if None
        base_channels: int = 128,
        temb_channels: int = None,
        channel_multiplier: tuple[int] | int = (1, 2, 4, 8),
        num_residual_blocks: tuple[int] | int = (3, 3, 3, 3),
        gn_num_groups: int = 32 // 4,
        gn_eps: float = 1e-6,
        attn_num_heads: int = 8,
        coords_encoding: Literal[
            "spherical_harmonics", "polar_coordinates", "fourier_features", None
        ] = "spherical_harmonics",
        ring: bool = True,
    ):
        super().__init__()
        self.resolution = _n_tuple(resolution, 2)
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        temb_channels = base_channels * 4 if temb_channels is None else temb_channels

        # spatial coords embedding
        coords = encoding.generate_polar_coords(*self.resolution)
        self.register_buffer("coords", coords)
        self.coords_encoding = None
        if coords_encoding == "spherical_harmonics":
            self.coords_encoding = encoding.SphericalHarmonics(levels=5)
            in_channels += self.coords_encoding.extra_ch
        elif coords_encoding == "polar_coordinates":
            self.coords_encoding = nn.Identity()
            in_channels += coords.shape[1]
        elif coords_encoding == "fourier_features":
            self.coords_encoding = encoding.FourierFeatures(self.resolution)
            in_channels += self.coords_encoding.extra_ch

        # timestep embedding
        self.time_embedding = nn.Sequential(
            ops.SinusoidalPositionalEmbedding(base_channels),
            nn.Linear(base_channels, temb_channels),
            nn.SiLU(),
            nn.Linear(temb_channels, temb_channels),
        )

        # parameters for up/down-sampling blocks
        updown_levels = 4
        channel_multiplier = _n_tuple(channel_multiplier, updown_levels)
        C = [base_channels] + [base_channels * m for m in channel_multiplier]
        N = _n_tuple(num_residual_blocks, updown_levels)

        cfgs = dict(
            emb_channels=temb_channels,
            gn_num_groups=gn_num_groups,
            gn_eps=gn_eps,
            attn_num_heads=attn_num_heads,
            dropout=0.0,
            ring=ring,
        )

        # weather condition
        self.in_conv_weather1 = ops.Conv2d(in_channels, in_channels-2, 7, 3, 3, ring=ring)
        self.in_conv_weather2 = ops.Conv2d(8, 8, 3, 2, 2, ring=ring)
        self.in_conv_weather3 = ops.Conv2d(1, 1, 3, 2, 2, ring=ring)
        self.encoder_weather1 = nn.Sequential(nn.Linear(7*87,512), 
                                              nn.ReLU(),)
        self.encoder_weather2 = nn.Sequential(nn.Linear(512,256), 
                                              nn.ReLU(),)
        self.weather_output = nn.Linear(256,512)


        # mamba settings
        self.mamba1 = nn.Sequential(PVMLayer(input_dim=C[3], output_dim=C[3]))
        self.mamba2 = nn.Sequential(PVMLayer(input_dim=C[4], output_dim=C[4]))
        self.mamba3 = nn.Sequential(PVMLayer(input_dim=C[3], output_dim=C[3]))
        self.mamba4 = nn.Sequential(PVMLayer(input_dim=C[2], output_dim=C[2]))
        self.mamba_wea_1 = nn.Sequential(PVMLayer(input_dim=32, output_dim=8))
        self.mamba_wea_2 = nn.Sequential(PVMLayer(input_dim=8, output_dim=1))
        self.ebn1 = nn.GroupNorm(4, C[3])
        self.ebn2 = nn.GroupNorm(4, C[4])
        self.ebn3 = nn.GroupNorm(4, C[3])
        self.ebn4 = nn.GroupNorm(4, C[2])
        self.silu = nn.SiLU()

        # downsampling blocks
        self.in_conv = ops.Conv2d(in_channels, C[0], 3, 1, 1, ring=ring)
        self.d_block1 = Block(C[0], C[1], N[0], **cfgs)
        self.d_block2 = Block(C[1], C[2], N[1], down=2, **cfgs)
        self.d_block3 = Block(C[2], C[3], N[2], down=2, **cfgs)
        self.d_block4 = Block(C[3], C[4], N[3], down=2, attn=True, **cfgs)

        # upsampling blocks
        self.u_block4 = Block(C[4], C[3], N[3], up=2, attn=True, **cfgs)
        self.u_block3 = Block(C[3] + C[3], C[2], N[2], up=2, **cfgs)
        self.u_block2 = Block(C[2] + C[2], C[1], N[1], up=2, **cfgs)
        self.u_block1 = Block(C[1] + C[1], C[0], N[0], **cfgs)
        self.out_conv = ops.Conv2d(C[0], self.out_channels, 3, 1, 1, ring=ring)
        self.out_conv.apply(ops.zero_out)

        # VAE encoder
        self.in_conv_vae = ops.Conv2d(2, 2, 7, 3, 3, ring=ring)
        self.in_conv_vae2 = ops.Conv2d(2, 1, 3, 2, 2, ring=ring)
        self.in_conv_vae3 = ops.Conv2d(1, 1, 3, 2, 2, ring=ring)
        self.encoder_vae = nn.Sequential(nn.Linear(7*87,256),
                    nn.ReLU(),
                    nn.Linear(256,128),
                    nn.ReLU(),)
        self.vae_mu = torch.nn.Linear(128, 10)
        self.vae_sigma = torch.nn.Linear(128, 10)

        # VAE decoder
        self.vae_linear1 = torch.nn.Linear(10, 128)
        self.vae_linear2 = torch.nn.Linear(128, 256)
        self.vae_linear3 = torch.nn.Linear(256, 609)

        # learnable_mask
        self.mask = nn.Parameter(torch.zeros(1, 2, 64, 1024), requires_grad=True) #fintune


    def forward(self, images: torch.Tensor, timesteps: torch.Tensor, images_condition: torch.Tensor, weather: torch.Tensor,
                alpha: torch.Tensor, sigma: torch.Tensor, train_model: str) -> torch.Tensor:
        a = alpha.clone()
        b = sigma.clone()

        # newmask
        if train_model == 'train':
            h = images + torch.ones_like(images) * self.mask
        elif train_model == 'finetune':
            h = images + 0 * self.mask
        elif train_model == 'clip':
            h = images + 0 * self.mask
            
        # h = images

        h_emb = images_condition

        # timestep embedding
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].repeat_interleave(h.shape[0], dim=0)
        temb = self.time_embedding(timesteps.to(h))

        # spatial embedding
        if self.coords_encoding is not None:
            cenc = self.coords_encoding(self.coords)
            cenc = cenc.repeat_interleave(h.shape[0], dim=0)
            h = torch.cat([h, cenc], dim=1)
            if train_model == 'train':
                h_emb = torch.cat([h_emb, cenc], dim=1) # B,34,64,1024
            elif train_model == 'finetune':
                h_emb = torch.cat([h_emb, cenc], dim=1) # B,34,64,1024
            # elif train_model == 'clipsample':
            #     h_emb = torch.cat([h_emb, cenc], dim=1) # B,34,64,1024

        if train_model == 'train':
            h_emb = self.in_conv_weather1(h_emb) # B,32,22,342
            h_emb = self.silu(self.mamba_wea_1(h_emb)) # B,8,22,342
            h_emb = self.in_conv_weather2(h_emb) # B,8,12,172
            h_emb = einops.rearrange(h_emb, "B C H W -> B C W H")
            h_emb = self.silu(self.mamba_wea_2(h_emb)) # B,1,12,172
            h_emb = einops.rearrange(h_emb, "B C H W -> B C W H")
            h_emb = self.in_conv_weather3(h_emb) # B,1,7,87
            h_emb = h_emb.view(-1, h_emb.shape[2] * h_emb.shape[3])
            h_emb = self.encoder_weather1(h_emb) # B,256
            h_emb = self.encoder_weather2(h_emb)
            weather_out = self.weather_output(h_emb) # B,512
        elif train_model == 'finetune':
            h_emb = self.in_conv_weather1(h_emb) # B,32,22,342
            h_emb = self.silu(self.mamba_wea_1(h_emb)) # B,8,22,342
            h_emb = self.in_conv_weather2(h_emb) # B,8,12,172
            h_emb = einops.rearrange(h_emb, "B C H W -> B C W H")
            h_emb = self.silu(self.mamba_wea_2(h_emb)) # B,1,12,172
            h_emb = einops.rearrange(h_emb, "B C H W -> B C W H")
            h_emb = self.in_conv_weather3(h_emb) # B,1,7,87
            h_emb = h_emb.view(-1, h_emb.shape[2] * h_emb.shape[3])
            h_emb = self.encoder_weather1(h_emb) # B,512
            h_emb = self.encoder_weather2(h_emb) # B,256
            weather_out = self.weather_output(h_emb) # B,512
        elif train_model == 'clip':
            h_emb = images_condition
            h_emb = self.encoder_weather2(h_emb)

        # u-net part
        h = self.in_conv(h) # B,64,64,1024
        h1 = self.d_block1(h, temb, h_emb) # B,64,64,1024
        h2 = self.d_block2(h1, temb, h_emb) # B,128,32,512
        h3 = self.d_block3(h2, temb, h_emb) # B,256,16,256
        h3_skip = h3

        h3_1 = self.silu(self.ebn1(self.mamba1(h3)))
        h3 = einops.rearrange(h3, "B C H W -> B C W H")
        h3_2 = self.silu(self.ebn1(self.mamba1(h3)))
        h3_2 = einops.rearrange(h3_2, "B C H W -> B C W H")
        h3 = h3_1 * 0.5 + h3_2 * 0.5

        h4 = self.d_block4(h3, temb, h_emb) # B,512,8,128
        h4 = self.silu(self.ebn2(self.mamba2(h4)))
        # up-sampling   8-16 add mamba
        h = self.u_block4(h4, temb, h_emb) # B,256,16,256
        h = self.silu(self.ebn3(self.mamba3(h)))
        h = self.u_block3(_join(h, h3_skip), temb, h_emb) # B,128,32,512
        h = self.silu(self.ebn4(self.mamba4(h)))
        h = self.u_block2(_join(h, h2), temb, h_emb) # B,64,64,1024
        h = self.u_block1(_join(h, h1), temb, h_emb) # B,64,64,1024
        h = self.out_conv(h) # B,64,64,1024

        ########################################################
        if train_model == 'train':
            x_0_ab = (images - b * h) / a
            # has_nan = torch.isnan(x_0_ab).any()
            # print("Contains NaN:", has_nan)
            x_0_ab = self.in_conv_vae(x_0_ab)
            x_0_ab = self.in_conv_vae2(x_0_ab)
            x_0_ab = self.in_conv_vae3(x_0_ab)
            s1, s2, s3, s4 = x_0_ab.size()
            x_0_ab = x_0_ab.view(-1, x_0_ab.shape[2] * x_0_ab.shape[3])
            x_0_ab = self.encoder_vae(x_0_ab)
            mu1 = self.vae_mu(x_0_ab)
            sigma1 = self.vae_sigma(x_0_ab)
            eps1 = torch.randn_like(sigma1)
            z1 = mu1 + eps1 * torch.sqrt(torch.exp(sigma1))
            z1 = self.vae_linear1(z1)
            z1 = self.vae_linear2(z1)
            z1 = self.vae_linear3(z1)
            x_0_domain1 = z1.view(s1, s2, s3, s4)

            weather = self.in_conv_vae(weather)
            weather = self.in_conv_vae2(weather)
            weather = self.in_conv_vae3(weather)
            w1, w2, w3, w4 = weather.size()
            weather = weather.view(-1, weather.shape[2] * weather.shape[3])
            weather = self.encoder_vae(weather)
            mu2 = self.vae_mu(weather)
            sigma2 = self.vae_sigma(weather)
            eps2 = torch.randn_like(sigma2)
            z2 = mu2 + eps2 * torch.sqrt(torch.exp(sigma2))
            z2 = self.vae_linear1(z2)
            z2 = self.vae_linear2(z2)
            z2 = self.vae_linear3(z2)
            weather_domain2 = z2.view(w1, w2, w3, w4)
        elif train_model == 'finetune':
            x_0_ab = (images - b * h) / a
            x_0_ab = self.in_conv_vae(x_0_ab)
            x_0_ab = self.in_conv_vae2(x_0_ab)
            x_0_ab = self.in_conv_vae3(x_0_ab)
            s1, s2, s3, s4 = x_0_ab.size()
            x_0_ab = x_0_ab.view(-1, x_0_ab.shape[2] * x_0_ab.shape[3])
            x_0_ab = self.encoder_vae(x_0_ab)
            mu1 = self.vae_mu(x_0_ab)
            sigma1 = self.vae_sigma(x_0_ab)
            eps1 = torch.randn_like(sigma1)
            z1 = mu1 + eps1 * torch.sqrt(torch.exp(sigma1))
            z1 = self.vae_linear1(z1)
            z1 = self.vae_linear2(z1)
            z1 = self.vae_linear3(z1)
            x_0_domain1 = z1.view(s1, s2, s3, s4)

            weather = self.in_conv_vae(weather)
            weather = self.in_conv_vae2(weather)
            weather = self.in_conv_vae3(weather)
            w1, w2, w3, w4 = weather.size()
            weather = weather.view(-1, weather.shape[2] * weather.shape[3])
            weather = self.encoder_vae(weather)
            mu2 = self.vae_mu(weather)
            sigma2 = self.vae_sigma(weather)
            eps2 = torch.randn_like(sigma2)
            z2 = mu2 + eps2 * torch.sqrt(torch.exp(sigma2))
            z2 = self.vae_linear1(z2)
            z2 = self.vae_linear2(z2)
            z2 = self.vae_linear3(z2)
            weather_domain2 = z2.view(w1, w2, w3, w4)
        elif train_model == 'clip':
            weather_out = 0
            x_0_domain1 = 0
            weather_domain2 = 0

        return h, weather_out, x_0_domain1, weather_domain2
