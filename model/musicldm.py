import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.extend("..")
from functools import partial
from einops import rearrange
from tqdm import tqdm

from model.hifigan import Generator
from model.autoencoder import AutoencoderKL
from model.unet import UNetModel
from model.utils import make_beta_schedule, extract_into_tensor
from model.modules import MixtureGuider

class MusicLDM(nn.Module):

    def __init__(self, config=None):
        super(MusicLDM, self).__init__()
        self.config = config
        self.training = config.training==1
        self.parameterization = config.parameterization
        # x [B, S, C=8, T=256, F=16]
        self.mix_attn = MixtureGuider(
                channels=8*16,
                num_heads=4,
                num_head_channels=32,
            )
        self.unet = UNetModel(config=config)
        
        self.autoencoder = AutoencoderKL(config=config)
        self.autoencoder.requires_grad_(False)
        self.hifigan = Generator(config=config)
        self.hifigan.requires_grad_(False)

        self.config = config

        betas = make_beta_schedule(
                config.beta_schedule,
                config.timesteps,
                linear_start=config.linear_start,
                linear_end=config.linear_end,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = config.linear_start
        self.linear_end = config.linear_end
        assert (
            alphas_cumprod.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.v_posterior = 0.0
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        else:
            raise NotImplementedError("mu not supported")

    def trainstep(self, fbanks, fbanks_mix, waves, set_t=0, is_train=1):

        self.training=1
        B = fbanks.shape[0]

        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.config.timesteps, [B]).to(fbanks.device)
        

        z, z_mix = self.forward(fbanks, fbanks_mix, waves)
        loss = self.p_losses(z, z_mix, t)

        return loss
    
    def forward(self, fbanks, fbanks_mix, waves):
        B = fbanks.shape[0]
        fbanks = rearrange(fbanks, "b c h w -> (b c) h w")
        
        z = self.autoencoder.encode(fbanks.unsqueeze(1))
        z_mix = self.autoencoder.encode(fbanks_mix)
        if self.training:
            z = z.sample()
            z = rearrange(z, "(b c) d h w->b c d h w", b=B)
            z_mix = z_mix.sample()
        else:
            z = z.mean
            z = rearrange(z, "(b c) d h w->b c d h w", b=B)
            z_mix = z_mix.mean
        return z, z_mix

    def make_decision(self, probability):
        if float(torch.rand(1)) < probability:
            return True
        else:
            return False
    
    def q_sample(self, x_start, t, noise=None):
        noise = noise if noise is not None else torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def p_losses(self, latent, latent_mixture, t, noise=None):

        B = latent.shape[0]
        noise = torch.randn_like(latent)
        latent_noisy = self.q_sample(x_start=latent, t=t, noise=noise)

        mask = torch.rand(latent_mixture.size(0), device=latent_mixture.device) < 0.1
        latent_mixture[mask] = 0.0 

        latent_noisy = self.mix_attn(latent_noisy, latent_mixture)
        model_out = self.unet(latent_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = latent
        else:
            raise NotImplementedError(
                f"Paramterization {self.parameterization} not yet supported"
            )

        loss = torch.nn.functional.mse_loss(model_out, target, reduction='none').mean()

        return loss
    
    @torch.no_grad()
    def p_sample(self, x, mix, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        
        mix = torch.zeros_like(mix)
        x = self.mix_attn(latent_noisy=x, latent_mixture=mix)
        x_t = self.unet(x, t)

        if self.parameterization == "eps":
            x_recon = extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

        elif self.parameterization == "x0":
            x_recon = x_t 

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_recon
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )

        repeat_noise = torch.randn((1, *x.shape[1:]), device=device).repeat(x.shape[0], *((1,) * (len(x.shape) - 1)))
        nonzero_mask = (
            (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1))).contiguous()
        )
        return posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance_clipped) * repeat_noise

    @torch.no_grad()
    def generate(self, nsamples, mixture, return_intermediates=False):
        self.training=0
        device = self.betas.device
        b = nsamples
        img = torch.randn((nsamples, 4, 8, 256, 16), device=device)
        
        mixture = self.autoencoder.encode(mixture).mean

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="Sampling t",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                x=img,
                mix=mixture,
                t=torch.full((b,), i, device=device, dtype=torch.long),
                clip_denoised=True,
            )
        
        img = rearrange(img, "b c d h w -> (b c) d h w")
        img = self.autoencoder.decode(img)
        if len(img.size()) == 4:
            mel = img.squeeze(1)
        mel = mel.permute(0, 2, 1)
        waveform = self.hifigan(mel)
        img = rearrange(img, "(b c) d h w -> b c d h w", b=nsamples)
        waveform = rearrange(waveform, "(b c) a f -> b c a f", b=nsamples)

        return img, waveform