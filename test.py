import torch
from tqdm import tqdm

from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader


from dataset.dataset import MultiSource_Slakh_Dataset
from model.autoencoder import AutoencoderKL
from model.unet import UNetModel
from model.musicldm import MusicLDM
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/mafzhang/code/music-generation/")
parser.add_argument("--training", type=int, default=1)
parser.add_argument("--parameterization", type=str, default="x0")
parser.add_argument("--beta_schedule", type=str, default="linear")
parser.add_argument("--timesteps", type=int, default=1000)
parser.add_argument("--linear_start", type=float, default=0.0015)
parser.add_argument("--linear_end", type=float, default=0.0195)
args = parser.parse_args()
model = MusicLDM(args)
model.to("cuda")
fbanks = torch.randn(2, 4, 1024, 64, device=model.betas.device)
fbanks_mix = torch.randn(2, 1, 1024, 64, device=model.betas.device)
waves = torch.randn(2, 163840, device=model.betas.device)
print(model.trainstep(fbanks, fbanks_mix, waves))
# print(model.generate((2, 4, 8, 256, 256)))
samples = model.generate(nsamples=1)
print(samples)
print(samples.shape)