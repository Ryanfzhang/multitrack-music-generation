import torch
from tqdm import tqdm
import yaml
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader
from timm.utils import AverageMeter
from audioldm_eval import EvaluationHelper
import wandb
import os
import torchaudio
import soundfile as sf

from dataset.dataset import MultiSource_Slakh_Dataset
from model.autoencoder import AutoencoderKL
from model.unet import UNetModel
from model.musicldm import MusicLDM
import json
from accelerate import Accelerator
from diffusers.optimization import get_cosine_schedule_with_warmup

import argparse

def main(args):
    
    train_dataset = MultiSource_Slakh_Dataset(dataset_path="/home/mafzhang/data/slakh2100/train", label_path=None, config=yaml.safe_load(open("./dataset/dataconfig.yaml")))
    test_dataset = MultiSource_Slakh_Dataset(dataset_path="/home/mafzhang/data/slakh2100/test", label_path=None, config=yaml.safe_load(open("./dataset/dataconfig.yaml")))

    train_dloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_dloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MusicLDM(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps= 1000, 
    num_training_steps=len(train_dloader) * args.epochs,
    )
    accelerator = Accelerator()
    model, optimizer, train_dloader, test_dloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dloader, test_dloader, lr_scheduler
    )
    
    #log
    logdir = os.path.join(args.logdir, args.task)
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(os.path.join(logdir, "val"), exist_ok=True)
    os.makedirs(os.path.join(logdir, "target"), exist_ok=True)


    wandb.init(config=args,
               project="multisource-music-generation",
               name="{}".format(args.task),
               dir=args.logdir,
               job_type="training")


    for epoch in range(args.epochs):
        loss_meter = AverageMeter()
        model.train()
        for step, batch in tqdm(enumerate(train_dloader), total=len(train_dloader)):
            loss = model.module.trainstep(batch['mel'], batch['mel_mix'], batch['waveform_mix'])
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_meter.update(loss.item(), batch["mel"].shape[0])

        print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_meter.avg}")
        wandb.log({"train/loss": loss_meter.avg}, step=epoch*len(train_dloader))

        if epoch % 10 == 0:
            model.eval()
            for stem in range(args.nstems):
                os.makedirs(os.path.join(logdir, "val/epoch_{}/stem_{}".format(epoch, stem)), exist_ok=True)
                os.makedirs(os.path.join(logdir, "target/stem_{}".format(stem)), exist_ok=True)

            with torch.no_grad():
                log_dir = {}

                for test_step, test_batch in tqdm(enumerate(test_dloader), total=len(test_dloader)):
                    val_mels, val_waveforms = model.module.generate(args.nsamples*test_batch['mel'].shape[0])
                    val_waveforms = accelerator.gather(val_waveforms)
                    target_waveforms = accelerator.gather(test_batch['waveform'])
                    
                    for i in range(val_waveforms.shape[0]):
                        for stem in range(args.nstems):
                            log_dir['val_stem_{}_sample_{}'.format(stem, test_step*val_waveforms.shape[0]+i)] = wandb.Audio(val_waveforms[i, stem].detach().cpu().numpy(), sample_rate=16000)
                            torchaudio.save(os.path.join(logdir, "val/epoch_{}/stem_{}".format(epoch, stem), "{}.wav".format(test_step*val_waveforms.shape[0]+i)), val_waveforms[i, stem].detach().cpu(), 16000)
                            torchaudio.save(os.path.join(logdir, "target/stem_{}".format(stem), "{}.wav".format(test_step*val_waveforms.shape[0]+i)), target_waveforms[i, stem].detach().cpu().unsqueeze(0), 16000)
                
                wandb.log(log_dir, step=epoch*len(train_dloader))

                # Evaluate
                if accelerator.is_main_process:
                    target_dir = os.path.join(logdir, "target/")
                    val_dir = os.path.join(logdir, "val/epoch_{}".format(epoch))
                    metrics_buffer = {}
                    accelerator.print("Evaluating...")
                    for stem in range(4):
                        evaluator = EvaluationHelper(16000, accelerator.device)
                        metrics = evaluator.main(str(target_dir)+"/stem_{}".format(stem), str(val_dir)+"/stem_{}".format(stem))
                        metrics_buffer = {
                                ("val/stem_{}/".format(stem) + k): float(v) for k, v in metrics.items()
                            }

                    wandb.log(metrics_buffer, step=epoch*len(train_dloader))
                    accelerator.print(metrics_buffer)
                    accelerator.print("Evaluation finished.")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--nstems", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--cosine_s", type=float, default=0.008)
    parser.add_argument("--path", type=str, default="/home/mafzhang/code/music-generation/")
    parser.add_argument("--logdir", type=str, default="/home/mafzhang/code/music-generation/logs/")
    parser.add_argument("--task", type=str, default="origin")
    parser.add_argument("--training", type=int, default=1)
    parser.add_argument("--parameterization", type=str, default="x0")
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--linear_start", type=float, default=0.0015)
    parser.add_argument("--linear_end", type=float, default=0.0195)
    args = parser.parse_args()
    main(args)
