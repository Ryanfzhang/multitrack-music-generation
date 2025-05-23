import sys

import csv
import json
import wave
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import dataset.audio as Audio
import librosa
import os
import torchvision
import yaml
import pandas as pd
import omegaconf

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, "r") as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row["mid"]] = row["index"]
            line_count += 1
    return index_lookup

class TextDataset(Dataset):
    def __init__(self, data, logfile):
        super().__init__()
        self.data = data
        self.logfile = logfile
    def __getitem__(self, index):
        data_dict = {}
         # construct dict
        data_dict['fname'] = f"infer_file_{index}"
        data_dict['fbank'] = np.zeros((1024,64))
        data_dict['waveform'] = np.zeros((32000))
        data_dict['text'] = self.data[index]
        if index == 0:
            with open(os.path.join(self.logfile), 'w') as f:
                f.write(f"{data_dict['fname']}: {data_dict['text']}")
        else:
            with open(os.path.join(self.logfile), 'a') as f:
                f.write(f"\n{data_dict['fname']}: {data_dict['text']}")
        return data_dict


    def __len__(self):
        return len(self.data)


class AudiostockDataset(Dataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__()

        self.train = train
        self.config = config


        self.melbins = config["preprocessing"]["mel"]["n_mel_channels"]
        self.freqm = config["preprocessing"]["mel"]["freqm"]
        self.timem = config["preprocessing"]["mel"]["timem"]
        self.mixup = config["augmentation"]["mixup"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hopsize = config["preprocessing"]["stft"]["hop_length"]
        self.target_length = config["preprocessing"]["mel"]["target_length"]
        self.use_blur = config["preprocessing"]["mel"]["blur"]
        self.segment_length = int(self.target_length * self.hopsize)
        self.whole_track = whole_track

        self.data = []
        if type(dataset_path) is str:
            self.data = self.read_datafile(dataset_path, label_path, train) 

        elif type(dataset_path) is list or type(dataset_path) is omegaconf.listconfig.ListConfig:
            for datapath in dataset_path:
                self.data +=  self.read_datafile(datapath, label_path, train) 
   
        else:
            raise Exception("Invalid data format")
        print("Data size: {}".format(len(self.data)))

        self.total_len = int(len(self.data) * factor)

        try:
            self.segment_size = config["preprocessing"]["audio"]["segment_size"]
            self.target_length = int(self.segment_size / self.hopsize)
            self.segment_length = int(self.target_length * self.hopsize)
            assert self.segment_size % self.hopsize == 0
            print("Use segment size of %s." % self.segment_size)
        except:
            self.segment_size = None
        
        if not train:
            self.mixup = 0.0
            self.freqm = 0
            self.timem = 0

        self.return_all_wav = False
        if self.mixup > 0:
            self.tempo_map = np.load(config["path"]["tempo_map"], allow_pickle=True).item()
            self.tempo_folder = config["path"]["tempo_data"]
        
        if self.mixup > 1:
            self.return_all_wav = config["augmentation"]["return_all_wav"] 

        # print("Use mixup rate of %s; Use SpecAug (T,F) of (%s, %s); Use blurring effect or not %s" % (self.mixup, self.timem, self.freqm, self.use_blur))

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

        print(f'| Audiostock Dataset Length:{len(self.data)} | Epoch Length: {self.total_len}')

    def read_datafile(self, dataset_path, label_path, train):
        data = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
        if (not train) and len(data) > 2000:
            data_dict = {}
            filelist = [os.path.basename(f).split('.')[0].split('_') for f in self.data]
            for f,idx in filelist:
                if f not in data_dict:
                    data_dict[f] = int(idx)
                else:
                    data_dict[f] = max(int(idx), data_dict[f])
            data = [os.path.join(dataset_path, f'{k}_{data_dict[k] // 2}.wav') for k in data_dict.keys()] + \
                [os.path.join(dataset_path, f'{k}_0.wav') for k in data_dict.keys()] + \
                [os.path.join(dataset_path, f'{k}_{data_dict[k]}.wav') for k in data_dict.keys()]

        
        self.label = []
        if label_path is not None:
            for d in data:
                lp = os.path.join(label_path, os.path.basename(d).split('.')[0] + '.json')
                assert os.path.exists(lp), f'the label file {lp} does not exists.'
                self.label.append(lp)   

        return data 
                
    def random_segment_wav(self, x):
        wav_len = x.shape[-1]
        assert wav_len > 100, "Waveform is too short, %s" % wav_len
        if self.whole_track:
            return x
        if wav_len - self.segment_length > 0:
            if self.train:
                sta = random.randint(0, wav_len -self.segment_length)
            else:
                sta = (wav_len - self.segment_length) // 2
            x = x[:, sta: sta + self.segment_length]
        return x
    
    def normalize_wav(self, x):
        x = x[0]
        x = x - x.mean()
        x = x / (torch.max(x.abs()) + 1e-8)
        x = x * 0.5
        x = x.unsqueeze(0)
        return x

    def read_wav(self, filename):
        y, sr = torchaudio.load(filename)
        # resample
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)
        # normalize
        y = self.normalize_wav(y)
        # segment
        y = self.random_segment_wav(y)
        # pad
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y

    def get_mel(self, filename, mix_filename = None):
        # mixup
        if mix_filename is None:
            y = self.read_wav(filename)
        else:
            # get name 
            anchor_name = os.path.basename(filename)
            target_name = os.path.basename(mix_filename)
            # load wav
            anchor_wav, asr = torchaudio.load(filename)
            target_wav, tsr = torchaudio.load(mix_filename)
            assert asr == tsr, f'mixup sample rate should be the same {asr} vs. {tsr}'
            # get downbeat
            anchor_downbeat = np.load(os.path.join(self.tempo_folder, f'{anchor_name.split(".")[0]}_downbeat_pred.npy'), allow_pickle=True)
            target_downbeat = np.load(os.path.join(self.tempo_folder, f'{target_name.split(".")[0]}_downbeat_pred.npy'), allow_pickle=True)
            
            if len(anchor_downbeat) > 1 and len(target_downbeat) > 1:
                adp = int(anchor_downbeat[np.random.randint(0, len(anchor_downbeat) - 1)] * asr) 
                tdp = int(target_downbeat[np.random.randint(0, len(target_downbeat) - 1)] * tsr)
                anchor_wav = anchor_wav[..., adp:]
                target_wav = target_wav[..., tdp:]
                mix_len = min(anchor_wav.size(-1), target_wav.size(-1))
                if mix_len <= 100:
                    mix_wav, _ = torchaudio.load(filename)
                    anchor_wav = mix_wav[::]
                    target_wav = mix_wav[::]
                else:
                    anchor_wav = anchor_wav[..., :mix_len]
                    target_wav = target_wav[..., :mix_len]
                    p = np.random.beta(5,5)
                    mix_wav = p * anchor_wav + (1-p) * target_wav
            else:
                mix_wav = anchor_wav
                # normalize
            if self.return_all_wav:
                anchor_wav = self.normalize_wav(anchor_wav)
                target_wav = self.normalize_wav(target_wav)
                anchor_wav = anchor_wav[..., :self.segment_length]
                target_wav = target_wav[..., :self.segment_length]
                anchor_wav = torch.nn.functional.pad(anchor_wav, (0, self.segment_length - anchor_wav.size(1)), 'constant', 0.)
                target_wav = torch.nn.functional.pad(target_wav, (0, self.segment_length - target_wav.size(1)), 'constant', 0.)
                # get mel
                anchor_melspec, _, _ = self.STFT.mel_spectrogram(anchor_wav)
                anchor_melspec = anchor_melspec[0].T
                target_melspec, _, _ = self.STFT.mel_spectrogram(target_wav)
                target_melspec = target_melspec[0].T

                if anchor_melspec.size(0) < self.target_length:
                    anchor_melspec = torch.nn.functional.pad(anchor_melspec, (0,0,0,self.target_length - anchor_melspec.size(0)), 'constant', 0.)
                else:
                    anchor_melspec = anchor_melspec[0: self.target_length, :]
                
                if anchor_melspec.size(-1) % 2 != 0:
                    anchor_melspec = anchor_melspec[:, :-1]
                
                if target_melspec.size(0) < self.target_length:
                    target_melspec = torch.nn.functional.pad(target_melspec, (0,0,0,self.target_length - target_melspec.size(0)), 'constant', 0.)
                else:
                    target_melspec = target_melspec[0: self.target_length, :]

                if target_melspec.size(-1) % 2 != 0:
                    target_melspec = target_melspec[:, :-1]
                
                mix_wav, _ = torchaudio.load(filename) # unmix one for latent mixup

            y = self.normalize_wav(mix_wav)
            y = self.random_segment_wav(y)
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        
        # get mel
        y.requires_grad=False
        melspec, _, _ = self.STFT.mel_spectrogram(y)
        melspec = melspec[0].T
        if melspec.size(0) < self.target_length:
            melspec = torch.nn.functional.pad(melspec, (0,0,0,self.target_length - melspec.size(0)), 'constant', 0.)
        else:
            if not self.whole_track:
                melspec = melspec[0: self.target_length, :]
        if melspec.size(-1) % 2 != 0:
            melspec = melspec[:, :-1]
        
        if self.return_all_wav:
            if mix_filename is None:
                anchor_melspec = melspec
                target_melspec = melspec

            return y[0].numpy(), melspec.numpy(), anchor_melspec.numpy(), target_melspec.numpy()
        else:
            return y[0].numpy(), melspec.numpy()


    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]
        lf = self.label[idx] if len(self.label) > 0 else None
        # mixup
        if random.random() < self.mixup:
            wav_folder = os.path.dirname(f)
            anchor_name = os.path.basename(f)
            tempo_group = self.tempo_map['tempo'][self.tempo_map['map'][anchor_name]]
            if len(tempo_group) <= 1:
                mix_f = None
            else:
                mix_f = np.random.choice(tempo_group)
                mix_f = os.path.join(wav_folder, mix_f)
        else:
            mix_f = None
        # get data
        if self.return_all_wav:
            waveform, fbank,fbank_1, fbank_2 = self.get_mel(f, mix_f)
            data_dict['fbank_1'] = fbank_1
            data_dict['fbank_2'] = fbank_2
        else:
            waveform, fbank = self.get_mel(f, mix_f)
        if lf is not None:
            with open(lf, 'r') as lff:
                label_data = json.load(lff)
                text = label_data['text'][0]
        else:
            text = ""
        # construct dict
        data_dict['fname'] = os.path.basename(f).split('.')[0]
        data_dict['fbank'] = fbank
        data_dict['waveform'] = waveform
        data_dict['text'] = text

        ### adding this just to make it artificially compatible with multicanel
        audio_list = []
        fbank_list = []
        for stem in self.config["path"]["stems"]:
            audio_list.append(np.zeros_like(waveform)[np.newaxis, :])  # Expand dims for audio
            fbank_list.append(np.zeros_like(fbank)[np.newaxis, :])  # Expand dims for fbank

        
        # construct dict
        data_dict['fbank_stems'] = np.concatenate(fbank_list, axis=0)
        data_dict['waveform_stems'] = np.concatenate(audio_list, axis=0)

        return data_dict

    def __len__(self):
        # return self.total_len
        return 12



class DS_10283_2325_Dataset(AudiostockDataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__(dataset_path, label_path, config, train = train, factor = factor, whole_track = whole_track)  

    def read_datafile(self, dataset_path, label_path, train):
        file_path = dataset_path
        # Open the file and read lines
        with open(file_path, "r") as fp:
            data_json = json.load(fp)
            data = data_json["data"]

        dataset_directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        # dataset_name = "train" #filename.split('-')[1]

        wav_directory = os.path.abspath(os.path.join(dataset_directory, "wav_files"))

        for entry in data:

            # get audio data path
            prompt_file_path = os.path.join(wav_directory, entry["audio_prompt"])
            response_file_path = os.path.join(wav_directory, entry["audio_response"])
            entry['wav'] = prompt_file_path
            entry["response"] = response_file_path

        # self.data = data
        self.label = data
        return data

    def read_wav(self, filename, frame_offset):

        audio_data, sr =torchaudio.load(filename, frame_offset =  frame_offset*48000, num_frames = 480000)

        # resample
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(audio_data, sr, self.sampling_rate)
        # normalize
        y = self.normalize_wav(y)
        # segment
        y = self.random_segment_wav(y)
        # pad
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y
    
    def get_mel(self, filename, mix_filename = None, frame_offset = 0):
        # mixup
        y = self.read_wav(filename, frame_offset)
        
        # get mel
        y.requires_grad=False
        melspec, _, _ = self.STFT.mel_spectrogram(y)
        melspec = melspec[0].T
        if melspec.size(0) < self.target_length:
            melspec = torch.nn.functional.pad(melspec, (0,0,0,self.target_length - melspec.size(0)), 'constant', 0.)
        else:
            if not self.whole_track:
                melspec = melspec[0: self.target_length, :]
        if melspec.size(-1) % 2 != 0:
            melspec = melspec[:, :-1]
        
        if self.return_all_wav:
            if mix_filename is None:
                anchor_melspec = melspec
                target_melspec = melspec

            return y[0].numpy(), melspec.numpy(), anchor_melspec.numpy(), target_melspec.numpy()
        else:
            return y[0].numpy(), melspec.numpy()
            
    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]
        lf = self.label[idx]
        if lf is not None:
            text = lf['text']
        else:
            text = ""        
        
        
        prompt, fbank_prompt = self.get_mel(f["wav"], None, f["frame_offset"])
        response, fbank_response = self.get_mel(f["response"], None, f["frame_offset"])

        # construct dict
        data_dict['fname'] = os.path.basename(lf['text']).split('.')[0]+"_from_"+str(f["frame_offset"])
        data_dict['fbank_prompt'] = fbank_prompt
        data_dict['prompt'] = prompt
        data_dict['text'] = text

        data_dict['fbank'] = fbank_response
        data_dict['waveform'] = response

        return data_dict



class MultiSource_Slakh_Dataset(DS_10283_2325_Dataset):
    def __init__(self, dataset_path, label_path, config, train = True, factor = 1.0, whole_track = False) -> None:
        super().__init__(dataset_path, label_path, config, train = train, factor = factor, whole_track = whole_track)  

        self.text_prompt = config.get('path', {}).get('text_prompt', None)
        self.stem_masking = config.get('augmentation', {}).get('masking', False)

    def get_duration_sec(self, file, cache=False):

        if not os.path.exists(file):
            # File doesn't exist, return 0
            return 0
        try:
            # Attempt to read cached duration from a file
            with open(file + ".dur", "r") as f:
                duration = float(f.readline().strip("\n"))
        except FileNotFoundError:
            # If cached duration is not found, use torchaudio to find the actual duration
            audio_info = torchaudio.info(file)
            duration = audio_info.num_frames / audio_info.sample_rate
            if cache:
                # Cache the duration for future use
                with open(file + ".dur", "w") as f:
                    f.write(str(duration) + "\n")
        return duration
    
    def filter(self, tracks, audio_files_dir):
        # Remove files too short or too long
        keep = []
        durations = []
        for track in tracks:
            track_dir = os.path.join(audio_files_dir, track)
            # files = librosa.util.find_files(f"{track_dir}", ext=["mp3", "opus", "m4a", "aac", "wav"])
            # Manually create the list of file paths based on stems defined in the configuration
            files = [os.path.join(track_dir, stem + ".wav") for stem in self.config["path"]["stems"]]
            
            
            # skip if there are no sources per track
            if not files:
                continue
            
            durations_track = np.array([self.get_duration_sec(file, cache=True) * self.config['preprocessing']['audio']['sampling_rate'] for file in files]) # Could be approximate
            
            # skip if there is a source that is shorter than minimum track length
            if (durations_track / self.config['preprocessing']['audio']['sampling_rate'] < 10.24).any():
                continue
            
            # skip if there is a source that is longer than maximum track length
            if (durations_track / self.config['preprocessing']['audio']['sampling_rate'] >= 640.0).any():
                # print("skiping_file:", track)
                continue
            
            # skip if in the track the different sources have different lengths
            if not (durations_track == durations_track[0]).all():
                # print(f"{track} skipped because sources are not aligned!")
                # print(durations_track)
                continue
            keep.append(track)
            durations.append(durations_track[0])
        
        # print(f"sr={self.config['preprocessing']['audio']['sampling_rate']}, min: {10}, max: {600}")
        # print(f"Keeping {len(keep)} of {len(tracks)} tracks")

        return keep, durations, np.cumsum(np.array(durations))

    def read_datafile(self, dataset_path, label_path, train):
        data = []
        # Load list of tracks and starts/durations
        # print(dataset_path)
        tracks = os.listdir(dataset_path)
        # print(f"Found {len(tracks)} tracks.")
        keep, durations, cumsum = self.filter(tracks, dataset_path)

        # Assuming keep, durations, and cumsum are lists of the same length
        for idx in range(len(keep)):
            # Construct a dictionary for each track with its name, duration, and cumulative sum
            track_info = {
                'wav_path': os.path.join(dataset_path,keep[idx]),
                'duration': durations[idx],
                # 'cumsum': cumsum[idx]
            }
            # Append the dictionary to the data list
            data.append(track_info)

        entries_to_remove = []  # List to store entries to be removed
        max_samples = 640.0 * self.config['preprocessing']['audio']['sampling_rate']

        # Temporary list to hold all data including new segments
        temp_data = []

        for entry in data:
            entry['frame_offset'] = 0
            duration = entry['duration']

            # Always add the original entry to temp_data
            temp_data.append(entry)

            # Handle long files by adding new segments immediately after the original entry
            if duration > self.segment_length:
                num_copies = int((min(duration, max_samples) - self.segment_length) / self.segment_length)
                for i in range(num_copies):
                    new_entry = entry.copy()
                    new_entry['frame_offset'] = (i + 1) * self.segment_length
                    temp_data.append(new_entry)  # Add new segment right after the original entry

            # Mark very short files for removal
            if duration < 0.2:
                entries_to_remove.append(entry)

        # Remove the short entries directly from temp_data
        temp_data = [entry for entry in temp_data if entry not in entries_to_remove]

        # Now, temp_data has all the data in the desired order
        data = temp_data

        return data


    def read_wav(self, filename, frame_offset):

        y, sr =torchaudio.load(filename, frame_offset =  int (frame_offset*22050), num_frames = int(22050*10.24)) # taking a little longer sample than 10 sec 

        # resample
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, sr, self.sampling_rate)

        # normalize
        # y = self.normalize_wav(y) ##### don't do this because mix matters!!!!
        # segment
        # y = self.random_segment_wav(y)
            
        # pad
        if not self.whole_track:
            y = torch.nn.functional.pad(y, (0, self.segment_length - y.size(1)), 'constant', 0.)
        return y

    def get_index_offset(self, item):

        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.segment_length // 2
        shift = np.random.randint(-half_interval, half_interval) if self.train else 0
        offset = item["frame_offset"] + shift  # Note we centred shifts, so adding now
        
        start, end = 0.0, item["duration"]  # start and end of current song
        
        # offset = np.random.randint(start, end)  # random shift

        if offset > end - self.segment_length:  # Going over song
            offset = max(start, offset - half_interval)  # Now should fit
        if offset < start:  # Going under zero
            offset = 0.0 # Now should fit
        # assert (
        #         start <= offset <= end - self.segment_length
        # ), f"Offset {offset} not in [{start}, {end - self.segment_length}]. End: {end}, SL: {self.segment_length}, Index: {item}"
        
        offset = offset/self.config['preprocessing']['audio']['sampling_rate']
        return item, offset

    def get_mel_from_waveform(self, waveform):
        # waveform
        y = torch.tensor(waveform).unsqueeze(0) #self.read_wav(filename, frame_offset)
        
        # get mel
        y.requires_grad=False
        melspec, _, _ = self.STFT.mel_spectrogram(y)
        melspec = melspec[0].T
        if melspec.size(0) < self.target_length:
            melspec = torch.nn.functional.pad(melspec, (0,0,0,self.target_length - melspec.size(0)), 'constant', 0.)
        else:
            if not self.whole_track:
                melspec = melspec[0: self.target_length, :]
        if melspec.size(-1) % 2 != 0:
            melspec = melspec[:, :-1]

        return melspec.numpy()

    def mask_audio_channels(self, audio, fbank):
        """
        Randomly masks 0, 1, 2, or 3 channels in a 4-channel audio input and updates the corresponding Mel spectrograms.
        
        Parameters:
        audio (list of np.ndarray): 4-channel audio input, where each sublist represents a channel.
        fbank (list): List to store the Mel spectrograms corresponding to the masked audio.
        
        Returns:
        tuple: (masked_audio, fbank)
            masked_audio (list of np.ndarray): Audio input with randomly masked channels.
            fbank (list of np.ndarray): Updated Mel spectrograms corresponding to the masked audio.
        """
        num_channels = len(audio)
        assert num_channels == 4, "Audio input must have 4 channels."
        
        # Determine the number of channels to mask (0, 1, 2, or 3)
        num_channels_to_mask = random.choice(range(num_channels))
        
        # Select the channels to mask
        channels_to_mask = random.sample(range(num_channels), num_channels_to_mask)
        
        # Create a copy of the audio list to avoid modifying the original input
        masked_audio = [channel.copy() for channel in audio]
        
        # Apply the mask to the selected channels
        for channel in channels_to_mask:
            masked_audio[channel] = np.zeros_like(masked_audio[channel])
        
        # Update the Mel spectrograms in the fbank for the masked channels
        for channel in channels_to_mask:
            fbank[channel] = np.expand_dims(self.get_mel_from_waveform(masked_audio[channel][0]), axis=0)
        
        return masked_audio, fbank


    def __getitem__(self, index):
        idx = index % len(self.data)
        data_dict = {}
        f = self.data[idx]

        index, frame_offset = self.get_index_offset(f)
        # wav = self.get_song_chunk(index, offset)
        # return self.transform(torch.from_numpy(wav))

        audio_list = []
        fbank_list = []
        for stem in self.config["path"]["stems"]:

            audio, fbank = self.get_mel(os.path.join(f["wav_path"],stem+".wav"), None, frame_offset)
            audio_list.append(audio[np.newaxis, :])  # Expand dims for audio
            fbank_list.append(fbank[np.newaxis, :])  # Expand dims for fbank

        
        if self.stem_masking and self.train:
            audio_list, fbank_list = self.mask_audio_channels(audio_list, fbank_list)


        # construct dict
        data_dict['fname'] = f['wav_path'].split('/')[-1]+"_from_"+str(int(frame_offset))
        data_dict['mel'] = np.concatenate(fbank_list, axis=0)
        data_dict['waveform'] = np.concatenate(audio_list, axis=0)
        data_dict['waveform_mix'] = np.clip(np.sum(data_dict['waveform'], axis=0), -1, 1) #np.sum(data_dict['waveform_stems'], axis=0)

        data_dict['mel_mix'] = self.get_mel_from_waveform(data_dict['waveform_mix'])[np.newaxis, :]

        return data_dict



def balance_test():
    import torch
    from tqdm import tqdm
    from pytorch_lightning import Trainer, seed_everything

    from torch.utils.data import WeightedRandomSampler
    from torch.utils.data import DataLoader
    from utilities.data.dataset import Dataset as AudioDataset

    seed_everything(0)

    # train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audioset_freesound_full/datafiles_extra_audio_files_2/audioset_bal_unbal_freesound_train_data.json"
    train_json = "/mnt/fast/nobackup/users/hl01486/metadata/audioset/datafiles/audioset_bal_unbal_train_data.json"

    samples_weight = np.loadtxt(train_json[:-5] + "_weight.csv", delimiter=",")

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    # dataset = AudioDataset(samples_weight = None, train=True)
    dataset = AudioDataset(samples_weight=samples_weight, train=True)

    loader = DataLoader(dataset, batch_size=10, num_workers=8, sampler=sampler)

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        if label_indices_total is None:
            label_indices_total = label_indices
        else:
            label_indices_total += label_indices

        if cnt > 2000:
            break

    np.save(
        "balanced_with_mixup_balance.npy",
        label_indices_total.cpu().detach().numpy() / 2000,
    )
    # np.save("balanced_with_no_mixup_balance.npy", label_indices_total.cpu().detach().numpy())
    ######################################
    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    dataset = AudioDataset(samples_weight=None, train=True)
    # dataset = AudioDataset(samples_weight = samples_weight, train=True)

    loader = DataLoader(dataset, batch_size=10, num_workers=8, sampler=sampler)

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        if label_indices_total is None:
            label_indices_total = label_indices
        else:
            label_indices_total += label_indices

        if cnt > 2000:
            break

    # np.save("balanced_with_mixup_balance.npy", label_indices_total.cpu().detach().numpy())
    np.save(
        "balanced_with_no_mixup_balance.npy",
        label_indices_total.cpu().detach().numpy() / 2000,
    )

    ######################################

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    dataset = AudioDataset(samples_weight=None, train=True)
    # dataset = AudioDataset(samples_weight = samples_weight, train=True)

    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=8,
        # sampler=sampler
    )

    result = []

    label_indices_total = None

    for cnt, each in tqdm(enumerate(loader)):
        (
            fbank,
            log_magnitudes_stft,
            label_indices,
            fname,
            waveform,
            clip_label,
            text,
        ) = each
        if label_indices_total is None:
            label_indices_total = label_indices
        else:
            label_indices_total += label_indices

        if cnt > 2000:
            break

    # np.save("balanced_with_mixup_balance.npy", label_indices_total.cpu().detach().numpy())
    np.save("no_balance.npy", label_indices_total.cpu().detach().numpy() / 2000)


def check_batch(batch):
    import soundfile as sf
    import matplotlib.pyplot as plt

    save_path = "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/stable_diffusion_for_audio/output/temp"
    os.makedirs(save_path, exist_ok=True)
    fbank, log_magnitudes_stft, label_indices, fname, waveform, clip_label, text = batch
    for fb, wv, description in zip(fbank, waveform, text):
        sf.write(
            save_path + "/" + "%s.wav" % description.replace(" ", "_")[:30], wv, 16000
        )
        plt.imshow(np.flipud(fb.cpu().detach().numpy().T), aspect="auto")
        plt.savefig(save_path + "/" + "%s.png" % description.replace(" ", "_")[:30])


if __name__ == "__main__":

    import torch
    from tqdm import tqdm
    from pytorch_lightning import Trainer, seed_everything

    from torch.utils.data import WeightedRandomSampler
    from torch.utils.data import DataLoader


    seed_everything(0)

   

    sampler = WeightedRandomSampler(
        samples_weight, len(samples_weight), replacement=True
    )
    dataset = MultiSource_Slakh_Dataset(
        dataset_path="/home/mafzhang/data/slakh2100/train",
        label_path=None,
        config=yaml.load("./dataconfig.yaml")
    )
    print(dataset.__getitem__(0))
