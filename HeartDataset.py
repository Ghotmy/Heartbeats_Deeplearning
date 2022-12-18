import os
import torch
from torch.utils.data import Dataset
import torchaudio


class HeartDataset(Dataset):

    def __init__(self, annotations, audio_dir, transformation,
                 target_sample_rate, num_samples, device):

        self.annotations = annotations
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample(signal, sr)
        signal = self._mix_down(signal)
        signal = self._cut(signal)
        signal = self._padding(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _padding(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, self.annotations.iloc[index, 1])
        if self.annotations.iloc[index, 0] == 'b':
            path = path.replace('Btraining_extrastole', 'extrastole_')
            path = path.replace('Btraining_murmur', 'murmur_')
            path = path.replace('Btraining_normal', 'normal_')
        # print(f'PATH OF AUDIO:{path}')
        return path

    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 2]
