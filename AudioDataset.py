import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import soundfile as sf
import torchaudio.transforms as T


# Global transforms for spectrogram generation
mel_transform = T.MelSpectrogram(sample_rate=16000, n_mels=64)
db_transform = T.AmplitudeToDB()

class AudioDataset(Dataset):
    def __init__(
        self,
        root_dir,
        cache_dir=None,
        preprocess=False,
        transform=None
    ):
        """
        Args:
            root_dir (str): Path to the dataset root. If it has subfolders, each is a class.
            cache_dir (str, optional): Directory to store/load precomputed spectrograms (.pt files).
            preprocess (bool): If True, compute and cache spectrograms on initialization.
            transform (callable, optional): Further transforms applied on the spectrogram.
        """
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        self.preprocess = preprocess
        self.transform = transform
        self.filepaths = []
        self.labels = []
        self.label2idx = {}
        self.idx2label = {}

        # Allowed classes
        allowed = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", 'silence'}

        # Prepare cache directory if needed
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Detect subdirectories
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        if subdirs:
            allowed_in_train = sorted([d for d in subdirs if d in allowed])
            has_unknown = any(d not in allowed for d in subdirs)

            idx = 0
            for lbl in allowed_in_train:
                self.label2idx[lbl] = idx
                self.idx2label[idx] = lbl
                idx += 1
            if has_unknown:
                self.label2idx["unknown"] = idx
                self.idx2label[idx] = "unknown"

            for lbl in subdirs:
                folder = os.path.join(root_dir, lbl)
                mapped_lbl = lbl if lbl in allowed else "unknown"
                label_idx = self.label2idx[mapped_lbl]
                for f in os.listdir(folder):
                    if f.endswith('.wav'):
                        full_path = os.path.join(folder, f)
                        self.filepaths.append(full_path)
                        self.labels.append(label_idx)
        else:
            # Single-class root
            lbl = os.path.basename(root_dir)
            mapped_lbl = lbl if lbl in allowed else "unknown"
            self.label2idx[mapped_lbl] = 0
            self.idx2label[0] = mapped_lbl
            for f in os.listdir(root_dir):
                if f.endswith('.wav'):
                    self.filepaths.append(os.path.join(root_dir, f))
                    self.labels.append(0)

        # Precompute spectrograms if requested
        if self.cache_dir and self.preprocess:
            for fp in self.filepaths:
                self._preprocess_file(fp)

    def _preprocess_file(self, filepath):
        """
        Compute spectrogram for a file and save it to cache_dir with .pt extension.
        """
        base = os.path.splitext(os.path.basename(filepath))[0]
        cache_path = os.path.join(self.cache_dir, f"{base}.pt")
        if not os.path.exists(cache_path):
            data, sr = sf.read(filepath)
            waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            spec = mel_transform(waveform)
            spec_db = db_transform(spec)
            torch.save(spec_db, cache_path)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        """
        Returns:
            spectrogram (Tensor): [1, n_mels, time]
            label (int)
        """
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        if self.cache_dir:
            # Load precomputed
            base = os.path.splitext(os.path.basename(filepath))[0]
            cache_path = os.path.join(self.cache_dir, f"{base}.pt")
            spec_db = torch.load(cache_path)
        else:
            # Compute on-the-fly
            data, sr = sf.read(filepath)
            waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            spec = mel_transform(waveform)
            spec_db = db_transform(spec)

        if self.transform:
            spec_db = self.transform(spec_db)

        return spec_db, label


def pad_collate_fn(batch):
    """
    Collate function to pad spectrograms in a batch to the same time dimension.
    batch: list of (spec [1, n_mels, T], label)
    """
    specs, labels = zip(*batch)
    max_time = max(s.size(2) for s in specs)
    padded = []
    for s in specs:
        pad_amt = max_time - s.size(2)
        padded.append(F.pad(s, (0, pad_amt), mode='constant', value=0))
    batch_specs = torch.stack(padded)
    batch_labels = torch.tensor(labels)
    return batch_specs, batch_labels