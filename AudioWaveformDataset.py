import os
import torch
import torchaudio

from torch.utils.data import Dataset
import torch.nn.functional as F

class AudioWaveformDataset(Dataset):
    def __init__(
        self,
        root_dir,
    ):
        self.root_dir = root_dir
        self.filepaths = []
        self.labels = []
        self.label2idx = {}
        self.idx2label = {}

        # Allowed classes
        allowed = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}

        # Detect subdirectories
        subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        if not subdirs:
            raise ValueError(f"No subdirectories found in {root_dir}. Ensure it contains class folders.")
            
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

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]
        waveform, _ = torchaudio.load(filepath)
        waveform = waveform.squeeze(0)
        return waveform, label


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