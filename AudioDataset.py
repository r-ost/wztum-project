import os
import torch
import soundfile as sf
import torchaudio.transforms as T
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset

# Global transforms for spectrogram generation
mel_transform = T.MelSpectrogram(sample_rate=16000, n_mels=64)
db_transform  = T.AmplitudeToDB()

class AudioDataset(Dataset):
    def __init__(
        self,
        root_dir,
        cache_dir=None,
        export_dir=None,
        preprocess=False,
        transform=None,
        allowed_classes=None,
        save_spectrograms=True
    ):
        """
        Args:
            root_dir (str): Path to the dataset root. If it has subfolders, each is a class.
            cache_dir (str, optional): Directory to store/load precomputed spectrograms (.pt files).
            export_dir (str, optional): Directory to save exported spectrogram images (.png files), preserving class subfolders.
            preprocess (bool): If True, compute and cache spectrograms (and exports) on initialization.
            transform (callable, optional): Further transforms applied on the spectrogram at __getitem__.
            allowed_classes (iterable of str, optional): Which folder names to treat as known classes.
                Defaults to {"yes","no","up","down","left","right","on","off","stop","go","silence"}.
            save_spectrograms (bool): If False, skip saving to disk (both cache and PNG exports).
        """
        self.root_dir          = root_dir
        self.cache_dir         = cache_dir
        self.export_dir        = export_dir
        self.preprocess        = preprocess
        self.transform         = transform
        self.save_spectrograms = save_spectrograms
        # allowed classes set (includes 'silence' by default)
        self.allowed = set(allowed_classes) if allowed_classes else {
            "yes","no","up","down","left","right","on","off","stop","go","silence"
        }

        self.filepaths = []
        self.labels    = []
        self.label2idx = {}
        self.idx2label = {}

        # create cache/export dirs if requested
        if self.save_spectrograms and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        if self.save_spectrograms and self.export_dir:
            os.makedirs(self.export_dir, exist_ok=True)

        # Detect subdirectories for class labels
        subdirs = [d for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))]
        if subdirs:
            # build label mapping
            known = sorted(d for d in subdirs if d in self.allowed)
            idx = 0
            for lbl in known:
                self.label2idx[lbl] = idx
                self.idx2label[idx] = lbl
                idx += 1
            # unknown bucket if necessary
            unknowns = [d for d in subdirs if d not in self.allowed]
            if unknowns:
                self.label2idx["unknown"] = idx
                self.idx2label[idx]    = "unknown"

            # collect file paths & labels
            for lbl in subdirs:
                folder     = os.path.join(root_dir, lbl)
                mapped_lbl = lbl if lbl in self.allowed else "unknown"
                label_idx  = self.label2idx[mapped_lbl]
                if self.save_spectrograms and self.export_dir:
                    os.makedirs(os.path.join(self.export_dir, mapped_lbl), exist_ok=True)
                for f in os.listdir(folder):
                    if f.lower().endswith('.wav'):
                        self.filepaths.append(os.path.join(folder, f))
                        self.labels.append(label_idx)
        else:
            # single-class root directory
            lbl = os.path.basename(root_dir)
            mapped_lbl = lbl if lbl in self.allowed else "unknown"
            self.label2idx[mapped_lbl] = 0
            self.idx2label[0]           = mapped_lbl
            if self.save_spectrograms and self.export_dir:
                os.makedirs(os.path.join(self.export_dir, mapped_lbl), exist_ok=True)
            for f in os.listdir(root_dir):
                if f.lower().endswith('.wav'):
                    self.filepaths.append(os.path.join(root_dir, f))
                    self.labels.append(0)

        # Precompute (cache + export) if requested
        if self.preprocess and self.save_spectrograms and (self.cache_dir or self.export_dir):
            for fp in self.filepaths:
                self._preprocess_file(fp)

    def _preprocess_file(self, filepath):
        """
        Compute mel-spectrogram (in dB) and optionally save:
          - a .pt tensor to cache_dir
          - a .png image to export_dir inside the appropriate class subfolder
        """
        base   = os.path.splitext(os.path.basename(filepath))[0]
        parent = os.path.basename(os.path.dirname(filepath))
        mapped = parent if parent in self.allowed else "unknown"

        # compute spectrogram
        data, _    = sf.read(filepath)
        waveform   = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        spec       = mel_transform(waveform)
        spec_db    = db_transform(spec)

        # save .pt cache
        if self.save_spectrograms and self.cache_dir:
            pt_path = os.path.join(self.cache_dir, f"{base}.pt")
            if not os.path.exists(pt_path):
                torch.save(spec_db, pt_path)

        # save .png export using PIL to reduce memory
        if self.save_spectrograms and self.export_dir:
            img_dir  = os.path.join(self.export_dir, mapped)
            img_path = os.path.join(img_dir, f"{base}.png")
            if not os.path.exists(img_path):
                # normalize spectrogram to 0-255
                arr = spec_db.squeeze(0).numpy()
                min_val, max_val = arr.min(), arr.max()
                norm = 255 * (arr - min_val) / (max_val - min_val + 1e-6)
                norm = norm.astype(np.uint8)
                im = Image.fromarray(norm)
                im.save(img_path)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        """
        Returns:
            spec_db (Tensor): [1, n_mels, time] mel-spectrogram in dB
            label   (int):   class index
        """
        filepath = self.filepaths[idx]
        label    = self.labels[idx]

        # load from cache or compute on the fly
        if self.save_spectrograms and self.cache_dir:
            base     = os.path.splitext(os.path.basename(filepath))[0]
            cache_pt = os.path.join(self.cache_dir, f"{base}.pt")
            spec_db  = torch.load(cache_pt)
        else:
            data, _  = sf.read(filepath)
            waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
            spec_db  = db_transform(mel_transform(waveform))

        if self.transform:
            spec_db = self.transform(spec_db)

        return spec_db, label


def pad_collate_fn(batch):
    """
    Collate function to pad spectrograms in a batch to the same time dimension.
    """
    specs, labels = zip(*batch)
    max_time      = max(s.size(2) for s in specs)
    padded        = []
    for s in specs:
        pad_amt = max_time - s.size(2)
        padded.append(F.pad(s, (0, pad_amt), mode='constant', value=0))
    return torch.stack(padded), torch.tensor(labels)
