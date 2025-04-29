import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from helper_code2 import *
import pytorch_lightning as pl



class ToTensor:
    def __call__(self, array):
        return torch.from_numpy(array).T.float() # (T, C) -> (C, T)


class NormalizeECG:

    def __call__(self, sample, eps=1e-7):
        mean = sample.mean(dim=0, keepdim=True)
        std = sample.std(dim=0, keepdim=True)
        result = (sample - mean) / (std + eps)
        return result

class PadECG:

    def __init__(self, pad_to=4096):
        self.pad_to = pad_to

    def __call__(self, sample):
        if sample.shape[-1] >= self.pad_to:
            return sample[:, :self.pad_to] 
        else:
            padding = (0, self.pad_to - sample.shape[1], 0, 0)
            data = F.pad(sample, padding, "constant", 0)
            return data
        
class ResizeECG:
    def __init__(self, out_size=4096):
        self.out_size = out_size
        
    def __call__(self, sample):
        if sample.shape[-1] >= self.out_size:
            return sample[:, :self.out_size]
        else:
            resized = F.interpolate(sample.unsqueeze(0), size=self.out_size, mode='linear')
            return resized.squeeze()


class FolderDataset(Dataset):
    def __init__(self, folder, transform=None, min_len=800):
        """
        Args:
            folder (str): Path to the folder containing the .dat and .hea pairs.
        """
        self.folder = folder
        self.min_len = min_len

        self.transform = transform

        self.record_paths, self.labels = self.find_records()
        self.remove_short()


    def find_records(self):
        root = Path(self.folder)

        records = []
        for p in root.rglob('*.dat'):
            p = p.with_suffix('')
            header = load_header(p)
            label = get_label(header)
            records.append([p, label])

        paths, labels = zip(*records)
        return list(paths), list(labels)


    def remove_short(self):
        i = 0
        while i < len(self.record_paths):
            path = self.record_paths[i]
            signal, fields = load_signals(str(path))
            signal_len = signal.shape[0]
            if signal_len < self.min_len:
                self.record_paths.pop(i)
                self.labels.pop(i)
            else:
                i += 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        record = self.record_paths[idx]
        signal, fields = load_signals(record)

        if self.transform:
            signal = self.transform(signal)

        return signal, self.labels[idx]
    
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class DataModule(pl.LightningDataModule):
    def __init__(self, path, transformation=None, augmentation=None, batchsize=64):
        super().__init__()
        self.path = path
        self.batchsize = batchsize
        self.transformation = transformation
        self.augmentation = augmentation
        
    def setup(self, stage=None):
        self.train_dataset = FolderDataset(self.path, transform=self.transformation)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batchsize, shuffle=True)