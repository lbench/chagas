import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from helper_code2 import *



class ToTensor:
    def __call__(self, array):
        return torch.from_numpy(array).T


class NormalizeECG:

    def __call__(self, sample):
        mean = sample.mean(dim=0, keepdim=True)
        std = sample.std(dim=0, keepdim=True)
        result = (sample - mean) / std
        return result

class PadECG:

    def __init__(self, pad_to=4096):
        self.pad_to = pad_to

    def __call__(self, sample):
        padding = (0, self.pad_to - sample.shape[1], 0, 0)
        data = F.pad(sample, padding, "constant", 0)
        return data




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