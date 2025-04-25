import torch
import numpy as np
import random

    
class TimeOut:
    """Set random segment to 0. Input is Tensor in (T,C) form. Output is Tensor in (C,T) form
    """
    def __init__(self, crop_ratio_range=[0.0, 0.5]):
        self.crop_ratio_range = crop_ratio_range
        
    def __call__(self, sample):
        data, label = sample
        data = data.clone()
        timesteps, channels = data.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_timesteps = int(crop_ratio*timesteps)
        start_idx = random.randint(0, timesteps - crop_timesteps-1)
        data[start_idx:start_idx+crop_timesteps, :] = 0
        return data.permute(1, 0), label
    
class RamdomResizeCrop:
    """Random crop and resize to original size. Input is Tensor in (T,C) form. Output is Tensor in (C,T) form
    """
    def __init__(self, crop_ratio_range=[0.5, 1.0]):
        self.crop_ratio_range = crop_ratio_range
        
    def __call__(self, sample):
        data, label = sample
        timesteps, channels = data.shape
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_timesteps = int(crop_ratio*timesteps)
        start = random.randint(0, timesteps - crop_timesteps-1)
        cropped_data = data[start: start + crop_timesteps, :]
        resized = torch.nn.functional.interpolate(cropped_data.permute(1, 0).unsqueeze(0), size=timesteps, mode='linear')
        return resized.squeeze(), label
    