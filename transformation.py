import torch
import numpy as np
import random
import torch.nn.functional as F

    
class TimeOut:
    """Set random segment to 0. Expect Input is Tensor in (B,C,T) form. Output is Tensor in (B,C,T) form.
    """
    def __init__(self, crop_ratio_range=[0.0, 0.5]):
        self.crop_ratio_range = crop_ratio_range
        
    def __call__(self, sample):
        data, label = sample
        data = data.clone()
        timesteps = data.shape[-1]
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_timesteps = int(crop_ratio*timesteps)
        start_idx = random.randint(0, timesteps - crop_timesteps-1)
        if data.dim() == 3:
            data[:, :, start_idx:start_idx+crop_timesteps] = 0
        else:
            data[:, start_idx:start_idx+crop_timesteps] = 0
        return data, label
    
class RandomResizeCrop:
    """Random crop and resize to original size. Input is Tensor in (B,C,T) form. Output is Tensor in (B,C,T) form
    """
    def __init__(self, crop_ratio_range=[0.5, 1.0], output_size=4096):
        self.crop_ratio_range = crop_ratio_range
        self.output_size=output_size
        
    def __call__(self, sample):
        data, label = sample
        timesteps = data.shape[-1]
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_timesteps = int(crop_ratio*timesteps)
        start = random.randint(0, timesteps - crop_timesteps-1)
        if data.dim() == 3:
            cropped_data = data[:, :, start: start + crop_timesteps]
            resized = F.interpolate(cropped_data, size=self.output_size, mode='linear')
            return resized, label
        else:
            cropped_data = data[:, start: start + crop_timesteps]
            resized = F.interpolate(cropped_data.unsqueeze(0), size=self.output_size, mode='linear')
            return resized.squeeze(), label
    
class RandomTransformation:
    """Generate augmentated data.
    """
    def __init__(self, to_range=[0.0, 0.5], rrc_range=[0.5, 1.0]):
        self.to = TimeOut(to_range)
        self.rrc = RandomResizeCrop(rrc_range)
        
    def __call__(self, x):
        z1 = self.to(self.rrc(x))
        z2 = self.to(self.rrc(x))
        return z1, z2
        #return z1