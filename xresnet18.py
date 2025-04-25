import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.conv1d1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv1d2 = nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.shorcut = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride)
            nn.BatchNorm1d(out_channel)
        )
        
    def forward(self, x):
        output = self.relu(self.bn1(self.conv1d1(x)))
        output = self.bn2(self.conv1d2(output))
        output += self.shorcut(x)
        output = self.relu(output)
        return output
    
class XResNet18(nn.Module):
    def __init__(self, in_channel=12, layers=[2, 2, 2, 2]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channel, 32, kernel_size=3, stride=2, padding=1)
            nn.BatchNorm1d(32)
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
            nn.BatchNorm1d(32)
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        )
        
# TO: Dataset function, Dataloader (allocate function). Loss function. Training loop.