import torch
import torch.nn as nn
import torch.nn.functional as F

class XResBlock1d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.conv1d1 = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv1d2 = nn.Conv1d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        if stride != 1 or in_channel != out_channel:
            self.shorcut = nn.Sequential(
                nn.AvgPool1d(kernel_size=2, stride=stride, ceil_mode=True),
                nn.Conv1d(in_channel, out_channel, kernel_size=1),
                nn.BatchNorm1d(out_channel)
            )
        else:
            self.shorcut = nn.Identity()
        nn.init.constant_(self.bn2.weight, 0)
        
    def forward(self, x):
        output = self.relu(self.bn1(self.conv1d1(x)))
        output = self.bn2(self.conv1d2(output))
        output += self.shorcut(x)
        output = self.relu(output)
        return output
    
class XResNet18(nn.Module):
    def __init__(self, in_channel=12, out_channel=64, layers=[2, 2, 2, 2]):
        super().__init__()
        # Encoder
        self.stem = nn.Sequential(
            nn.Conv1d(in_channel, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.stem_pool = nn.MaxPool1d(3,2, padding=1)
        
        self.block1 = self.make_layer(64, 64, layers[0])
        self.block2 = self.make_layer(64, 128, layers[1], stride=2)
        self.block3 = self.make_layer(128, 256, layers[2], stride=2)
        self.block4 = self.make_layer(256, 512, layers[3], stride=2)
        
        # Projector
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(512, out_channel)
        
    def make_layer(self, in_channel, out_channel, n_block, stride=1):
        blocks = [XResBlock1d(in_channel, out_channel, stride=stride)]
        for _ in range(n_block-1):
            blocks.append(XResBlock1d(out_channel, out_channel, stride=1))
        return nn.Sequential(*blocks)
    
    def forward_encoder(self, x):
        out = self.stem_pool(self.stem(x))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        return out
    
    def forward_projection(self, feature):
        out = self.avgpool(feature)
        out = self.projection(out.squeeze(-1))
        return F.normalize(out, dim=1)
    
    def forward(self, x):
        feature = self.forward_encoder(x)
        out = self.forward_projection(feature)
        return out
        
        
# TO: Dataset function, Dataloader (allocate function). Training loop.