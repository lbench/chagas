import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def same_padding(kernel: int, stride: int = 1) -> int:
    return max(0, math.ceil(kernel / 2) - 1) if stride == 1 else \
           max(0, math.ceil((kernel - stride) / 2))


def downsample_ratio(n_in: int, n_out: int) -> int:
    if n_out > n_in:
        raise ValueError("n_samples must monotonically decrease.")
    if n_in % n_out != 0:
        raise ValueError("n_samples should decrease by an *integer* factor.")
    return n_in // n_out


class PreActResBlock1d(nn.Module):
    """
    1‑D pre‑activation residual block:
     (BN→ReLU→Conv→Drop) → (BN→ReLU→Conv→Drop) + shortcut
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        *,
        kernel: int = 17,
        p_drop: float = 0.0,
    ) -> None:
        super().__init__()

        if kernel % 2 == 0:
            raise ValueError("`kernel` must be odd.")

        pad1 = same_padding(kernel, stride=1)
        pad2 = same_padding(kernel, stride=stride)

        # first pre‑act conv
        self.bn1   = nn.BatchNorm1d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel,
                               stride=1, padding=pad1, bias=False)
        self.drop1 = nn.Dropout(p_drop)

        # second pre‑act conv (with stride)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel,
                               stride=stride, padding=pad2, bias=False)
        self.drop2 = nn.Dropout(p_drop)

        # shortcut: max‑pool if downsampling, 1×1 conv if channels change
        shortcut: List[nn.Module] = []
        if stride > 1:
            shortcut.append(nn.MaxPool1d(stride, stride))
        if in_ch != out_ch:
            shortcut.append(nn.Conv1d(in_ch, out_ch, 1, bias=False))
        self.shortcut = nn.Sequential(*shortcut) if shortcut else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.drop1(self.conv1(self.relu1(self.bn1(x))))
        out = self.drop2(self.conv2(self.relu2(self.bn2(out))))

        return out + identity


class PreActResNet1d(nn.Module):
    """
    Pre‑activation ResNet for 1‑D signals.

    input_dim:  (C_in, L_in)
    blocks_dim: [(C1, L1), (C2, L2), ...]
    """
    def __init__(
        self,
        input_dim: Tuple[int, int],
        blocks_dim: List[Tuple[int, int]],
        n_classes: int,
        *,
        kernel_size: int = 17,
        dropout: float   = 0.2,
    ):
        super().__init__()

        c_in,  l_in  = input_dim
        c_out, l_out = blocks_dim[0]

        # stem: no activation here, pre‑act starts in block
        stride0 = downsample_ratio(l_in, l_out)
        pad0    = same_padding(kernel_size, stride=stride0)
        self.stem = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size,
                      stride=stride0, padding=pad0, bias=False),
            nn.BatchNorm1d(c_out),
        )

        # build pre‑act blocks
        stages: List[nn.Module] = []
        in_c, in_l = c_out, l_out
        for out_c, out_l in blocks_dim:
            stride = downsample_ratio(in_l, out_l)
            stages.append(
                PreActResBlock1d(
                    in_c, out_c, stride,
                    kernel=kernel_size, p_drop=dropout
                )
            )
            in_c, in_l = out_c, out_l
        self.blocks = nn.ModuleList(stages)

        # classifier
        self.classifier = nn.Linear(in_c * in_l, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)


# quick shape check
if __name__ == "__main__":
    model = PreActResNet1d((12, 4096),
                           [(64, 4096), (128, 1024), (196, 256), (256, 64), (320, 16)],
                           n_classes=1)
    out = model(torch.randn(8, 12, 4096))
    assert out.shape == (8, 1)
    print("✅ shape is correct")
