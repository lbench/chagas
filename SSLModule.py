import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from xresnet18 import *
from dataloader import *


class SSLModule(pl.LightningModule):
    def __init__(self, enconder, loss_fn=None, lr=1e-3, temperature=0.5):
        super().__init__()
        self.encoder = enconder
        self.lr = lr
        self.temperature = temperature
        self.loss_fn = loss_fn
    
    def training_step(self, batch, batch_idx):
        (x1, _), (x2, _) = self.trainer.datamodule.augmentation((batch))
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        loss = self.loss_fn(z1, z2, temperature=self.temperature)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return [optimizer], [scheduler]
