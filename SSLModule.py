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
    
class ClassifierModule(pl.LightningModule):
    def __init__(self, encoder, out_dim=1, lr=1e-3, epochs=10, pos_weight=1, linear=False, frozen=True):
        super().__init__()
        self.encoder = encoder
        self.lr = lr
        if linear:
            self.classifier = nn.Linear(512, out_dim)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, out_dim)
                )
        self.epochs = epochs
        self.best_val_loss = 1
        self.frozen = frozen 
        
        # Freeze encoder 
        if self.frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False        
            
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        
    def forward(self, x):
        feature = self.encoder.forward_encoder(x)
        out = self.classifier(feature)
        return out.squeeze(-1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y.float())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y.float())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return [optimizer], [scheduler]
    
    def on_validation_epoch_end(self):
        loss = self.trainer.callback_metrics.get("val_loss")
        if loss < self.best_val_loss:
            self.best_val_loss = loss
            torch.save(self.classifier.state_dict(), "Best_classifier.pt")
            if not self.frozen:
                torch.save(self.encoder.state_dict(), "Best_encoder.pt")
