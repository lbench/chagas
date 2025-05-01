import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from helper_code2 import *
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import pandas as pd
from dataloader import *
from xresnet18 import *
from transformation import *
from SSLModule import *
from nt_xent_loss import *

pretrain_epochs = 100
finetune_epochs = 20
pretrain_lr=5e-3
finetune_lr = 1e-3
out_channel = 256
out_dim = 1
layers = [3, 4,  6, 3]
temperature = 0.2
bs = 128


if ('__name__' == "__main__"):
    # Pretrain
    transformation = Compose([ToTensor(), NormalizeECG(), ResizeECG()])
    augmentation = RandomTransformation()
    
    datamodule = DataModule(path="./code15_output/", transformation=transformation, augmentation=augmentation, batchsize=bs)
    encoder = XResNet18(out_channel=out_channel, layers=layers)
    model = SSLModule(enconder=encoder, loss_fn=nt_xent_loss, lr=pretrain_lr, temperature=temperature, epochs=pretrain_epochs)
    trainer = pl.Trainer(max_epochs=pretrain_epochs, accelerator='gpu')
    
    trainer.fit(model, datamodule=datamodule)
    
    # Fintune
    train_loader = FinetuneDataModule(train_path="./training_data/", val_path='./val_data/', transformation=transformation, batchsize=bs, upsampling=False)
    train_loader.setup()
    pos_weight = train_loader.train_dataset.get_weight()
    
    classifier = ClassifierModule(encoder=encoder, lr=finetune_lr, out_dim = out_dim, epochs=finetune_epochs, pos_weight=pos_weight, frozen=False)
    trainer = pl.Trainer(max_epochs=finetune_epochs, accelerator='gpu')
    
    trainer.fit(classifier, datamodule=train_loader)
    