#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os

import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from helper_code import *
from dataloader import *
from resnet import PreActResNet1d

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

SEQ_LENGTH = 4096
N_CLASSES = 2 - 1
dropout_rate = 0.5
kernel_size = 17
filter_size = [64, 128, 196, 256, 320, 512, 800, 1024]
net_length = [4096, 2048, 1024, 512, 256, 128, 64, 16]
EPOCHS = 15




# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')


    separate_data(data_folder)
    other_data, sami_data = get_datasets(data_folder)

    other_train = DataLoader(other_data, batch_size=16, shuffle=True, drop_last=True)
    sami_train = DataLoader(sami_data, batch_size=16, drop_last=True,
    sampler=torch.utils.data.RandomSampler(
        sami_data, replacement=True, num_samples=len(other_train)*16
    ))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    net = ResNet1d(input_dim=(12, SEQ_LENGTH),
                   blocks_dim=list(zip(filter_size, net_length)),
                   n_classes=N_CLASSES,
                   kernel_size=kernel_size,
                   dropout_rate=dropout_rate).to(device)

    net.train()

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.95)

    # Train the models.
    if verbose:
        print("Using device:", device)
        print('Training the model on the data...')


    losses = []

    for epoch in range(EPOCHS):

        running_loss = 0.0
        i = 0
        for (code, code_labels), (sami, sami_labels) in zip(other_train, sami_train):
            inputs, labels = torch.cat([code, sami], dim=0), torch.cat([code_labels, sami_labels], dim=0)
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float)
            inputs = torch.permute(inputs, (0, 2, 1))

            optimizer.zero_grad()
            outputs = torch.squeeze(net(inputs))
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            i += 1

        scheduler.step()
        losses.append(running_loss)

        if verbose:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / i + 1:.3f}')

    if verbose:
        print("Done Training...")
    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, net)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = PreActResNet1d(input_dim=(12, SEQ_LENGTH),
                         blocks_dim=list(zip(filter_size, net_length)),
                         n_classes=N_CLASSES,
                         kernel_size=kernel_size,
                         dropout=dropout_rate).to(device)


    model.load_state_dict(torch.load(os.path.join(model_folder, 'model3.pth'), weights_only=True))

    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    signal, fields = load_signals(record)
    signal = torch.from_numpy(signal).to(device, dtype=torch.float32).T

    padding = (0, 4096 - signal.shape[1], 0, 0)
    signal = F.pad(signal, padding, "constant", 0)

    signal = torch.unsqueeze(signal, 0)

    # Load the model.


    # Get the model outputs.
    model.eval()
    probability_output = model(signal)
    probability_output = F.sigmoid(probability_output)
    binary_output = torch.round(probability_output)

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)

    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    return np.asarray(features, dtype=np.float32)

# Save your trained model.
def save_model(model_folder, model):
    torch.save(model.state_dict(), os.path.join(model_folder, 'model2.pth'))