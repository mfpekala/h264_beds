"""
Tools for using the autoencoder that can be imported elsewhere.
"""

import os
import cv2
import numpy as np
import torch
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
from typing import Optional


class SmallGGAutoencoder(nn.Module):
    """
    A partially equi-variant auto-encoder that takes a 16x16 image and reduces
    it to 16 dimensions
    """

    def __init__(self):
        super(SmallGGAutoencoder, self).__init__()
        self.prelude = nn.Sequential(nn.Conv2d(1, 1, kernel_size=5, padding=2))
        self.encoder = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 16 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (16, 16, 16)),
            nn.ConvTranspose2d(16, 1, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.prelude(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def embed(self, x):
        """
        Run just the encoder to get the embedding of the full image
        """
        return self.encoder(self.prelude(x))

    def embed_with_prelude(self, x):
        """
        Run just the encoder, returning the full image embedding and the activations
        to be used for the cheap embeddor
        """
        prelude = self.prelude(x)
        return prelude, self.encoder(prelude)

    def embed_from_prelude(self, prelude):
        """
        Embed from the prelude (conv activation) instead of full layer
        """
        return self.encoder(prelude)

    def save_model(self, file_path):
        """
        Save the model's state dict to a file
        """
        torch.save(self.state_dict(), file_path)

    @classmethod
    def load_model(cls, file_path):
        """
        Load the model from a file and return an instance of the model
        """
        model = cls()
        model.load_state_dict(torch.load(file_path))
        return model
