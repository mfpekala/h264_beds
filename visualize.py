import torch
from torchviz import make_dot
import cv2
import numpy as np
from small_gg import SmallGGAutoencoder
import random


def get_chunks(filepath, chunk_size=16, quantize=64):
    """
    Load a random 16x16 chunk from an image
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = np.round(img * quantize) / quantize

    # Determine padding size
    height, width = img.shape[:2]
    pad_height = (chunk_size - height % chunk_size) % chunk_size
    pad_width = (chunk_size - width % chunk_size) % chunk_size

    # Pad image if necessary
    img = np.pad(img, ((0, pad_height), (0, pad_width)), mode="constant")

    # Reshape image into chunks
    chunks = [
        img[i : i + chunk_size, j : j + chunk_size]
        for i in range(0, img.shape[0], chunk_size)
        for j in range(0, img.shape[1], chunk_size)
    ]

    return [torch.from_numpy(chunk).unsqueeze(0) for chunk in chunks]


autoencoder = SmallGGAutoencoder.load_model("autoencoders/gen0.pt")
image_chunks = get_chunks("frames/i/frame_0.png")

output = autoencoder(image_chunks[0])

dot = make_dot(output, params=dict(autoencoder.named_parameters()))
dot.render("autoencoder_graph")
