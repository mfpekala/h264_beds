import numpy as np
import os
from dictlearn import DictionaryLearning
import cv2
import numpy as np
import torch
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from typing import Optional
from sklearn.linear_model import OrthogonalMatchingPursuit
import math
from visualize import show_vec


def filepath2vecs(filepath, chunk_size=16, quantize=64):
    """
    Loads the image and breaks it into (default) 16x16 chunks, returning a tensor with pixel values for each chunk.
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

    # Convert each chunk into a row vector and stack them into a tensor
    chunk_tensors = [torch.from_numpy(chunk.flatten()) for chunk in chunks]
    image_tensor = torch.stack(chunk_tensors)

    return image_tensor


def folder_to_tensor(folder_path):
    """
    Loads all images from a folder as tensors and concatenates them into a tensor
    """
    image_tensors = []

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            vecs = filepath2vecs(filepath)
            image_tensors.append(vecs)

    image_tensor = torch.cat(image_tensors)
    return image_tensor


def initialize_dct_basis(num_basis, basis_length):
    """
    Initialize the dictionary with a DCT basis.
    """
    identity_matrix = np.eye(basis_length)
    basis = []

    for i in range(num_basis):
        dct_basis = dct(identity_matrix[i], norm="ortho")
        basis.append(dct_basis)

    basis_array = np.array(basis).T

    basis_array -= np.min(basis_array)
    basis_array /= np.max(basis_array)

    return basis_array


def learn_folder(path: str, basis_size=16) -> (np.ndarray, list[float]):
    """
    Produces a dictionary of vectors corresponding to 16x16 pixel chunks that
    produce good sparse representations of the images found in at the given path.

    Input: A path to a folder containing i-frames as png images.
    Output: A ndarray of shape (basis_size, 256), where each row is a basis element of the dictionary
    """
    n_components = basis_size
    n_nonzero_coefs = basis_size // 4

    max_iter = 10
    fit_algorithm = "sgk"
    transform_algorithm = "omp"

    data = folder_to_tensor("frames/i")

    dl = DictionaryLearning(
        n_components=n_components,
        max_iter=max_iter,
        fit_algorithm=fit_algorithm,
        transform_algorithm=transform_algorithm,
        n_nonzero_coefs=n_nonzero_coefs,
        code_init=None,
        dict_init=initialize_dct_basis(num_basis=n_components, basis_length=256),
        # dict_init=None,
        verbose=False,
        random_state=None,
        kernel_function=None,
        data_sklearn_compat=True,
    )

    dl.fit(data)

    return (dl.D_.copy(), dl.error_)


class Dictionary:
    def __init__(self, basis_size, D, losses):
        self.basis_size = basis_size
        self.D = D
        self.losses = losses

    @staticmethod
    def from_iframes(folder: str, basis_size=16) -> "Dictionary":
        """
        Create a new dictionary from a given folder of iframes
        """
        (D, losses) = learn_folder(folder, basis_size)
        return Dictionary(basis_size, D, losses)

    def plot_losses(self):
        """
        Plot the losses during the learning process
        """
        plt.plot(range(len(self.losses)), self.losses)
        plt.xlabel("Epoch")
        plt.ylabel("Dictionary Loss")
        plt.title("Loss While Learning Dictionary")
        plt.show()

    def represent_vec(self, vec, n_nonzero_coefs: Optional[int] = None) -> np.ndarray:
        """
        Represent a vector using the learned dictionary. Produces shape (16,)
        """
        n_nonzero_coefs = (
            n_nonzero_coefs if n_nonzero_coefs is not None else self.basis_size
        )
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, tol=None)
        omp.fit(self.D.T, vec)
        coefficients = omp.coef_
        # show_vec(vec)
        # show_vec(self.reconstruct_vec(coefficients))
        return coefficients

    def represent_image(
        self, image, n_nonzero_coefs: Optional[int] = None
    ) -> (np.ndarray, tuple[int, int]):
        """
        Represent an image using the learned dictionary. Produces shape (16 * X,),
        where X is the total number of 16x16 chunks in the image. Also returns the chunk
        width of the image in case it's needed in reconstruction
        """
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        height, width = img.shape[:2]
        height = int(math.ceil(height / 16))
        width = int(math.ceil(width / 16))
        tens = filepath2vecs(image).detach().numpy()
        reps = []
        for row in tens:
            reps.append(self.represent_vec(row, n_nonzero_coefs=self.basis_size // 2))
        return (np.concatenate(reps), (width, height))

    def reconstruct_vec(self, rep) -> np.ndarray:
        """
        Take in a representation of shape (16,) and produce a reconstructed vector of shape (256,)
        """
        return np.dot(self.D.T, rep)

    def reconstruct_image(self, rep, size) -> np.ndarray:
        """
        Take in a concatenated representation of an image shaped (16 * X,), as well as the original
        size _in number of chunks_ (i.e. (chunks_wide, chunks_tall)) and attempt to recreate the
        original image
        """
        chunk_height, chunk_width = 16, 16
        image_height = chunk_height * size[1]
        image_width = chunk_width * size[0]
        chunks = np.split(rep, size[0] * size[1])
        reconstructed_image = np.zeros((image_height, image_width))
        k = 0
        for i in range(0, image_height, chunk_height):
            for j in range(0, image_width, chunk_width):
                reconstructed_image[i : i + chunk_height, j : j + chunk_width] = (
                    self.reconstruct_vec(chunks[k]).reshape(chunk_height, chunk_width)
                )
                k += 1
        return reconstructed_image


D = Dictionary.from_iframes("frames/i", basis_size=32)

(rep, size) = D.represent_image("frames/i/frame_0.png")
D.plot_losses()
img = D.reconstruct_image(rep, size)
plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
plt.axis("off")
plt.show()
