"""
Our various embeddors.
"""

from codec import Decoder, Chunker
from small_gg import SmallGGAutoencoder
import numpy as np
from typing import Optional
import math
import torch
import time
import matplotlib.pyplot as plt
import pickle


def show(chunk):
    """
    Given a 2-dimensional greyscale rep of an image, show it
    """
    plt.imshow(chunk, cmap="gray", vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.show()


def show2(chunk1, chunk2):
    """
    Given two 2-dimensional grayscale representations of images, show them side by side.
    """
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(chunk1.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].axis("off")
    axes[1].imshow(chunk2.squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].axis("off")
    plt.show()


class NaiveEmbeddor:
    """
    A naive embeddor which produces embeddings first by decoding every
    frame into a full image, and then embedding each chunk and concatenating
    """

    def __init__(self, vid_name):
        self.decoder = Decoder(vid_name)
        self.model = SmallGGAutoencoder.load_model("autoencoders/gen3_200.pt")
        self.running_time = 0.0
        self.frame_times = []
        self.beds = []

    def embed(self, verbose=True, limit: Optional[int] = None) -> np.ndarray:
        beds = []
        last_img = None
        for ix, frame in enumerate(self.decoder.generate_frames()):
            start_time = time.time()
            img = frame.to_img(last_img)
            last_img = img.copy()
            this_bed = []
            for chunk in Chunker(img).to_chunks():
                chunk_bed = self.model.embed(chunk)
                this_bed.append(chunk_bed.detach().numpy())
            this_bed = np.array(this_bed).reshape((-1,))
            beds.append(this_bed)

            this_time = time.time() - start_time
            self.running_time += this_time
            self.frame_times.append(self.running_time)
            if verbose:
                print(f"Embedded frame {ix}")
            if limit is not None and ix + 1 >= limit:
                break
        self.beds = np.array(beds)
        return np.array(beds)


class ReconstructEmbeddor:
    """
    An arguably useless embeddor, mainly for testing the diff reconstruction logic
    as it will be needed in the CheapEmbeddor. Uses diffs to reconstruct each chunk
    and then embeds each chunk. Should give the exact same results as NaiveEmbeddor.
    (It will be interesting to see if this is faster than naive)
    """

    def __init__(self, vid_name):
        self.decoder = Decoder(vid_name)
        self.model = SmallGGAutoencoder.load_model("autoencoders/gen3_200.pt")
        self.running_time = 0.0
        self.frame_times = []
        self.beds = []

    def embed(self, verbose=True, limit: Optional[int] = None) -> np.ndarray:
        beds = []
        last_img = np.zeros(self.decoder.img_shape, dtype=np.float32).T
        chunk_size = 16

        def dx_to_top_left(dx):
            _, img_height = self.decoder.img_shape
            chunk_height = int(math.ceil(img_height / chunk_size))
            x = (dx // chunk_height) * chunk_size
            y = (dx % chunk_height) * chunk_size
            return (x, y)

        for ix, frame in enumerate(self.decoder.generate_frames()):
            start_time = time.time()
            if frame.kind == "I":
                img = frame.to_img(None)
                this_bed = []
                for chunk in Chunker(img).to_chunks():
                    _, chunk_bed = self.model.embed_with_prelude(chunk)
                    this_bed.append(chunk_bed.detach().numpy())
                last_img = img.copy()
                this_bed = np.array(this_bed).reshape((-1,))
                beds.append(this_bed)
            else:
                this_bed = []
                new_last_img = np.zeros(self.decoder.img_shape, dtype=np.float32).T
                for dx, diff in enumerate(frame.diffs):
                    x, y = dx_to_top_left(dx)
                    ox, oy = x + diff.offset[0], y + diff.offset[1]
                    ref = last_img[oy : oy + chunk_size, ox : ox + chunk_size]
                    chunk = ref + diff.diff
                    chunk_tens = torch.from_numpy(chunk).unsqueeze(0)
                    _, chunk_bed = self.model.embed_with_prelude(chunk_tens)
                    this_bed.append(chunk_bed.detach().numpy())
                    new_last_img[y : y + chunk_size, x : x + chunk_size] = chunk
                last_img = new_last_img
                this_bed = np.array(this_bed).reshape((-1,))
                beds.append(this_bed)

            this_time = time.time() - start_time
            self.running_time += this_time
            self.frame_times.append(self.running_time)
            if verbose:
                print(f"Embedded frame {ix}")
            if limit is not None and ix + 1 >= limit:
                break
        self.beds = np.array(beds)
        return np.array(beds)


class CheapEmbeddor:
    """
    An embeddor which takes advantage of translational equivariance to compute embeddings
    using less FLOPs. Notably, it does not have to remember entire frames or compute
    convolutions over the entire frame.
    """

    def __init__(self, vid_name):
        self.decoder = Decoder(vid_name)
        self.model = SmallGGAutoencoder.load_model("autoencoders/gen3_200.pt")
        self.running_time = 0.0
        self.frame_times = []
        self.no_diff_times = []
        self.beds = []

    def embed(self, verbose=True, limit: Optional[int] = None) -> np.ndarray:
        beds = []
        last_preludes = np.zeros(
            (self.decoder.img_shape[1], self.decoder.img_shape[0]), dtype=np.float32
        )
        scratch_preludes = np.zeros(
            (self.decoder.img_shape[1], self.decoder.img_shape[0]), dtype=np.float32
        )
        chunk_size = 16
        diff_bonus = 0.0

        def dx_to_top_left(dx):
            _, img_height = self.decoder.img_shape
            chunk_height = int(math.ceil(img_height / chunk_size))
            x = (dx // chunk_height) * chunk_size
            y = (dx % chunk_height) * chunk_size
            return (x, y)

        for ix, frame in enumerate(self.decoder.generate_frames()):
            start_time = time.time()
            if frame.kind == "I":
                img = frame.to_img(None)
                this_bed = []
                for cx, chunk in enumerate(Chunker(img).to_chunks()):
                    x, y = dx_to_top_left(cx)
                    prelude, chunk_bed = self.model.embed_with_prelude(chunk)
                    this_bed.append(chunk_bed.detach().numpy())
                    last_preludes[y : y + chunk_size, x : x + chunk_size] = (
                        prelude.detach().numpy()
                    )
                this_bed = np.array(this_bed).reshape((-1,))
                beds.append(this_bed)
            else:
                this_bed = []
                for dx, diff in enumerate(frame.diffs):
                    x, y = dx_to_top_left(dx)
                    ox, oy = x + diff.offset[0], y + diff.offset[1]
                    prelude = last_preludes[oy : oy + chunk_size, ox : ox + chunk_size]
                    diff_tens = torch.from_numpy(diff.diff).unsqueeze(0)
                    diff_bonus_start = time.time()
                    diff_prelude = self.model.prelude(diff_tens)
                    bias = self.model.prelude.state_dict()["0.bias"][0]
                    diff_prelude -= bias
                    diff_bonus += time.time() - diff_bonus_start
                    prelude_tens = torch.from_numpy(prelude).unsqueeze(0) + diff_prelude
                    chunk_bed = self.model.embed_from_prelude(prelude_tens)
                    this_bed.append(chunk_bed.detach().numpy())
                    scratch_preludes[y : y + chunk_size, x : x + chunk_size] = (
                        prelude_tens.detach().numpy()
                    )
                tmp = last_preludes.copy()
                last_preludes = scratch_preludes.copy()
                scratch_preludes = tmp
                this_bed = np.array(this_bed).reshape((-1,))
                beds.append(this_bed)

            this_time = time.time() - start_time
            self.running_time += this_time
            self.frame_times.append(self.running_time)
            self.no_diff_times.append(self.running_time - diff_bonus)
            if verbose:
                print(f"Embedded frame {ix}")
            if limit is not None and ix + 1 >= limit:
                break
        self.beds = np.array(beds)
        return np.array(beds)


if __name__ == "__main__":
    limit = None
    gen = 3

    naive = NaiveEmbeddor("basic")
    naive_beds = naive.embed(limit=limit)
    with open(f"embeddor_runs/gen{gen}/naive.pkl", "wb") as fout:
        pickle.dump(naive, fout)

    reconstruct = ReconstructEmbeddor("basic")
    reconstruct_beds = reconstruct.embed(limit=limit)
    with open(f"embeddor_runs/gen{gen}/reconstruct.pkl", "wb") as fout:
        pickle.dump(reconstruct, fout)

    cheap = CheapEmbeddor("basic")
    cheap_beds = cheap.embed(limit=limit)
    with open(f"embeddor_runs/gen{gen}/cheap.pkl", "wb") as fout:
        pickle.dump(cheap, fout)


"""
Next steps:

1. Visualize the preludes for make sure it looks okay-ish and the diff is just 
from things adding up, not necessarily from a stupid mistake I can fix
WOAH it was something stupid I did, error is reasonable now
# Not going to do this one 2. Without interfering with timing, come up with a way to store the preludes every 10 frames or something
3. "^" store the final embeddings every 10 frames or something
4. Run on all the data
5. Make pretty plots for the results section
6. add sparsity line and quantify justififcation

In parallel
1. Make diagram for the model architecture
2. Make diagram for the key insight of computation
"""
