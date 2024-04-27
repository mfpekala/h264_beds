from codec import Decoder, Chunker
from small_gg import SmallGGAutoencoder
import numpy as np
from typing import Optional
import math
import torch
import time


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

    def embed(self, verbose=True, limit: Optional[int] = None) -> np.ndarray:
        beds = []
        last_img = np.zeros(self.decoder.img_shape, dtype=np.float32)
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

    def embed(self, verbose=True, limit: Optional[int] = None) -> np.ndarray:
        beds = []
        last_preludes = None
        chunk_size = 16

        def chunk_ix_to_center(ix):
            _, img_height = self.decoder.img_shape
            chunk_height = int(math.ceil(img_height / chunk_size))
            x = ix // chunk_height + chunk_size // 2
            y = ix % chunk_height + chunk_size // 2
            return (x, y)

        for ix, frame in enumerate(self.decoder.generate_frames()):
            if frame.kind == "I":
                img = frame.to_img(None)
                last_preludes = np.zeros_like(img)
                this_bed = []
                for chunk in Chunker(img).to_chunks():
                    prelude, chunk_bed = self.model.embed_with_prelude(chunk)
                    preludes.append(prelude)
                    this_bed.append(chunk_bed.detach().numpy())
                last_preludes = preludes
                this_bed = np.array(this_bed).reshape((-1,))
                beds.append(this_bed)
            else:
                for dx, diff in enumerate(frame.diffs):
                    pass

            if verbose:
                print(f"Embedded frame {ix}")
            if limit is not None and ix + 1 >= limit:
                break
        return np.array(beds)


naive = NaiveEmbeddor("basic")
naive_beds = naive.embed(limit=20)
reconstruct = ReconstructEmbeddor("basic")
reconstruct_beds = reconstruct.embed(limit=20)

diff = naive_beds - reconstruct_beds
print(np.sum(np.abs(diff)))

print(naive.frame_times)
print(reconstruct.frame_times)
