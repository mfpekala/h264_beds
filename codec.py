import numpy as np
from dataclasses import dataclass
from typing import Optional
import os
import cv2
import math
import matplotlib.pyplot as plt
import torch


def custom_load_img(image_path: str):
    """
    Loads an image with our custom modifications. Namely:
    - Greyscale
    - Pad to be a multiple of 16 on each axis
    - Quanitize to 64 possible values
    """
    chunk_size = 16
    quantize = 64

    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    img = np.round(img * quantize) / quantize

    # Determine padding size
    height, width = img.shape[:2]
    pad_height = (chunk_size - height % chunk_size) % chunk_size
    pad_width = (chunk_size - width % chunk_size) % chunk_size

    # Pad image if necessary
    img = np.pad(img, ((0, pad_height), (0, pad_width)), mode="constant")
    return img


@dataclass
class ChunkDiff:
    """
    P-frames are stored as lists of chunk-diffs. This is ~basically~ h264, with a lot
    of optimizations and customization removed
    """

    # NOTE: Offset is measured from the center of the chunk to the center of the chunk
    offset: Optional[tuple[int, int]]
    # A 16x16 array storing the residual after accounting for the motion given by offset
    diff: Optional[np.ndarray]

    def to_str(self):
        tmp = f"{self.offset[0]}x{self.offset[1]}="
        for row in self.diff:
            for el in row:
                tmp += f"{el},"
        return f"{tmp}\n"

    @staticmethod
    def from_str(diff_str: str) -> "ChunkDiff":
        parts = diff_str.strip().split("=")
        offset_parts = parts[0].split("x")
        offset = (int(offset_parts[0]), int(offset_parts[1]))
        values = [float(num) for num in parts[1].strip().split(",")[:-1]]
        diff = np.zeros((16, 16), dtype=np.float32)
        for ix, val in enumerate(values):
            diff[ix // 16, ix % 16] = val
        return ChunkDiff(offset=offset, diff=diff)


def find_reference_block(
    last_frame: np.ndarray, this_frame: np.ndarray, center: (int, int)
) -> "ChunkDiff":
    """
    Searching in a window around center, find the 16x16 block from `last_frame` that
    matches the 16x16 block at center in `this_frame`.
    """
    chunk_size = 16
    (height, width) = last_frame.shape
    truth = this_frame[
        center[1] - chunk_size // 2 : center[1] + chunk_size // 2,
        center[0] - chunk_size // 2 : center[0] + chunk_size // 2,
    ]
    window_size = 2
    min_residual = float("inf")
    min_diff_chunk = None
    min_offset = (-1, -1)

    for y in range(
        max(chunk_size // 2, center[1] - window_size),
        min(height - chunk_size // 2 - 1, center[1] + window_size),
    ):
        for x in range(
            max(chunk_size // 2, center[0] - window_size),
            min(width - chunk_size // 2 - 1, center[0] + window_size),
        ):
            ref = last_frame[
                y - chunk_size // 2 : y + chunk_size // 2,
                x - chunk_size // 2 : x + chunk_size // 2,
            ]
            diff = truth - ref
            residual = np.sum(np.abs(diff))
            if residual < min_residual:
                min_diff_chunk = diff
                min_residual = residual
                min_offset = (x - center[0], y - center[1])
            last_ref = ref

    # # Sanity check the found chunks
    # do_sanity_check = False
    # if do_sanity_check:
    #     y = center[1] + min_offset[1]
    #     x = center[0] + min_offset[0]
    #     ref = last_frame[
    #         y - chunk_size // 2 : y + chunk_size // 2,
    #         x - chunk_size // 2 : x + chunk_size // 2,
    #     ]
    #     plot_diff = ref + min_diff_chunk - truth
    #     if np.sum(np.abs(plot_diff)) > 0.0:
    #         fig, axs = plt.subplots(1, 5, figsize=(12, 3))  # Adjust figsize as needed
    #         axs[0].imshow(truth, cmap="gray", vmin=0.0, vmax=1.0)
    #         axs[0].set_title("Truth")
    #         axs[1].imshow(ref, cmap="gray", vmin=0.0, vmax=1.0)
    #         axs[1].set_title("Reference")
    #         axs[2].imshow(min_diff_chunk, cmap="gray", vmin=0.0, vmax=1.0)
    #         axs[2].set_title("Min Diff Chunk")
    #         axs[3].imshow(ref + min_diff_chunk, cmap="gray", vmin=0.0, vmax=1.0)
    #         axs[3].set_title("Ref + Min Diff Chunk")
    #         axs[4].imshow(plot_diff, cmap="gray", vmin=0.0, vmax=1.0)
    #         axs[4].set_title("Diff")
    #         plt.tight_layout()
    #         plt.show()
    #         raise ValueError("Did not produce good chunk")

    return ChunkDiff(min_offset, min_diff_chunk)


class Chunker:
    def __init__(self, full_img: np.ndarray, custom_size: Optional[int] = None):
        self.full_img = full_img
        self.custom_size = custom_size

    def to_chunks(self):
        chunk_size = 16 if self.custom_size is None else self.custom_size
        (height, width) = self.full_img.shape
        (chunk_height, chunk_width) = int(math.ceil(height / chunk_size)), int(
            math.ceil(width / chunk_size)
        )
        for cw in range(chunk_width):
            for ch in range(chunk_height):
                x = cw * chunk_size
                y = ch * chunk_size
                np_version = self.full_img[y : y + chunk_size, x : x + chunk_size]
                yield torch.from_numpy(np_version).unsqueeze(0)


@dataclass
class Frame:
    """
    Either an "I"-frame or "P"-frame
    """

    kind: str
    whole_image: np.ndarray
    diffs: list[ChunkDiff]

    @staticmethod
    def produce_i_frame(base: str, full_frame: np.ndarray, ix: int):
        with open(f"{base}/frame_{ix}.txt", "w") as fout:
            fout.write(f"I\n")
            for row in full_frame:
                for el in row:
                    fout.write(f"{el},")
                fout.write("\n")

    @staticmethod
    def produce_p_frame(
        base: str, last_frame: np.ndarray, this_frame: np.ndarray, ix: int
    ):
        chunk_size = 16
        with open(f"{base}/frame_{ix}.txt", "w") as fout:
            fout.write(f"P\n")
            (height, width) = last_frame.shape
            (chunk_height, chunk_width) = int(math.ceil(height / chunk_size)), int(
                math.ceil(width / chunk_size)
            )
            for cw in range(chunk_width):
                tmp = ""
                for ch in range(chunk_height):
                    center = (
                        cw * chunk_size + chunk_size // 2,
                        ch * chunk_size + chunk_size // 2,
                    )
                    chunk_diff = find_reference_block(last_frame, this_frame, center)
                    tmp += chunk_diff.to_str()
                fout.write(tmp)

    @staticmethod
    def from_file(filepath: str):
        with open(filepath) as fin:
            kind = fin.readline().strip()
            if kind == "I":
                rows = []
                for row in fin.readlines():
                    nums = [float(num) for num in row.strip().split(",")[:-1]]
                    rows.append(nums)
                return Frame(kind, np.array(rows, dtype=np.float32), None)
            elif kind == "P":
                diffs = []
                for row in fin.readlines():
                    diffs.append(ChunkDiff.from_str(row))
                return Frame(kind, None, diffs)
            else:
                raise ValueError("Bad type reading")

    def to_img(self, last_frame: Optional[np.ndarray]):
        """
        Returns a numpy array that is the full image. For I-frames this is easy. For P-frames
        this involves some reconstruction
        """
        chunk_size = 16
        if self.kind == "I":
            return self.whole_image
        if last_frame is None:
            raise ValueError("Can't call to_img on P-frame without last frame")
        (height, width) = last_frame.shape
        (chunk_height, chunk_width) = int(math.ceil(height / chunk_size)), int(
            math.ceil(width / chunk_size)
        )
        result = np.zeros(last_frame.shape, dtype=last_frame.dtype)
        for ix, diff in enumerate(self.diffs):
            cw = ix // chunk_height
            ch = ix % chunk_height
            x = cw * chunk_size
            y = ch * chunk_size
            ox = x + diff.offset[0]
            oy = y + diff.offset[1]
            ref = last_frame[oy : oy + chunk_size, ox : ox + chunk_size]
            result[y : y + chunk_size, x : x + chunk_size] = ref + diff.diff
        return result


class Encoder:
    """
    A class to produce compressed frames given the I and P frames of some video

    Frames will be stored in `coded_vids/{vid_name}/frame_X.txt`
    """

    def __init__(self, vid_name: str, I_folder: str, P_folder: str):
        self.vid_name = vid_name
        self.I_folder = I_folder
        self.P_folder = P_folder
        self.coded_vids_folder = f"coded_vids/{vid_name}"
        if os.path.exists(self.coded_vids_folder):
            for file_name in os.listdir(self.coded_vids_folder):
                file_path = os.path.join(self.coded_vids_folder, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)
            os.rmdir(self.coded_vids_folder)
        os.makedirs(self.coded_vids_folder)

    def get_full_i_frame(self, ix: int) -> Optional[np.ndarray]:
        """
        Attempt to load the frame `{I_folder}/frame_{ix}.png` as an ndarray.
        Returns None if no such frame exists
        """
        frame_path = os.path.join(self.I_folder, f"frame_{ix}.png")
        if os.path.exists(frame_path):
            frame = custom_load_img(frame_path)
            return frame
        else:
            return None

    def get_full_p_frame(self, ix: int) -> Optional[np.ndarray]:
        """
        Attempt to load the frame `{P_folder}/frame_{ix}.png` as an ndarray.
        Returns None if no such frame exists
        """
        frame_path = os.path.join(self.P_folder, f"frame_{ix}.png")
        if os.path.exists(frame_path):
            frame = custom_load_img(frame_path)
            return frame
        else:
            return None

    def write_meta_file(self):
        """
        Writes the meta file for the video, which for our simple cases just includes size information
        """
        first_frame = self.get_full_i_frame(0)
        if first_frame is None:
            raise ValueError("Video does not have I-frame at index 0")
        with open(f"{self.coded_vids_folder}/meta.txt", "w") as fout:
            (height, width) = first_frame.shape
            fout.write(f"{width},{height}\n")

    def encode(self, verbose=True):
        self.write_meta_file()
        ix = 0
        last_full_frame: Optional[np.ndarray] = None
        while True:
            if verbose:
                print(f"Encoded frame {ix}")
            i_frame = self.get_full_i_frame(ix)
            p_frame = self.get_full_p_frame(ix)
            if i_frame is not None and p_frame is not None:
                raise ValueError(f"Frame {ix} has both an i-frame and a p-frame")
            if i_frame is None and p_frame is None:
                break
            if i_frame is not None:
                Frame.produce_i_frame(self.coded_vids_folder, i_frame, ix)
                last_full_frame = i_frame
            else:
                if last_full_frame is None:
                    raise ValueError("Video starts with a P-frame. This is invalid.")
                Frame.produce_p_frame(
                    self.coded_vids_folder, last_full_frame, p_frame, ix
                )
                last_full_frame = p_frame
            ix += 1


class Decoder:
    """
    A class to read a folder of encoded .txt files and produce frames
    """

    def __init__(self, vid_name):
        self.coded_vids_folder = f"coded_vids/{vid_name}"
        with open(f"{self.coded_vids_folder}/meta.txt") as fin:
            raw = fin.readline().strip().split(",")
            self.img_shape = (int(raw[0]), int(raw[1]))

    def generate_frames(self):
        ix = 0
        last_img = None
        while True:
            frame_path = os.path.join(self.coded_vids_folder, f"frame_{ix}.txt")
            if not os.path.exists(frame_path):
                return
            frame = Frame.from_file(frame_path)
            yield frame
            ix += 1


# encoder = Encoder("basic", "frames/basic/i", "frames/basic/p")
# encoder.encode()

decoder = Decoder("basic")

# For estimating matrix sparsity in given representation
# total = 0.0
# num = 0

# for frame in decoder.generate_frames():
#     if frame.kind == "I":
#         continue
#     for diff in frame.diffs:
#         num_nonzero = np.count_nonzero(diff.diff)
#         total += num_nonzero / 256
#         num += 1

# print(total / num)

"""
How to store this?


Step back. What do we need?

1. Some way of encoding.
Inputs are a path to the i frame folder and p frame folder. Produces a list of frames with the correct type
Implies we should have a "Frame" class. Type I or P.
- If I type, is just the whole numpy object of the image
- If P type, then just the motion vectors and differences for each macroblock

2. Storage!
Probably best to just create a folder for the video, and store frames as .txt files (basically) labelled
like frame_0.txt, frame_1.txt...
Implies that frame should have a efficient serialization to/from .txt file

3. Naive Embedding Computation
This is basically a way of reading the video where the entire frame is recreated.
Then the image is chunked and the embedding is computed.
You must remember the entire image to successfully calculate the next frame.

4. Cheap Embedding Computation
This is our novel way of computing the embedding.
For I-frames, this is just chunking and then computing the embedding for each chunk.
You must remember the embedding for each chunk BUT NOT THE ACTUAL FRAME
For P-frames, it's a bit foggy rn but it works something like this:
- Iterate over each chunk.
- Find the difference vector. This should touch <= 4 chunks from the previous frame.
- Apply equivariance to estimate (or maybe exactly compute? we'll see) the first layer activation
of this chunk _AS IF_ it were exactly the previous chunk it points to.
- Apply difference + linearity to get the actual activation of this chunk
- Once actual activation is calculated, maxpool + feed-forward to get embedding

5. Compare accuracy
Compare the embeddings produced by naive/cheap method, and observe how much they diverge (if at all)

6. Compare speed
Is the cheap embedding computation faster?

7. Compare memory usage
Does the cheap embedding computation use less memory?

8. Compare other performance considerations
What if we restrict to one thread?


"""
