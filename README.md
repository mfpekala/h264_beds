# H264-Beds

A framework for producing online video-embeddings directly on H-264 encodings.

The most important files are likely: `embedding.py`, `small_gg.py`, `small_gg.ipynb`, `downstream.ipynb`, `codec.py` (in that order). The rest are mainly failed experiments, plotting, or data.

## Overview of Files

`autoencoders` - Folder containining saved weights for trained models.

`embeddor_runs` - Saved embeddings from running the embeddors on a single video.

`frames` - Frames as images.

`cheap.ipynb` - Scratch work for using translations to compute embeddings more efficiently. Not necessary to review.

`codec.py` - Our simple implementation of H-264. When run, looks in `frames` and produces simulated H-264 data in the `coded_vids` folder.

`dictionary.py` - Old work experimenting with using dictionary learning instead of autoencoders.

`dictlearn.ipynb` - Old work experiment with using dictionary learning instead of autoencoders.

`downstream.ipynb` - Our simple downstream anomoly detection task.

`embedding.py` - Defines our three embeddors. Notably contains implementations for the naive embeddor and the invariant based embeddor. Probably one of the most relevant/interesting files.

`plot_embeddors.ipyb` - For plotting saved embeddor runs.

`small_gg.ipynb` - Notebook that trains the autoencoder.

`small_gg.py` - File defining the autoencoder architecture.
