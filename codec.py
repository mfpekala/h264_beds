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
