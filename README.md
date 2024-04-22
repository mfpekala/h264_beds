# Isodiff

For generating I, P, B frames:
https://superuser.com/questions/604858/ffmpeg-extract-intra-frames-i-p-b-frames

https://trac.ffmpeg.org/wiki/Debug/MacroblocksAndMotionVectors

file:///Users/mork/Desktop/APMATH%20220/project/T-REC-H.264-201602-S!!PDF-E.pdf

^ the ITU recommendation for a decoder specification is over 800 pages long.
-> It's kind of infeasabile to implement a custom decoder for a

## New Goal for Project

-x264opts bframes=0:keyint_min=250

https://stackoverflow.com/questions/34647652/h264-encoder-decoder-writing-from-scratch

https://video.stackexchange.com/questions/15800/how-can-i-set-the-first-frame-of-a-video-to-an-i-frame-and-the-rest-to-p-frames

https://last.hit.bme.hu/download/vidtech/k%C3%B6nyvek/Iain%20E.%20Richardson%20-%20H264%20(2nd%20edition).pdf

I want this project to focus mostly on the machine learning aspects, and less on the video-wrangling aspects.

My thoughts are thus:

- Figure out a way to use ffmpeg to get a video which is only I and P frames (sorry B-frames) with I frames happening at regular intervals (say every 5 seconds). This could be reasonably justified by saying that it reduces the memory requirements of both the encoder and decoder, which could be crucial in many surveilance applications.

- Read enough of the h264 + decoder specification to understand what's a reasonable approximation for how real P frames convey information in relation to previous frames.

- Create some synthetic data, where there are iframes, and then "fake" p frames which contain just simplified diff information.

- Perform learning and plotting off of that.

Once it comes to the actual learning, it'll look like this (from OG paper):

- Determine a way to measure similarity between images

- Determine a threshold distance to ignore

- For n images, make an nxn matrices with these pairwise distances

- Apply isomap to this matrix, choosing output dimension

- You then have a spline curve through space

### From the top:

You have some security footage from a single fixed camera, where nothing is expected to get too close to the camera (assumption, probably pretty reasonable).

For some amount of time, you observe image distances. This is just computing the pair-wise distances between all the images I think. (See `Hmm` section below on using little memory and doing fast).

Grr wait I might be fucked, can you add to isomap after initial embedding?

Okay the answer is no. What I really want is this: You do some observation and "learning" of the latent space as above. This produces some way of embedding I-frames into some k-dimensional space. Yay!

THE KEY PART IS that given the embedding of some image X, and an image X' such that X' is close to X (and you only know X' - X, not X' itself), you can get a good approximation of the embedding of X'.

THEN (and only then) you can plot the trajectory in (basically) real-time. This alone I think willl be interesting, just juicing this and seeing what kind of FPS we can get.

THEN as an after thought you can do some kind of learning on this trajectory to see if you can correctly flag "sus" activities (this part probably won't work).

### Hmm, how do you do this without much memorym, and faster?

What if you require there to be exactly 1 I frame at the beginning? Then the data itself is the distance to frame one. Could fill out the first row of the matrix. Then go through and add diff to each frame, O(n) (where n is number of frames.) Hmm is this actually any faster? I guess it's nice because you don't need scratch memory. But I think it still ends up being

n rows

scan n images for each row

to calculate distance requires F operations

so n^2F. Then n times you have update the diff, so another n^2. n^2(1 + F)

Naively would be: pick each combination of two images (n^2). Calculate their diff, (F). Ens up still being n^2F. BUT I guess the memory patter is a lot better in the second case

### Dictionary Learning

Lecture 12, bottom.

## After dictlearn

So take a 256 x 256 image
it gets broken down into 256 16x16 blocks. Each of those blocks is then encoded using the 64-element dictionary. So there's a total of 16384 dimensions. Naively there would be 4 times as many dimensions.

Could maybe get it down to 16 dimensions per block, which would amount to a 16x dimensionality reduction, leaving 256 x 16 dimensions.

We could then go one step further and use an auto-encoder to try and do further dimensionality reduction down to three dimensions. After training this, we would have a simple, smaller neural network. To do that embedding.

The spaces (dimensions):

- Full space (the full image, say 256 x 256 dimensions)
- Block space (once we embed into blocks, say 256 x 16 dimensions)
- Trajectory space (further reduced from block space, say 3 dimensions) [NOTE: Can play around with this. 3 dimensions will be good for visualization (as a potential task of interest), and trajectory analysis, but could also try trajectory analysis on higher dimensional stuff]

SO it's:

- Get P-frame
- Use the macroblock differences to compute the embedding in block space based on the already embedded i-frame
- Use the auto-encoder to compute the embedding in trajectory space.

### Gah fuck it doesn't work for another reason

https://www.kaggle.com/code/datastrophy/vgg16-pytorch-implementation
^VGG implemented from scratch in pytorch, useful for understanding how the layers and non-linearities flow together
