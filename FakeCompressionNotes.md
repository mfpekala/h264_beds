# Fake Compression Notes

Things that are important about actual H264 that I'll probably want to do as I'm implementing "fake" H264 to make it realistic

## Macroblocks

Macroblocks are 16x16 pixel regions.

First step is motion estimation. Start your search centered at the block you want to predict, and have some kind of max search distance. Search that, and then do motion estimation using that block and whatever is closest from a previous frame.

IGNORE the chrominance luminance stuff I think.

In the real world you can do half and quarter pixel interpolation for macroblock searching which has benefits.

PRESENTATION NOTE: These "residual" images (where they are grayscale showing the energy difference between frames and their representations) are useful for communicating information.

Can also do intra-frame, which is usinga macroblock from this frame that has already been encoded. I think you just add these blocks to be searched in the process of finding stuff for residuals.

In the real world each block would end up being DCT'd, but probably should just avoid doing that because it adds a lot of complexity and is not ML related at all. You can also do some wavelet stuff. Whatever ends up being used, you then have all the goodies you'd expect from DCT or wavelet (run length encoding, ordering, etc.)

path
