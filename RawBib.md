A simple example for dictionary learning in PyTorch that I may build on top of to find the original embedding space

https://carlosoliver.co/2021/11/29/dictionary-learning.html

IDEA: Use 16x16 blocks for the dictionary "atoms" instead of something the size of the entire image

dictlearn package

Paper on dictionary learning
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2944020/
Mostly focusing on how to take an overcomplete dictionary and find a sparse representation

Autoencoders chapter
https://suhangwang.ist.psu.edu/publications/DeepLearningFeatureRepresentation.pdf

Survery on methods for sparse matrix multiplication. suggesting some workloads can see up to 3x performance improvement
https://arxiv.org/pdf/2002.11273
^ the sparsity of diffs is (on average) 12%! I.e. 12% nonzero, wow
