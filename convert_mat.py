import scipy.io
import pandas as pd

data = scipy.io.loadmat("vol01.mat")

vol = data["vol"]

data_df.to_csv("output.csv", index=False)
