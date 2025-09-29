# dev_inverse_net.py
#%% 
import torch
from train import load_dataset

data_dir = "dataset/jb"
mat_files = ("only_k_xx",)

X, Y, grid, head_names = load_dataset(data_dir=data_dir,mat_files=mat_files)

#%%
