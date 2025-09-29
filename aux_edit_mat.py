#%%
import h5py
import numpy as np

with h5py.File("dataset/jb/only_k_xx/dataset.mat", "r+") as f:
    y = f["y"]
    arr = y[()]            # 메모리로 읽기
    print(arr.shape)
    # arr = np.expand_dims(arr,0)
    # print(arr.shape)
    # del f["y"]
    # f.create_dataset("y",data=arr)


# %%
