#%%
# run_manual.py
from train import TrainSettings, run_training

settings = TrainSettings(
    model="deeponet",
    data_dir="dataset/jb",
    mat_files=("20250925_T_110633",),
    head_names=("Kxx", "Kxy", "Kyx", "Kyy", "Cxx", "Cxy", "Cyx", "Cyy",),
    seed=123,
    epochs=1000,
    batch_size=256,
    lr=1e-4,
)

result = run_training(settings)
print(result["metrics"]["val"])
# %%
