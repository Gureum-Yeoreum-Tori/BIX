#%%
import matplotlib.pyplot as plt
from types import SimpleNamespace

fs = SimpleNamespace(
    F=(6.8, 4.2),             # full width
    SC=(6, 3.7),              # single column
    SC_two_third=(4, 3.7),    # single column 2/3
    SC_one_third=(2.1, 3.7),  # single column 1/3
    SC_two_third_s=(4.2, 2.4),# single column 2/3 short
    SC_one_third_s=(2.4, 2.4),# single column 1/3 short
    SC_s=(6, 2.8),            # single column short
    DC=(3.3, 2.06),           # double column
    DC_b=(3.5, 2.18),         # double column big
    DC_tall=(3.3, 3.3)        # double column tall
)



def _default_rcparams():
    plt.rcParams.update({
        "figure.figsize": fs.DC,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.linewidth": 1
    })

def main():
    cm = 1/2.54
    _default_rcparams()

if __name__ == "__main__":
    main()
# %%
