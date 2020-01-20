"""Script for quickly generating some plots of TRAK simulation results"""
import os
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
# import numpy as np
import scipy.interpolate

from tct.simulations import Trak

CWD = r"C:\TRAKTEMP\REX_NA_DRAWING_ringwidth_+1mm_1300K"
XLIM = [-.057, .1]
YLIM = [0, .015]
XLIM_ZOOM = [-.057, 0.0]
YLIM_ZOOM = [0.0, .006]

os.chdir(CWD)

magdf = pd.read_csv("./BREX_RING_AXIS_SCAN.TXT", comment="#", sep=r"\s+")
magdf.Z = magdf.Z/1000
field_interp = scipy.interpolate.interp1d(magdf.Z, magdf.Bz)

files = os.listdir(CWD)
files = [f for f in files if f.endswith(".tin")]
files = [f for f in files if "repot" not in f]


for f in tqdm.tqdm(files):
    # print(f)
    fnamestub = f[:-4]
    trak = Trak(f)
    beam = trak.beam

    # ################ Trajectory plot
    fig = trak.plot_trajectories()
    fig.gca().set(title=f"{fnamestub} - {beam.current} A", xlim=XLIM, ylim=YLIM)
    plt.tight_layout()
    fig.savefig(fnamestub + "_2.png")
    plt.close(fig)

    # ################ Transverse plot
    fig, ax = plt.subplots(figsize=(12, 9))
    for p in beam.particles[::5]:
        ax.plot(p.x, p.y)
    ax.set(xlabel="x (m)", ylabel="y (m)", title=f"{fnamestub} - {beam.current} A")
    fig.savefig(fnamestub + "_transverse_2.png")
    plt.close(fig)

    ################ Trajectory plot with fields
    fig = trak.plot_trajectories(efield={"fill":True})
    ax = fig.gca()
    ax.set(title=f"{fnamestub} - {beam.current} A", xlim=XLIM, ylim=YLIM)
    plt.tight_layout()

    ax2 = ax.twinx()
    ax2.plot(magdf.Z + trak.permag.shift, magdf.Bz, "tab:blue", label="B_z(r=0)")
    ax2.set_ylim((0, 1.0))
    ax2.set_ylabel("$B_z$ (T)", color="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:blue")
    plt.tight_layout()

    fig.savefig(fnamestub + "_fields_2.png")

    ax.set(xlim=XLIM_ZOOM, ylim=YLIM_ZOOM)
    fig.savefig(fnamestub + "_fields_zoom_2.png")
    plt.close(fig)
