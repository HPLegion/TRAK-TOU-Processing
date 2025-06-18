"""
I no longer remember what this does or was used for
But at some point the ep_analysis function was called from tou_batch,
this was probably an experiment that I did at some point

/HP
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.constants import electron_mass as M_E
from scipy.constants import elementary_charge as Q_E
from scipy.constants import epsilon_0 as EPS_0
from scipy.constants import pi as PI

from tct import import_tou_as_beam

### Constants
CWD = r"C:\TRAKTEMP\REXGUN_THERMAL\long_anode_xs_5mm"
Z0 = 0.045
BMAX = 2.0
R_D = 0.005

### Load mag field line scan
magdf = pd.read_csv(
    os.path.join(CWD, "./FBREX-36-28_6+0_7MM_500GS_AXIS_SCAN.TXT"),
    comment="#",
    sep=r"\s+"
)
magdf.Z = magdf.Z/1000
field_interp = scipy.interpolate.interp1d(magdf.Z, magdf.Bz)


def phi_sc(r_e, r_d, cur, phi_a):
    """Returns a space charge correction for given parameters"""
    c = (2 * np.log(r_e/r_d)-1) * 1/(4*PI*EPS_0) * np.sqrt(M_E/(2*Q_E)) * cur
    coeff = np.array([1, phi_a, 0, 0-c**2])
    return np.roots(coeff)[1]

# files = os.listdir(CWD)
# files = [f for f in files if f.endswith("TOU")]


# for f in files:

def ep_analysis(beam, fname):
    res = OrderedDict()
    res["EP_file"] = fname

    fdata = fname.split("_")
    ektheo = 10000 + float(fdata[-3][:-1])
    res["EP_ektheo"] = ektheo
    # cur = float(fdata[-2] + "." + fdata[-1][:-5])

    b = beam
    par = b.particles

    cur = b.current
    res["EP_cur"] = cur

    maxi = b.outer_radius_closest_max(Z0)
    maxi = dict(z=maxi[0], r=maxi[1], keypref="EP_MAX_")

    mini = b.outer_radius_closest_min(Z0)
    mini = dict(z=mini[0], r=mini[1], keypref="EP_MIN_")

    zavg = (maxi["z"] + mini["z"]) / 2
    avg = dict(z=zavg, r=np.max([np.interp(zavg, p.z, p.r) for p in par]), keypref="EP_AVG_")

    for case in [mini, avg, maxi]:
        z = case["z"]
        bs = case["r"]
        kp = case["keypref"]
        res[kp + "z"] = z
        res[kp + "bs"] = bs

        res[kp + "curden"] = cur / PI / bs / bs

        e = [np.interp(z, p.z, p.kin_energy) for p in par]
        res[kp + "e_min"] = np.min(e)
        res[kp + "e_max"] = np.max(e)

        etrans = [np.interp(z, p.z, p.kin_energy_trans) for p in par]
        res[kp + "etrans_min"] = np.min(etrans)
        res[kp + "etrans_max"] = np.max(etrans)


        sc = phi_sc(bs, .005, cur, ektheo)
        res[kp + "sc"] = sc

        B = field_interp(z)
        res[kp + "B"] = B

        bs_prj = bs * np.sqrt(B/BMAX)
        res[kp + "bs_prj"] = bs_prj

        res[kp + "curden_prj"] = cur / PI / bs_prj / bs_prj

        sc_prj = phi_sc(bs_prj, .005, cur, ektheo)
        res[kp + "sc_prj"] = sc_prj

        elong_prj = np.array(e) - np.array(etrans) * BMAX / B - sc + sc_prj
        res[kp + "elong_prj_min"] = np.min(elong_prj)
        res[kp + "elong_prj_max"] = np.max(elong_prj)

        if np.min(elong_prj) < 1/3 * ektheo:
            res[kp + "problem"] = True
        else:
            res[kp + "problem"] = False

    return res