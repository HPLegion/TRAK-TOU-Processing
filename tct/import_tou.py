"""
Contains functions to import a TREK TOU output file
"""

import pandas as pd
from .tou_particle import TouParticle, TouBeam

_TOU_COLNAMES = ["t", "x", "y", "z"]

def _read_tou_blockwise(filename, zmin=None, zmax=None):
    """
    Reads a TREK TOU Trajectory Output File blockwise (one particle at a time)
    If zmin and / or zmax are set, they limit the imported trajectory range by z value
    Returns a tuple consisting of a
    --- pandas dataframe with trajectory data [t, x, y, z]
    --- dictionary with scalar particle properties [id, mass, charge]
    """

    trajectory = list()

    with open(filename, mode='r') as f:
        # Skip 5 header lines
        # print("Printing Fileheader:")
        for _ in range(5):
            # print(f.readline())
            f.readline()

        for line in f:
            # discard separation line
            if "---" in line:
                continue
            # if new particle read scalar information
            elif "Particle" in line:
                constants = _parse_trajectory_info(line)
            # if trajectory point append to trajectory block
            else:
                line_data = line.split()
                line_data = [float(num) for num in line_data]
                do_append = True
                if line_data[0] == -1:
                    do_append = False
                if zmin is not None and zmin > line_data[3]:
                    do_append = False
                if zmax is not None and zmax < line_data[3]:
                    do_append = False
                if do_append:
                    trajectory.append(line_data)

                # if last line of trajectory block process trajectory information
                if line_data[0] == -1:
                    trajectory = trajectory[:-1] # Skip last row (TRAK repeats it with t = -1)
                    df_trajectory = pd.DataFrame(trajectory, columns=_TOU_COLNAMES)
                    yield (df_trajectory, constants)
                    trajectory = list()

def _parse_trajectory_info(line):
    """
    reads the info from a trajectory header line
    """
    line_data = line.split()
    particle_id = int(line_data[1])
    if "Current:" in line_data:
        ind = line_data.index("Current:")
        current = float(line_data[ind + 1])
    else:
        current = float("nan")
    if "Mass:" in line_data:
        ind = line_data.index("Mass:")
        mass = float(line_data[ind + 1]) # in proton masses or amu ?
    else:
        mass = float("nan")
    if "Charge:" in line_data:
        ind = line_data.index("Charge:")
        charge = float(line_data[ind + 1])
    else:
        charge = float("nan")
    return {"id":particle_id, "mass":mass, "charge":charge, "current":current}

def _read_tou(filename, zmin=None, zmax=None):
    """
    Reads a TREK TOU Trajectory Output File
    If zmin and / or zmax are set, they limit the imported trajectory range by z value
    Returns a tuple consisting of two lists which each containing
    --- a pandas dataframe with trajectory data [t, x, y, z]
    --- a dictionary with scalar particle properties [id, mass, charge]
    for each particle in the input file
    """
    if "tou" not in str(filename).lower():
        raise ValueError("Can only import TOU Files")
    trajectories = []
    constants = []
    for block in _read_tou_blockwise(filename, zmin, zmax):
        trajectories.append(block[0])
        constants.append(block[1])
    return (trajectories, constants)

def import_tou_as_particles(filename, zmin=None, zmax=None):
    """
    Reads a TOU file and returns a list of TouParticle objects holding the
    relevant information in easily accesible form
    If zmin and / or zmax are set, they limit the imported trajectory range by z value
    """
    trajectories, constants = _read_tou(filename, zmin, zmax)
    return [TouParticle(trajectories[k], constants[k]) for k in range(len(trajectories))]

def import_tou_as_beam(filename, zmin=None, zmax=None):
    """
    Reads a TOU file and returns a TouBeam holding the
    relevant information in easily accesible form
    If zmin and / or zmax are set, they limit the imported trajectory range by z value
    """
    return TouBeam(import_tou_as_particles(filename, zmin=zmin, zmax=zmax))
