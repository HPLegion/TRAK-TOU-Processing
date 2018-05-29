"""
Contains functions to import a TREK TOU output file
"""

import pandas as pd
from tou_particle import TouParticle

import warnings
warnings.warn("Uncertain Units. Current units may depend on the problem symmetry. Role of DUnit unclear.")

def read_tou_blockwise(filename, zmin=None, zmax=None):
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
        print("Printing Fileheader:")
        for _ in range(5):
            print(f.readline())

        for line in f:
            # discard separation line
            if "---" in line:
                pass
            # if new particle read scalar information
            elif "Particle" in line:
                line_data = line.split()
                particle_id = int(line_data[1])
                if "Current:" in line_data:
                    ind = line_data.index("Current:")
                    current = float(line_data[ind + 1]) # in proton masses or amu ?
                else:
                    current = float("nan")
                if "Mass:" in line_data:
                    ind = line_data.index("Mass:")
                    mass = float(line_data[ind + 1]) # in proton masses or amu ?
                else:
                    mass = float("nan")
                if "Charge:" in line_data:
                    ind = line_data.index("Charge:")
                    charge = float(line_data[ind + 1]) # in proton masses or amu ?
                else:
                    charge = float("nan")
                constants = {"id":particle_id, "mass":mass, "charge":charge, "current":current}
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
                    #if trajectory == []:
                     #   trajectory = [[-1,-1,-1,-1]]
                    #df_constants = pd.DataFrame(constants,[0])
                    df_trajectory =  pd.DataFrame(trajectory, columns=["t", "x", "y", "z"])
                    #yield (df_trajectory, df_constants)
                    yield (df_trajectory, constants)
                    trajectory = list()

def read_tou(filename, zmin=None, zmax=None):
    """
    Reads a TREK TOU Trajectory Output File
    If zmin and / or zmax are set, they limit the imported trajectory range by z value
    Returns a tuple consisting of two lists which each containing
    --- a pandas dataframe with trajectory data [t, x, y, z]
    --- a dictionary with scalar particle properties [id, mass, charge]
    for each particle in the input file
    """
    trajectories = []
    constants = []
    for block in read_tou_blockwise(filename, zmin, zmax):
        trajectories.append(block[0])
        constants.append(block[1])
    return (trajectories, constants)

def particles_from_tou(filename, zmin=None, zmax=None):
    """
    Reads a TOU file and returns a list of TouParticle objects holding the
    relevant information in easily accesible form
    If zmin and / or zmax are set, they limit the imported trajectory range by z value
    """
    trajectories, constants = read_tou(filename, zmin, zmax)
    return [TouParticle(trajectories[k], constants[k]) for k in range(len(trajectories))]
