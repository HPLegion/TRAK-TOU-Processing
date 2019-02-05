"""
Contains functions to import a TREK TOU output file
"""

import pandas as pd
from tou_particle import TouParticle
import io
# import warnings
# warnings.warn("Uncertain Units. Current units may depend on the problem symmetry. Role of DUnit unclear.")

TOU_COLWIDTHS = [14, 14, 14, 14] # bytes per data column
TOU_COLNAMES = ["t", "x", "y", "z"]

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
                    #if trajectory == []:
                     #   trajectory = [[-1,-1,-1,-1]]
                    #df_constants = pd.DataFrame(constants,[0])
                    df_trajectory = pd.DataFrame(trajectory, columns=TOU_COLNAMES)
                    #yield (df_trajectory, df_constants)
                    yield (df_trajectory, constants)
                    trajectory = list()

# def read_tou2(filename, zmin=None, zmax=None):
#     """
#     Reads a TREK TOU Trajectory Output File
#     If zmin and / or zmax are set, they limit the imported trajectory range by z value
#     Returns a tuple consisting of two lists which each containing
#     --- a pandas dataframe with trajectory data [t, x, y, z]
#     --- a dictionary with scalar particle properties [id, mass, charge]
#     for each particle in the input file
#     """
#     trajectories = []
#     constants = []

#     with open(filename, mode='r') as f:

#         # Skip header lines WITHOUT first "----" seperator line
#         while True:
#             pos = f.tell()
#             line = f.readline()
#             if "---" in line:
#                 f.seek(pos)
#                 break

#         block = None
#         for line in f:
#             # discard separation line
#             if "---" in line:
#                 if block:
#                     trajectories.append(block)
#                 constants.append(next(f)) # Read next set of constants
#                 # block = []
#                 block = io.StringIO()
#             else:
#                 # block.append([line[1:14], line[15:28], line[29:42], line[43:56]])
#                 block.write(line)
#     trajectories.append(block)

#     constants = list(map(_parse_trajectory_info, constants))
#     trajectories = [_block_to_df(df, zmin, zmax) for df in trajectories]
#     return (trajectories, constants)

# def _block_to_df(block, zmin, zmax):
#     block.seek(0)
#     # df = pd.DataFrame(block, columns=TOU_COLNAMES, dtype="float")
#     df = pd.read_csv(block, names=TOU_COLNAMES, sep="\s+")
#     block.close()
#     df = df[:-1] # drop last row (t=-1 repetition)
#     if zmin:
#         df = df[df["z"] >= zmin]
#     if zmax:
#         df = df[df["z"] <= zmax]
#     return df

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
