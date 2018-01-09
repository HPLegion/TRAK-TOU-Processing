"""
Contains functions to import a TREK TOU output file
"""

import pandas as pd
from simple_tou_particle import SimpleTouParticle

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
        for _ in range(5):
            f.readline()

        for line in f:
            # discard separation line
            if "---" in line:
                pass
            # if new particle read scalar information
            elif "Particle" in line:
                line_data = line.split()
                particle_id = int(line_data[1])
                mass = float(line_data[3]) # in proton masses or amu ?
                charge = int(float(line_data[5])) # in elementary charges ?
                constants = {"id":particle_id, "mass":mass, "charge":charge}
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
    Reads a TOU file and returns a list of SimpleTouParticle objects holding the
    relevant information in easily accesible form
    If zmin and / or zmax are set, they limit the imported trajectory range by z value
    """
    trajectories, constants = read_tou(filename, zmin, zmax)
    return [SimpleTouParticle(trajectories[k], constants[k]) for k in range(len(trajectories))]


#for particle in read_tou_blockwise("sample.TOU", zmin=3.776107E-01, zmax=3.797542E-01):
for particle in read_tou_blockwise("sample_big.TOU"):
#for particle in read_tou_blockwise("sample.TOU", zmin=3.776107E-01, zmax=-1):
    print(particle[1])
    print(particle[0].head())
    a = SimpleTouParticle(particle[0], particle[1])
print("wait")