"""
Contains a function to import a TREK TOU output file
"""

import pandas as pd

def read_tou_blockwise(filename):
    """
    Reads a TREK TOU Trajectory Output File blockwise (one particle at a time)
    Returns a tuple consisting of a
    --- pandas dataframe with trajectory data
    --- dictionary with scalar particle properties
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
                constants = {"particle_id":particle_id, "mass":mass, "charge":charge}
            # if trajectory point append to trajectory block
            else:
                line_data = line.split()
                line_data = [float(num) for num in line_data]
                trajectory.append(line_data)

                # if last line of trajectory block process trajectory information
                if line_data[0] == -1:
                    del trajectory[-1]
                    #df_constants = pd.DataFrame(constants,[0])
                    df_trajectory =  pd.DataFrame(trajectory, columns=["t", "x", "y", "z"])
                    #yield (df_trajectory, df_constants)
                    yield (df_trajectory, constants)
                    trajectory = list()

for particle in read_tou_blockwise("sample.TOU"):
    print(particle[1])
    #print(particle[1].head())

