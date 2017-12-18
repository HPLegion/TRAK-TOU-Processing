"""
Contains a function to import a TREK TOU output file
"""

import pandas as pd

def import_tou(filename):
    """
    Imports a TREK TOU Trajectory Output File 
    Returns a tuple consisting of a
    --- pandas panel with one dataframe per particle
    --- dataframe with scalar particle properties
    """
