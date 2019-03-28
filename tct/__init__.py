"""
tct is a collection of tools for working with input and output files of the tricomp
simulation suite
"""
from .import_tou import import_tou_as_beam, import_tou_as_particles
from .import_min import import_min_as_regions
from .beam_particle import Beam, Particle
from .geometry import Region
