"""
tct is a collection of tools for working with input and output files of the tricomp
simulation suite
"""
from __future__ import annotations

from .beam import Beam
from .geometry import Region
from .import_tou import import_tou_as_beam
from .import_tou import import_tou_as_particles
from .particle import Particle
