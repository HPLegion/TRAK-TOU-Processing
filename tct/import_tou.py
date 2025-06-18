"""
Contains functions to import a TREK TOU output file
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator

import numpy as np

from .beam import Beam
from .particle import Particle


def import_tou_as_particles(
    filename: str,
    zmin: float | None = None,
    zmax: float | None = None,
) -> list[Particle]:
    """
    Reads a TOU file and returns a list of Particle objects holding the
    relevant information in easily accesible form
    If zmin and / or zmax are set, they limit the imported trajectory range by z value
    """
    return list(_stream_particles_from_file(filename, zmin, zmax))


def import_tou_as_beam(
    filename: str,
    zmin: float | None = None,
    zmax: float | None = None,
) -> Beam:
    """
    Reads a TOU file and returns a Beam holding the
    relevant information in easily accesible form
    If zmin and / or zmax are set, they limit the imported trajectory range by z value
    """
    return Beam(import_tou_as_particles(filename, zmin=zmin, zmax=zmax))


def _stream_particles_from_file(
    filename: str,
    zmin: float | None = None,
    zmax: float | None = None,
) -> Generator[Particle, None, None]:
    """
    Stream content of a TOU file, one trajectory after another
    If zmin and / or zmax are set, they limit the imported trajectory range by z value
    """
    if "tou" not in str(filename).lower():
        raise ValueError("Can only import TOU Files")

    tr_buffer: list[list[float]] = []
    header: _Header | None = None

    with open(filename, mode="r") as f:
        # Skip 5 header lines
        for _ in range(5):
            f.readline()

        for line in f:
            # discard separation line
            if "---" in line:
                continue
            # Stop importing when hitting empty line (skip end of file blocks)
            elif line in ("", "\n"):
                break
            # if new particle read scalar information
            elif "Particle" in line:
                header = _Header.parse(line)
            # if trajectory point append to trajectory block
            else:
                if header is None:
                    raise RuntimeError("Attempting to process data without header.")

                data = [float(num) for num in line.split()]

                block_end = data[0] == -1
                und_rng = zmin is not None and zmin > data[3]
                ovr_rng = zmax is not None and zmax < data[3]

                if not (block_end or und_rng or ovr_rng):
                    tr_buffer.append(data)

                # if last line of trajectory block process trajectory information
                if block_end:
                    tr = np.array(tr_buffer)
                    trh = header

                    tr_buffer = []
                    header = None

                    yield Particle(
                        pid=trh.pid,
                        mass=trh.mass,
                        charge=trh.charge,
                        current=trh.current,
                        t=tr[:, 0],
                        x=tr[:, 1],
                        y=tr[:, 2],
                        z=tr[:, 3],
                    )


@dataclass(frozen=True)
class _Header:
    pid: int  #: Particle ID
    mass: float  #: Mass (amu)
    charge: int  #: Charge number
    current: float  #: Current (unknown units)

    @staticmethod
    def parse(text: str) -> _Header:
        """
        reads the info from a trajectory header line
        """
        data = text.split()

        def _get_field_value(key: str) -> float:
            if key in data:
                i = data.index(key)
                return float(data[i + 1])
            else:
                return float("nan")

        try:
            particle_id = int(data[1])
        except ValueError:
            particle_id = -1

        return _Header(
            pid=particle_id,
            mass=_get_field_value("Mass:"),
            charge=_get_field_value("Charge:"),
            current=_get_field_value("Current:"),
        )
