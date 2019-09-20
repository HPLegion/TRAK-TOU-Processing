"""
This module defines classes wrapping tricomp simulation input and output files
"""
import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt

from .import_tou import import_tou_as_beam
from .geometry import Region

def preparse_input_file(filepath):
    """
    Preprocessing of tricomp input files, parses into list of lists
    Each list represents a line
    Ints and floats are casted into their data type
    Strings are capitalised
    """
    # read file
    with open(filepath, "r") as f:
        lines = f.readlines()
    # Process lines
    out = []
    for line in lines:
        line = line.strip("\n ,\t:()=")
        if line == "" or line.startswith(r"*"): #Skip empty lines and comments
            continue
        fields = re.split(r"[\s,\t:\(\)=]+", line)
        for i, field in enumerate(fields): # Try to cast to numbers
            try:
                field = int(field)
            except ValueError:
                try:
                    field = float(field)
                except ValueError:
                    field = field.upper()
            fields[i] = field
        out.append(fields)
        if fields[0] == "ENDFILE":
            break
    return out


def input_suffix(name, suffix):
    """
    Process a name given in a input file to resolve the name of
    another input file it is referring to
    """
    name = name.upper()
    if any(name.endswith(x) for x in [".MOU", ".EOU", ".POU", ".TOU"]):
        name = name[:-4]
    if not name.endswith(suffix):
        name += suffix
    return name


class TricompSim:
    """Base class implementation, handles file names and preprocessing"""
    def __init__(self, input_file):
        self.input_file = os.path.abspath(input_file)
        self.file_dir = os.path.dirname(self.input_file)
        self.input_file_name = os.path.basename(self.input_file).upper()
        self.output_file_name = self.input_file_name[:-2] + "OU"
        self.output_file = os.path.join(self.file_dir, self.output_file_name)
        self.raw_input = preparse_input_file(self.input_file)
        self._process_input()

    def _process_input(self):
        raise NotImplementedError


class FieldSim(TricompSim):
    """Base class implementation, for Estat and Permag"""
    def __init__(self, input_file, shift=0.0):
        self._shift = shift
        self.dunit = 1.0
        self.mesh = None
        super().__init__(input_file)

    def _process_input(self):
        for line in self.raw_input:
            command = line[0]

            if command == "MESH":
                mfile = os.path.join(self.file_dir, input_suffix(line[1], ".MIN"))

            elif command == "GEOMETRY":
                self.geometry = "cylindrical" if line[1] == "CYLIN" else "planar"

            elif command == "DUNIT":
                self.dunit = line[1]
        self.mesh = Mesh(mfile, shift=self.shift, scale=1/self.dunit)

    @property
    def shift(self):
        return self._shift
    @shift.setter
    def shift(self, val):
        self._shift = val
        if self.mesh is not None:
            self.mesh.shift = self.shift


class Estat(FieldSim):
    """Estat simulations"""
    def __init__(self, input_file, shift=0.0):
        self.potentials = {}
        self.permittivities = {}
        super().__init__(input_file, shift=shift)

    def _process_input(self):
        super()._process_input()
        for line in self.raw_input:
            command = line[0]

            if command == "POTENTIAL":
                self.potentials[line[1]] = line[2]

            elif command == "EPSI":
                if len(line) == 3:
                    self.permittivities[line[1]] = line[2]
                else:
                    self.permittivities[line[1]] = line[2:]

    def plot_elements(self, ax=None, **kwargs):
        """Plots a list of regions presenting a problem geometry as mpl patches"""
        if not ax:
            _, ax = plt.subplots(figsize=(12, 9))
        for reg_id in self.potentials:
            reg = self.mesh.regions[reg_id]
            patch = mpl.patches.PathPatch(reg.to_mpl_path(), **kwargs)
            ax.add_patch(patch)
        plt.tight_layout()
        return ax.figure


class Permag(FieldSim):
    """Permag simulations"""
    def __init__(self, input_file, shift=0.0):
        self.currents = {}
        self.permeabilities = {}
        super().__init__(input_file, shift=shift)

    def _process_input(self):
        super()._process_input()
        for line in self.raw_input:
            command = line[0]

            if command == "CURRENT":
                self.currents[line[1]] = line[2]

            elif command == "MU":
                if len(line) == 3:
                    self.permeabilities[line[1]] = line[2]
                else:
                    self.permeabilities[line[1]] = line[2:]

    def plot_elements(self, ax=None, **kwargs):
        """Plots a list of regions presenting a problem geometry as mpl patches"""
        if not ax:
            _, ax = plt.subplots(figsize=(12, 9))
        for reg_id, perm in self.permeabilities.items():
            if reg_id in self.currents or not isinstance(perm, float):
                reg = self.mesh.regions[reg_id]
                patch = mpl.patches.PathPatch(reg.to_mpl_path(), **kwargs)
                ax.add_patch(patch)
        plt.tight_layout()
        return ax.figure


class Trak(TricompSim):
    """Trak simulations"""
    def __init__(self, input_file):
        self.estat = None
        self.permag = None
        self._beam = None
        super().__init__(input_file)

    def _process_input(self):
        dunit = 1.0
        efile = None
        bfile = None
        eshift = 0.0
        bshift = 0.0
        for line in self.raw_input:
            command = line[0]

            if command == "DUNIT":
                dunit = line[1]

            elif command == "EFILE":
                efile = os.path.join(self.file_dir, input_suffix(line[1], ".EIN"))

            elif command == "BFILE":
                bfile = os.path.join(self.file_dir, input_suffix(line[1], ".PIN"))

            elif command == "SHIFT":
                if line[1] == "B":
                    bshift = line[2] / dunit
                elif line[1] == "E":
                    eshift = line[2] / dunit

        if efile:
            self.estat = Estat(efile, shift=eshift)
        if bfile:
            self.permag = Permag(bfile, shift=bshift)


    @property
    def beam(self):
        if self._beam is None:
            self._beam = import_tou_as_beam(self.output_file)
        return self._beam

    def plot_trajectories(self, ax=None, egeo=True, bgeo=True, **kwargs):
        """Plots trajectories of particles in the beam together with geometries if provided"""
        if not ax:
            _, ax = plt.subplots(figsize=(12, 9))

        if bgeo:
            self.permag.plot_elements(ax=ax, edgecolor="k", facecolor="cornflowerblue")
        if egeo:
            self.estat.plot_elements(ax=ax, edgecolor="k", facecolor="tab:gray")

        self.beam.plot_trajectories(ax=ax, lw=".75", **kwargs)

        ax.set_xlabel("$z$ (m)")
        ax.set_ylabel("$r$ (m)")
        plt.tight_layout()
        return ax.figure


class Mesh(TricompSim):
    """Meshing tool"""
    def __init__(self, input_file, shift=0.0, scale=1.0):
        self._shift = shift
        self._scale = scale
        self.regions = {}
        self.xmesh = None
        self.ymesh = None
        self.rmesh = None
        self.zmesh = None
        super().__init__(input_file)

    def _process_input(self):
        lines = self.raw_input[:]
        idx = 1
        while True:
            try:
                line = lines.pop(0)
            except IndexError:
                break
            command = line[0]

            if command == "GLOBAL":
                while True:
                    gline = lines.pop(0)
                    gcommand = gline[0]
                    if gcommand in ["XMESH", "YMESH", "RMESH", "ZMESH"]:
                        self.__setattr__(gcommand.lower(), [])
                        while True:
                            mline = lines.pop(0)
                            if mline[0] == "END":
                                break
                            self.__getattribute__(gcommand.lower()).append(mline)
                    if gcommand == "END":
                        break

            elif command == "REGION": #read region
                if "FILL" in line:
                    fill = True
                    line.remove("FILL")
                else:
                    fill = False
                if len(line) == 2:
                    name = str(line[1])
                else:
                    name = ""

                segments = []
                scale = self.scale
                shift = self.shift
                while True:
                    rline = lines.pop(0)
                    rcommand = rline[0]
                    if rcommand == "END":
                        break

                    if rcommand == "P":
                        s = dict(t="P", x0=scale*rline[1]+shift, y0=scale*rline[2])

                    elif rcommand == "L":
                        s = dict(
                            t="L",
                            x0=scale*rline[1]+shift, y0=scale*rline[2],
                            x1=scale*rline[3]+shift, y1=scale*rline[4]
                        )

                    elif rcommand == "A":
                        s = dict(
                            t="A",
                            x0=scale*rline[1]+shift, y0=scale*rline[2],
                            x1=scale*rline[3]+shift, y1=scale*rline[4],
                            xc=scale*rline[5]+shift, yc=scale*rline[6]
                        )
                    segments.append(s)

                self.regions[idx] = Region(name=name, fill=fill, segments=segments)
                idx += 1

    @property
    def shift(self):
        return self._shift
    @shift.setter
    def shift(self, val):
        self._shift = val
        self._process_input()

    @property
    def scale(self):
        return self._scale
    @scale.setter
    def scale(self, val):
        self._scale = val
        self._process_input()
