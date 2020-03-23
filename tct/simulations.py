"""
This module defines classes wrapping tricomp simulation input and output files
"""
import os
import re
from types import SimpleNamespace
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

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

def find_file_case_sensitive(filepath):
    d = os.path.dirname(filepath)
    f = os.path.basename(filepath)
    files = os.listdir(d)
    files_lower = [fn.lower() for fn in files]
    fname_lower = f.lower()
    k = files_lower.index(fname_lower)
    return os.path.join(d, files[k])

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
        self.input_file = find_file_case_sensitive(self.input_file)
        self.file_dir = os.path.dirname(self.input_file)
        self.input_file_name = os.path.basename(self.input_file).upper()
        self.output_file_name = self.input_file_name[:-2] + "OU"
        self.output_file = os.path.join(self.file_dir, self.output_file_name)
        self.output_file = find_file_case_sensitive(self.output_file)
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
        self._field = None
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
    @property
    def field(self):
        if self._field is None:
            self._field = EOU(self.output_file)
        return self._field

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
    @property
    def field(self):
        if self._field is None:
            raise NotImplementedError
            # self._field = POU(self.output_file)
        return self._field

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
        self.emission = None
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

            elif command == "EMIT":
                self.emission = SimpleNamespace()
                try:
                    self.emission.T_c = line[7]/8.617333e-5 # k_B in eV/K
                except IndexError:
                    self.emission.T_c = 0

        if efile:
            self.estat = Estat(efile, shift=eshift)
        if bfile:
            self.permag = Permag(bfile, shift=bshift)


    @property
    def beam(self):
        if self._beam is None:
            self._beam = import_tou_as_beam(self.output_file)
        return self._beam

    def plot_trajectories(self, ax=None, p_slice=None, egeo=True, bgeo=True, efield=False, **kwargs):
        """Plots trajectories of particles in the beam together with geometries if provided"""
        if not ax:
            _, ax = plt.subplots(figsize=(12, 9))

        title = kwargs.pop("title", self.output_file_name)
        xlabel = kwargs.pop("xlabel", "$z$ (m)")
        ylabel = kwargs.pop("ylabel", "$r$ (m)")

        if efield:
            if isinstance(efield, dict):
                self.estat.field.plot_potential_contour(ax=ax, **efield)
            else:
                self.estat.field.plot_potential_contour(ax=ax, fill=False)

        if bgeo:
            if isinstance(bgeo, dict):
                self.permag.plot_elements(ax=ax, **bgeo)
            else:
                self.permag.plot_elements(ax=ax, edgecolor="k", facecolor="tab:blue")
        if egeo:
            if isinstance(egeo, dict):
                self.estat.plot_elements(ax=ax, **egeo)
            else:
                self.estat.plot_elements(ax=ax, edgecolor="k", facecolor="tab:gray")

        self.beam.plot_trajectories(ax=ax, p_slice=p_slice, **kwargs)

        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
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


class FieldOutput:
    def __init__(self, output_file):
        self.output_file = os.path.abspath(output_file)
        self.output_file = find_file_case_sensitive(self.output_file)
        self.file_dir = os.path.dirname(self.output_file)
        self.output_file_name = os.path.basename(self.output_file).upper()
        self.run_params = {}
        # self.region_properties = []
        # self.region_names = []
        self.data = None
        self.elements = None
        self._parse_output_file(output_file)
        self._generate_elements_list()
        self.x, self.y = self.data["X"].values, self.data["Y"].values

    def _parse_output_file(self, output_file):
        with open(output_file) as f:
            f.readline() # Skip --- Run parameters --- line
            # Read Run parameters
            while True:
                line = f.readline()
                if line.strip() == "":
                    break
                key, val = line.split(":")
                val = val.strip()
                try:
                    val = int(val)
                except ValueError:
                    val = float(val)
                self.run_params[key.upper()] = val

            f.readline() # Skip --- Nodes --- line
            n_rows = self.run_params["KMAX"] * self.run_params["LMAX"]
            names = [x.upper() for x in f.readline().split()]
            f.readline() # Skip =========== line
            self.data = pd.read_csv(f, nrows=n_rows, names=names, sep=r"\s+")

            #### These lines have no effect currently, pandas exhausts the file so that the cursor
            #### position is not useful, since we don't need this info at the moment I don't care
            # f.readline() # Skip empty line

            # f.readline() # Skip --- Region properties ---
            # for _ in range(self.run_params["NREG"] + 2):
            #     print(f.readline())
            #     self.region_properties.append(f.readline().strip())
            # f.readline() # Skip empty line

            # f.readline() # Skip --- Region names ---:
            # for _ in range(self.run_params["NREG"]):
            #     self.region_names.append(f.readline().strip())

    def _generate_elements_list(self):
        kmax = self.run_params["KMAX"]
        lmax = self.run_params["LMAX"]
        elements = np.zeros((2 * (kmax-1) * lmax, 3))
        krange = np.arange(1, kmax, dtype=int)
        elements = np.zeros((2 * (kmax-1) * lmax, 3))
        for l in range(1, lmax+1):
            ix = (l-1)*kmax + (krange-1)
            if l % 2 == 0:
                data_up = np.column_stack((ix, ix + 1, ix + kmax))
                data_dn = np.column_stack((ix, ix - kmax, ix + 1))
            else:
                data_up = np.column_stack((ix, ix + 1, ix + kmax + 1))
                data_dn = np.column_stack((ix, ix - kmax + 1, ix + 1))
            start = 2*(l-1) * (kmax - 1)
            stop = start+2*(kmax-1)
            elements[start:stop:2] = data_dn
            elements[start+1:stop:2] = data_up
        elements = np.delete(elements, np.s_[-1:-2*(kmax-1):-2], axis=0)
        elements = np.delete(elements, np.s_[:2*(kmax-1):2], axis=0)
        self.elements = elements.astype(int)


class EOU(FieldOutput):
    def __init__(self, output_file):
        super().__init__(output_file)
        self.data = self.data[["X", "Y", "PHI"]]
        self.phi = self.data["PHI"]
        self._interpolator = None

    def plot_potential_contour(self, ax=None, fill=False, **kwargs):
        if not ax:
            fig, ax = plt.subplots(figsize=(12, 9))
        else:
            fig = ax.figure

        kwargs.setdefault("cmap", "plasma")
        kwargs.setdefault("levels", 21)
        kwargs.setdefault("zorder", 1)
        kwargs.setdefault("extend", "both")
        title = kwargs.pop("title", self.output_file_name)

        plotfun = ax.tricontourf if fill else ax.tricontour
        _cont = plotfun(self.x, self.y, self.elements, self.phi, **kwargs)
        cbar = fig.colorbar(_cont, ax=ax)
        cbar.ax.set_ylabel("Potential (V)")

        ax.set(title=title)
        plt.tight_layout()
        return fig

    def interpolate_phi(self, x, y):
        if not self._interpolator:
            self._interpolator = LinearNDInterpolator(self.data[["X", "Y"]], self.data["PHI"])
        return self._interpolator(np.vstack((x, y)).T)
