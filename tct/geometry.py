"""
Module for stuff related to Mesh Geometries mostly for plotting
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MPLPath


class Region:
    """
    A simple class containing information about paths defining Regions in a TRAK MESH Input File
    Offers methods to convert to different path descriptions for plotting/drawing
    """

    def __init__(self, name="", fill=False, segments=None):
        self.name = name
        self.fill = fill
        self.segments = segments

    def to_svg_path(self):
        """
        Outputs the region as an svg path string made up of lines and cubic bezier curves
        """
        # path = []
        # path.append(f"M {self.segments[0]['x0']} {self.segments[0]['y0']} ")
        # for seg in self.segments:
        #     if seg["t"] == "p":
        #         continue
        #     elif seg["t"] == "l":
        #         path.append(f"L {seg['x1']} {seg['y1']} ")
        #     elif seg["t"] == "a":
        #         x0, y0, x1, y1 = seg["x0"], seg["y0"], seg["x1"], seg["y1"]
        #         xc, yc = seg["xc"], seg["yc"]
        #         r = np.sqrt((x0-xc)**2 + (y0-yc)**2)
        #         t0 = np.arctan2((y0-yc), (x0-xc))
        #         t1 = np.arctan2((y1-yc), (x1-xc))
        #         sweepflag = 1 if (t0 <= t1) else 0 # arc direction
        #         arclenflag = 1 # TRAK arcs always < 180
        #         rotation = 0
        #         path.append(f"A {r} {r} {rotation} {arclenflag} {sweepflag} {x1} {y1} ")
        # path.append("Z")
        # return "".join(path)

        ## Convert the matplotlib path into a svg path
        mpl_to_svg_codes = {
            MPLPath.MOVETO: "M",
            MPLPath.LINETO: "L",
            MPLPath.CURVE3: "Q",
            MPLPath.CURVE4: "C",
            MPLPath.CLOSEPOLY: "Z",
        }

        path = self.to_mpl_path()
        codes = path.codes
        vertices = path.vertices

        svg = []
        i = 0
        while True:
            c = codes[i]
            n_vert = MPLPath.NUM_VERTICES_FOR_CODE[c]
            verts = vertices[i : i + n_vert]

            svg.append(mpl_to_svg_codes[c])
            if c != MPLPath.CLOSEPOLY:
                for v in verts:
                    svg.append(str(v[0]))
                    svg.append(str(v[1]))
            i += n_vert
            if i >= len(codes):
                break
        return " ".join(svg)

    def to_mpl_path(self):
        """
        Converts the Region into a matplotlib path
        """
        codes = []
        verts = []
        for seg in self.segments:
            codes.append(MPLPath.MOVETO)
            verts.append((seg["x0"], seg["y0"]))
            if seg["t"] == "P":
                continue
            if seg["t"] == "L":
                codes.append(MPLPath.LINETO)
                verts.append((seg["x1"], seg["y1"]))
            elif seg["t"] == "A":
                x0, y0, x1, y1 = seg["x0"], seg["y0"], seg["x1"], seg["y1"]
                xc, yc = seg["xc"], seg["yc"]
                r = np.sqrt((x0 - xc) ** 2 + (y0 - yc) ** 2)
                t0 = np.arctan2((y0 - yc), (x0 - xc))
                t1 = np.arctan2((y1 - yc), (x1 - xc))
                t0 = np.rad2deg(t0)
                t1 = np.rad2deg(t1)
                if t0 < 0:
                    t0 = 360 + t0
                if t1 < 0:
                    t1 = 360 + t1
                if t1 - t0 > 180:
                    t1 -= 360
                if t0 <= t1:
                    arc = MPLPath.arc(t0, t1)
                if t0 > t1:
                    arc = MPLPath.arc(t1, t0)
                arccodes = list(arc.codes[:])
                arcverts = list(arc.vertices[:])
                if t0 > t1:
                    arcverts = arcverts[::-1]
                    arccodes = arccodes[::-1]
                    arccodes[0], arccodes[-1] = arccodes[-1], arccodes[0]
                arcverts = [v * r + np.array([xc, yc]) for v in arcverts]
                codes += arccodes
                verts += arcverts
        if self.fill:
            codes.append(MPLPath.CLOSEPOLY)
            verts.append((0, 0))
            c0, v0 = codes[0], verts[0]
            verts = [v for i, v in enumerate(verts) if codes[i] != MPLPath.MOVETO]
            codes = [c for c in codes if c != MPLPath.MOVETO]
            verts.insert(0, v0)
            codes.insert(0, c0)

        return MPLPath(verts, codes)

    def __str__(self):
        s1 = f"Region: {self.name}, Fill = {self.fill}\n"
        s2 = "\n".join([str(seg) for seg in self.segments])
        return s1 + s2
