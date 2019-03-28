"""
Contains methods for import Tricomp Mesh Input files
"""

import re

from .geometry import Region

_MIN_SEPCHAR = r"[\s,\t:\(\)=]+"

def import_min_as_regions(path, scale=1.0, xshift=0.0, yshift=0.0):
    """
    Reads a Tricomp Mesh Input file and returns a list of Region objects describing the individual
    regions defined in the MIN file
    """
    regions = []

    with open(path) as f:
        flines = f.readlines()
    flines = [l.strip().lower() for l in flines]

    while True:
        line = flines.pop(0)

        if line.startswith("endfile") or line is None:
            break # Break on end of file, unnecessary in theory but you never know

        elif line.startswith(r"*"):
            continue # Skip comments

        elif line.startswith("global"): # skip global block
            kw = 0
            while True:
                gline = flines.pop(0)
                if "xmesh" in gline or "ymesh" in gline or "rmesh" in gline or "zmesh" in gline:
                    kw += 1
                if "end" in gline:
                    kw -= 1
                    if kw < 0:
                        break

        elif line.startswith("region"): #read region
            if "fill" in line:
                fill = True
                line = line.replace("fill", "")
            else:
                fill = False

            line = re.split(_MIN_SEPCHAR, line)
            if len(line) == 2:
                name = line[1]
            else:
                name = ""

            segments = []
            while True:
                rline = flines.pop(0)
                if rline.startswith("end"):
                    break
                if rline.startswith(r"*"):
                    continue # Skip comments
                rline = re.split(_MIN_SEPCHAR, rline)
                rline[1:] = [scale*float(val) for val in rline[1:]]

                if rline[0] == "p":
                    s = dict(t="p", x0=rline[1]+xshift, y0=rline[2]+yshift)

                elif rline[0] == "l":
                    s = dict(
                        t="l",
                        x0=rline[1]+xshift, y0=rline[2]+yshift,
                        x1=rline[3]+xshift, y1=rline[4]+yshift
                    )

                elif rline[0] == "a":
                    s = dict(
                        t="a",
                        x0=rline[1]+xshift, y0=rline[2]+yshift,
                        x1=rline[3]+xshift, y1=rline[4]+yshift,
                        xc=rline[5]+xshift, yc=rline[6]+yshift
                    )

                segments.append(s)

            regions.append(Region(name=name, fill=fill, segments=segments))

    return regions
