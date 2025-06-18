"""
A script that crawls through a directory full of TOU files conforming to a certain prefix
and analyses them, there are certain requirements for the format of the file name in order for them
to be broken into parameters correctly
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from collections import OrderedDict
from multiprocessing import RLock
from typing import TYPE_CHECKING
from typing import TypedDict

import pandas as pd

from tct import Beam
from tct import import_tou_as_particles

if TYPE_CHECKING:

    class JobArgs(TypedDict):
        fpath: str  #: absolute(!) filepath
        zmin: float  #: minimum for z range
        zmax: float  #: maximum for z range
        fpref: str  #: filename prefix
        fprof: str  #: filename postfix


__version__ = "2019-04-01 10:00"
_LOG = logging.getLogger(__name__)

### Constants
# Names of dataframe columns
DF_FNAME = "fname"
DF_NTR = "ntr"
DF_NTR_LOST = "ntr_lost"
DF_BEAM_RADIUS_Z0 = "beam_radius_z0"
DF_BEAM_RADIUS_MEAN = "beam_radius_mean"
DF_BEAM_RADIUS_STD = "beam_radius_std"
DF_BEAM_RADIUS_PERIOD = "beam_radius_period"
# the following lists contains the names of the methods of the trajectory objects that we want to
# evaluate and find the maximum of within each file
MAX_TASK_LIST = [
    "max_ang_with_z",
    "max_kin_energy_trans",
    "mean_ang_with_z",
    "mean_kin_energy_trans",
]
DF_COLS = [DF_FNAME, DF_NTR, DF_NTR_LOST]
for taskname in MAX_TASK_LIST:
    DF_COLS.append("max_" + taskname + "_z0")
    DF_COLS.append("max_" + taskname)
DF_COLS.append(DF_BEAM_RADIUS_Z0)
DF_COLS.append(DF_BEAM_RADIUS_MEAN)
DF_COLS.append(DF_BEAM_RADIUS_STD)
DF_COLS.append(DF_BEAM_RADIUS_PERIOD)


def find_file_case_sensitive(filepath: str) -> str:
    d = os.path.dirname(filepath)
    if d == "":
        d = "./"
    f = os.path.basename(filepath)
    files = os.listdir(d)
    files_lower = [fn.lower() for fn in files]
    fname_lower = f.lower()
    k = files_lower.index(fname_lower)
    return os.path.join(d, files[k])


_FNAME_VAL_PATTERN = re.compile(r"(-?\d+_?\d*)")


def parse_filename(fname: str, fpref: str, fposf: str) -> dict[str, float]:
    """Parse the parameters from a filename"""
    # Strip prefix and postfix
    fname = fname.replace(fpref, "")
    fname = fname.replace(fposf, "")

    vals = re.findall(_FNAME_VAL_PATTERN, fname)

    return {"p" + str(i): float(v.replace("_", ".")) for i, v in enumerate(vals)}


def single_file_pipeline(job: JobArgs) -> OrderedDict:
    """
    pipeline for a single file analysis
    job must be a dict containing:
    "fpath" = absolute(!) filepath
    "zmin" = minimum for z range
    "zmax" = maximum for z range
    "fpref" = filename prefix
    "fprof" = filename postfix

    returns dict with results
    """
    out = OrderedDict()
    fname = os.path.split(job["fpath"])[1]
    out[DF_FNAME] = fname
    param = parse_filename(fname, job["fpref"], job["fposf"])
    out.update(param)

    trajs = import_tou_as_particles(job["fpath"], zmin=job["zmin"], zmax=job["zmax"])
    out[DF_NTR] = len(trajs)
    trajs = [tr for tr in trajs if tr.has_data]
    out[DF_NTR_LOST] = out[DF_NTR] - len(trajs)

    for tsk in MAX_TASK_LIST:
        # the following expression evaluates the task for all trajectories and extracts the
        # largest values and the corresponding value of z
        try:
            z0, res = max([getattr(tr, tsk)() for tr in trajs], key=lambda x: x[1])
        except ValueError:
            z0 = res = float("nan")
        out["max_" + tsk + "_z0"] = z0
        out["max_" + tsk] = res

    try:
        beam = Beam(trajs)
        (
            out[DF_BEAM_RADIUS_Z0],
            out[DF_BEAM_RADIUS_MEAN],
            out[DF_BEAM_RADIUS_STD],
            out[DF_BEAM_RADIUS_PERIOD],
        ) = beam.outer_radius_characteristics()
    except IndexError:
        out[DF_BEAM_RADIUS_Z0] = float("nan")
        out[DF_BEAM_RADIUS_MEAN] = float("nan")
        out[DF_BEAM_RADIUS_STD] = float("nan")
        out[DF_BEAM_RADIUS_PERIOD] = float("nan")

    return out


class ResultLogger:
    def __init__(self, total_jobs: int) -> None:
        self._total = total_jobs
        self._processed = 0
        self._lock = RLock()

    def __call__(self, result) -> None:
        """
        Passes the results of a single file to the logger
        """
        with self._lock:
            self._processed = self._processed + 1

            _LOG.info("Processed %d / %d", self._processed, self._total)
            _LOG.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Incoming Result ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            for key, val in result.items():
                if isinstance(val, str):
                    _LOG.info("{:<30}: {}".format(key, val))
                else:
                    _LOG.info("{:<30}: {:.3e}".format(key, val))
            _LOG.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  End of Result  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def configure_logging(logfile: str) -> logging.FileHandler:
    """Sets up logging to STDOUT and a logfile"""
    _LOG.setLevel(logging.INFO)
    file_handler = logging.FileHandler(logfile)
    stream_handler = logging.StreamHandler()
    _LOG.addHandler(file_handler)
    _LOG.addHandler(stream_handler)
    return file_handler


def main():
    """MAIN MONSTROSITY"""
    TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S")
    LOGFNAME = "tou_batch_" + TIMESTAMP + ".log"

    logfile_handler = configure_logging(LOGFNAME)

    class Config:
        """Static class that deals with everything related to the config and settings"""

        CFGFILE = "tou-batch.cfg"  # Not dependent on cfg file
        FPOSTFIX = ".tou"  # Not dependent on cfg file
        FPATH = None
        FPREFIX_RC = None
        FPREFIX = None
        ZMIN = None
        ZMAX = None
        PROCESSES = None
        _comment = "#"
        _assignment = "="
        _field_fpath = "path"
        _field_fprefix = "prefix"
        _field_zmin = "zmin"
        _field_zmax = "zmax"
        _field_processes = "processes"
        _fields = [_field_fpath, _field_fprefix, _field_zmin, _field_zmax, _field_processes]
        _header = [
            _comment + " Config File for TOU-BATCH\n",
            _comment + " Lines starting with " + _comment + " are comments\n",
            _comment + " One assignment (" + _assignment + ") per line\n",
            _comment + " Empty and undefined vars will be assigned defaults if possible\n",
        ]

        @classmethod
        def exists(cls):
            """Check if config file is present"""
            return os.path.exists(cls.CFGFILE)

        @classmethod
        def create(cls):
            """Creates a new config file"""
            if os.path.exists(cls.CFGFILE):
                raise FileExistsError("Cannot create " + cls.CFGFILE + "; file already exists.")
            with open(cls.CFGFILE, "w") as f:
                f.writelines(cls._header)
                for cfgstr in cls._fields:
                    f.write(cfgstr + cls._assignment + "\n")
            print("Config file created at:")
            print(os.path.abspath(cls.CFGFILE))

        @classmethod
        def _readfile(cls):
            """Reads a config file and returns the relevant contents as dict"""
            cfg = {}
            with open(cls.CFGFILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(cls._comment) or line == "":
                        continue
                    elif any(cfgstr in line for cfgstr in cls._fields):
                        try:
                            (key, val) = line.split(cls._assignment)
                            key = key.strip()
                            val = val.strip()
                            cfg[key] = val
                        except Exception as e:
                            _LOG.exception("Error while parsing line: %s", line)
                            raise e
                    else:
                        _LOG.error("Unknown config parameter in line: %s", line)
                        raise ValueError()
            _LOG.info("Read config from file: %s", str(cfg))
            return cfg

        @classmethod
        def initialise(cls):
            """Initialise the fields of config using the data read from the config file"""
            cfg = cls._readfile()
            # path
            fpath = cfg.get(cls._field_fpath, "")
            if fpath == "":
                fpath = "./"
            cls.FPATH = os.path.abspath(fpath)

            # file prefix
            cls.FPREFIX_RC = cfg.get(cls._field_fprefix, "")
            cls.FPREFIX = cls.FPREFIX_RC.lower()  # lower case file prefix

            # zmin and max
            zmin = cfg.get(cls._field_zmin, "")
            if zmin == "":
                cls.ZMIN = None
            else:
                try:
                    cls.ZMIN = float(zmin)
                except Exception as e:
                    _LOG.exception("Error while trying to cast zmin='%s' to float.", zmin)
                    raise e
            zmax = cfg.get(cls._field_zmax, "")
            if zmax == "":
                cls.ZMAX = None
            else:
                try:
                    cls.ZMAX = float(zmax)
                except Exception as e:
                    _LOG.exception("Error while trying to cast zmax='%s' to float.", zmax)
                    raise e

            processes = cfg.get(cls._field_processes, "")
            if processes == "":
                cls.PROCESSES = mp.cpu_count()
            else:
                try:
                    cls.PROCESSES = int(processes)
                except Exception as e:
                    _LOG.exception("Error while trying to cast processes='%s' to int.", processes)
                    raise e

            _LOG.info("Set Config.FPATH to %s", cls.FPATH)
            _LOG.info("Set Config.FPREFIX_RC to %s", cls.FPREFIX_RC)
            _LOG.info("Set Config.FPREFIX to %s", cls.FPREFIX)
            _LOG.info("Set Config.ZMIN to %s", str(cls.ZMIN))
            _LOG.info("Set Config.ZMAX to %s", str(cls.ZMAX))
            _LOG.info("Set Config.PROCESSES to %s", str(cls.PROCESSES))

    def get_files():
        """Acquire all matching TOU filenames"""
        files = os.listdir(Config.FPATH)
        files = [f.lower() for f in files]
        files = [f for f in files if (Config.FPOSTFIX in f and Config.FPREFIX in f)]
        _LOG.info("Found %d TOU files with matching prefix.", len(files))
        return files

    def merge_results(reslist):
        """Merges and sorts the data"""
        df = pd.DataFrame(reslist)
        avail_params = sorted(
            list(filter(re.compile(r"^p\d+$").match, list(df))), key=lambda x: int(x[1:])
        )

        df = df[avail_params + DF_COLS]
        df = df.sort_values(avail_params)
        return df

    def main_routine():
        """The main routine, orchestrating the rest"""
        _LOG.info("This is the %s version of this script", __version__)
        ### check if config exists
        if not Config.exists():
            print("Could not find config file. Attempting to create one...")
            Config.create()
            print("Please adjust config file and restart.")
            input("Press enter to exit...")
            logfile_handler.close()
            os.remove(LOGFNAME)
            sys.exit()

        ### set up config values
        Config.initialise()
        t_start = time.time()

        ### get list of relevant files
        fnames = get_files()
        fnames = [find_file_case_sensitive(filename) for filename in fnames]

        ### run computations
        defargs = {
            "zmin": Config.ZMIN,
            "zmax": Config.ZMAX,
            "fpref": Config.FPREFIX,
            "fposf": Config.FPOSTFIX,
        }
        jobs = []
        for fn in fnames:
            j = {"fpath": os.path.join(Config.FPATH, fn)}
            j.update(defargs)
            jobs.append(j)

        log_result = ResultLogger(total_jobs=len(jobs))

        reslist = []
        with mp.Pool(Config.PROCESSES) as pool:
            for j in jobs:
                res = pool.apply_async(
                    single_file_pipeline,
                    args=(j,),
                    callback=log_result,
                    error_callback=_LOG.error,
                )
                reslist.append(res)
            reslist = [res.get() for res in reslist]

        res_df = merge_results(reslist)

        # archive results and inputs
        _LOG.info("Finished analysis, time elapsed: %d s.", int(time.time() - t_start))
        _LOG.info("Starting to save results")
        save_name = Config.FPREFIX_RC + TIMESTAMP
        res_path = os.path.join(Config.FPATH, save_name)
        os.mkdir(res_path)
        _LOG.info("Created output directory at %s", res_path)

        # dump linear result table
        res_dump_path = os.path.join(res_path, save_name + "_resultdump.csv")
        res_df.to_csv(res_dump_path, index=False)
        _LOG.info("Saved result table dump to %s", res_dump_path)

        # create a default pivot task for each max_task that has been evaluated
        piv_task_list = []
        for tsk in MAX_TASK_LIST:
            piv_task_list.append({"values": "max_" + tsk, "index": "p0", "columns": "p1"})
        piv_task_list.append({"values": DF_BEAM_RADIUS_MEAN, "index": "p0", "columns": "p1"})
        piv_task_list.append({"values": DF_BEAM_RADIUS_STD, "index": "p0", "columns": "p1"})
        piv_task_list.append({"values": DF_BEAM_RADIUS_PERIOD, "index": "p0", "columns": "p1"})

        # and execute them
        for tsk in piv_task_list:
            try:
                piv = res_df.pivot(index=tsk["index"], columns=tsk["columns"], values=tsk["values"])
                name = "_".join([save_name, tsk["index"], tsk["columns"], tsk["values"] + ".csv"])
                piv_path = os.path.join(res_path, name)
                piv.to_csv(piv_path)
                _LOG.info("Saved pivot table %s to %s", str(tsk), piv_path)
            except Exception as e:
                _LOG.error("Could not perform pivot task %s", str(tsk))
                _LOG.error("Intercepted exception, %s", str(e))

        # backup config file
        cfg_bckp_path = os.path.join(res_path, save_name + ".cfg")
        shutil.copy2(Config.CFGFILE, cfg_bckp_path)
        _LOG.info("Saved config file at %s", cfg_bckp_path)

        # close and move logfile and shutdown
        log_path = os.path.join(res_path, save_name + ".log")
        _LOG.info("Unlinking log file. Will be moved to %s", log_path)
        logfile_handler.close()
        shutil.move(LOGFNAME, log_path)
        input("Press enter to exit...")
        sys.exit()

    main_routine()


if __name__ == "__main__":
    main()
