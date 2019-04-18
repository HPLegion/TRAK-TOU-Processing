"""
A script that crawls through a directory full of TOU files conforming to a certain prefix
and analyses them, there are certain requirements for the format of the file name in order for them
to be broken into parameters correctly
"""
__version__ = "2019-04-01 10:00"

import os
import sys
import re
import multiprocessing as mp
import shutil
import logging
import time
from collections import OrderedDict
import pandas as pd

# import import_tou
# from tou_particle import TouBeam
from tct import import_tou_as_particles, Beam

### temporary task patch-in
# TODO remove hot patch
from energyprojection import ep_analysis

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
MAX_TASK_LIST = ["max_ang_with_z",
                 "max_kin_energy_trans",
                 "mean_ang_with_z",
                 "mean_kin_energy_trans"]
DF_COLS = [DF_FNAME, DF_NTR, DF_NTR_LOST]
for taskname in MAX_TASK_LIST:
    DF_COLS.append("max_" + taskname + "_z0")
    DF_COLS.append("max_" + taskname)
DF_COLS.append(DF_BEAM_RADIUS_Z0)
DF_COLS.append(DF_BEAM_RADIUS_MEAN)
DF_COLS.append(DF_BEAM_RADIUS_STD)
DF_COLS.append(DF_BEAM_RADIUS_PERIOD)



def parse_filename(fname, fpref, fposf):
    """Parse the parameters from a filename"""
    # Strip prefix and postfix
    fname = fname.replace(fpref, "")
    fname = fname.replace(fposf, "")

    pattern = r"(-?\d+_?\d*)"
    vals = re.findall(pattern, fname)
    param = {}
    for i, v in enumerate(vals):
        param["p"+str(i)] = float(v.replace("_", "."))
    # # remove all remaining alphabetic characters
    # for c in "ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvwxyz":
    #     fname = fname.replace(c, "")
    # # Assemble values and return as list
    # values = fname.split("_")
    # param = {}
    # for i in range(len(values)//2):
    #     param["p"+str(i)] = float(values[2*i] + "." + values[2*i+1])
    return param

def single_file_pipeline(job):
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

    beam = Beam(trajs)
    try:
        (out[DF_BEAM_RADIUS_Z0],
         out[DF_BEAM_RADIUS_MEAN],
         out[DF_BEAM_RADIUS_STD],
         out[DF_BEAM_RADIUS_PERIOD]) = beam.outer_radius_characteristics()
    except IndexError:
        (out[DF_BEAM_RADIUS_Z0],
         out[DF_BEAM_RADIUS_MEAN],
         out[DF_BEAM_RADIUS_STD],
         out[DF_BEAM_RADIUS_PERIOD]) = float("nan"), float("nan"), float("nan"), float("nan")

    # TODO remove hot patch
    try:
        ep = ep_analysis(beam, fname)
        out.update(ep)
    except IndexError:
        pass
    return out

def main():
    """MAIN MONSTROSITY"""
    TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S")
    LOGFNAME = "tou_batch_" + TIMESTAMP + ".log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger_file_handler = logging.FileHandler(LOGFNAME)
    logger_stream_handler = logging.StreamHandler()
    logger.addHandler(logger_file_handler)
    logger.addHandler(logger_stream_handler)

    class Progress:
        """Simple static class for keeping track of the progress"""
        total = 0
        current = 0

    class Config:
        """Static class that deals with everything related to the config and settings"""
        CFGFILE = "tou-batch.cfg" # Not dependent on cfg file
        FPOSTFIX = ".tou" # Not dependent on cfg file
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
        _header = [_comment + " Config File for TOU-BATCH\n",
                   _comment + " Lines starting with " + _comment + " are comments\n",
                   _comment + " One assignment (" + _assignment + ") per line\n",
                   _comment + " Empty and undefined vars will be assigned defaults if possible\n"]

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
                            logger.exception("Error while parsing line: %s", line)
                            raise e
                    else:
                        logger.error("Unknown config parameter in line: %s", line)
                        raise ValueError()
            logger.info("Read config from file: %s", str(cfg))
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
            cls.FPREFIX = cls.FPREFIX_RC.lower() # lower case file prefix

            # zmin and max
            zmin = cfg.get(cls._field_zmin, "")
            if zmin == "":
                cls.ZMIN = None
            else:
                try:
                    cls.ZMIN = float(zmin)
                except Exception as e:
                    logger.exception("Error while trying to cast zmin='%s' to float.", zmin)
                    raise e
            zmax = cfg.get(cls._field_zmax, "")
            if zmax == "":
                cls.ZMAX = None
            else:
                try:
                    cls.ZMAX = float(zmax)
                except Exception as e:
                    logger.exception("Error while trying to cast zmax='%s' to float.", zmax)
                    raise e

            processes = cfg.get(cls._field_processes, "")
            if processes == "":
                cls.PROCESSES = mp.cpu_count()
            else:
                try:
                    cls.PROCESSES = int(processes)
                except Exception as e:
                    logger.exception("Error while trying to cast processes='%s' to int.", processes)
                    raise e

            logger.info("Set Config.FPATH to %s", cls.FPATH)
            logger.info("Set Config.FPREFIX_RC to %s", cls.FPREFIX_RC)
            logger.info("Set Config.FPREFIX to %s", cls.FPREFIX)
            logger.info("Set Config.ZMIN to %s", str(cls.ZMIN))
            logger.info("Set Config.ZMAX to %s", str(cls.ZMAX))
            logger.info("Set Config.PROCESSES to %s", str(cls.PROCESSES))

    def get_files():
        """Acquire all matching TOU filenames"""
        files = os.listdir(Config.FPATH)
        files = [f.lower() for f in files]
        files = [f for f in files if (Config.FPOSTFIX in f and Config.FPREFIX in f)]
        logger.info("Found %d TOU files with matching prefix.", len(files))
        return files

    def merge_results(reslist):
        """Merges and sorts the data"""
        df = pd.DataFrame(reslist)
        avail_params = sorted(list(filter(re.compile(r"^p\d+$").match, list(df))),
                              key=lambda x: int(x[1:]))
        # TODO remove hot patch
        DF_COLS = list(df)
        for p in avail_params:
            DF_COLS.remove(p)

        df = df[avail_params + DF_COLS]
        df = df.sort_values(avail_params)
        return df

    def log_results(result):
        """
        Passes the results of a single file to the logger
        """
        Progress.current = Progress.current + 1
        logger.info("Processed %d / %d", Progress.current, Progress.total)
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Incoming Result ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        for key, val in result.items():
            if isinstance(val, str):
                logger.info("{:<30}: {}".format(key, val))
            else:
                logger.info("{:<30}: {:.3e}".format(key, val))
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  End of Result  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def main_routine():
        """The main routine, orchestrating the rest"""
        logger.info("This is the %s version of this script", __version__)
        ### check if config exists
        if not Config.exists():
            print("Could not find config file. Attempting to create one...")
            Config.create()
            print("Please adjust config file and restart.")
            input("Press enter to exit...")
            logger_file_handler.close()
            os.remove(LOGFNAME)
            sys.exit()

        ### set up config values
        Config.initialise()
        t_start = time.time()

        ### get list of relevant files
        fnames = get_files()

        ### run computations
        defargs = {"zmin":Config.ZMIN, "zmax":Config.ZMAX,
                   "fpref":Config.FPREFIX, "fposf":Config.FPOSTFIX}
        jobs = []
        for fn in fnames:
            j = {"fpath":os.path.join(Config.FPATH, fn)}
            j.update(defargs)
            jobs.append(j)

        Progress.total = len(jobs)

        reslist = []
        with mp.Pool(Config.PROCESSES) as pool:
            for j in jobs:
                res = pool.apply_async(single_file_pipeline, args=(j,),
                                       callback=log_results, error_callback=logger.info)
                reslist.append(res)
            reslist = [res.get() for res in reslist]

        res_df = merge_results(reslist)

        # archive results and inputs
        logger.info("Finished analysis, time elapsed: %d s.", int(time.time()-t_start))
        logger.info("Starting to save results")
        save_name = Config.FPREFIX_RC + TIMESTAMP
        res_path = os.path.join(Config.FPATH, save_name)
        os.mkdir(res_path)
        logger.info("Created output directory at %s", res_path)

        # dump linear result table
        res_dump_path = os.path.join(res_path, save_name + "_resultdump.csv")
        res_df.to_csv(res_dump_path, index=False)
        logger.info("Saved result table dump to %s", res_dump_path)

        # create a default pivot task for each max_task that has been evaluated
        piv_task_list = []
        for tsk in MAX_TASK_LIST:
            piv_task_list.append({"values":"max_" + tsk, "index":"p0", "columns":"p1"})
        piv_task_list.append({"values":DF_BEAM_RADIUS_MEAN, "index":"p0", "columns":"p1"})
        piv_task_list.append({"values":DF_BEAM_RADIUS_STD, "index":"p0", "columns":"p1"})
        piv_task_list.append({"values":DF_BEAM_RADIUS_PERIOD, "index":"p0", "columns":"p1"})

        # and execute them
        for tsk in piv_task_list:
            try:
                piv = res_df.pivot(index=tsk["index"], columns=tsk["columns"], values=tsk["values"])
                name = "_".join([save_name, tsk["index"], tsk["columns"], tsk["values"] + ".csv"])
                piv_path = os.path.join(res_path, name)
                piv.to_csv(piv_path)
                logger.info("Saved pivot table %s to %s", str(tsk), piv_path)
            except Exception as e:
                logger.error("Could not perform pivot task %s", str(tsk))
                logger.error("Intercepted exception, %s", str(e))

        # backup config file
        cfg_bckp_path = os.path.join(res_path, save_name + ".cfg")
        shutil.copy2(Config.CFGFILE, cfg_bckp_path)
        logger.info("Saved config file at %s", cfg_bckp_path)

        # close and move logfile and shutdown
        log_path = os.path.join(res_path, save_name + ".log")
        logger.info("Unlinking log file. Will be moved to %s", log_path)
        logger_file_handler.close()
        shutil.move(LOGFNAME, log_path)
        input("Press enter to exit...")
        sys.exit()

    main_routine()

if __name__ == "__main__":
    main()
