"""
A script that crawls through a directory full of TOU files conforming to a certain prefix
and analyses them, there are certain requirements for the format of the file name in order for them
to be broken into parameters correctly
"""
__version__ = "2018-12-19 14:50"

import os
import sys
import shutil
import logging
import time

import pandas as pd

import import_tou

# Constants and setup

TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S")
LOGFNAME = "tou_batch_" + TIMESTAMP + ".log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_file_handler = logging.FileHandler(LOGFNAME)
logger_stream_handler = logging.StreamHandler()
logger.addHandler(logger_file_handler)
logger.addHandler(logger_stream_handler)

DF_FNAME = "fname"
DF_NTR = "ntr"
DF_NTR_LOST = "ntr_lost"
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


class Config:
    """Static class that deals with everything related to the config and settings"""
    CFGFILE = "tou-batch.cfg"
    FPOSTFIX = ".tou" # Not dependent on cfg file
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
            raise FileExistsError("Cannot create cfg file; a file of the same name already exists.")
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

        logger.info("Set Config.FPATH to %s", cls.FPATH)
        logger.info("Set Config.FPREFIX_RC to %s", cls.FPREFIX_RC)
        logger.info("Set Config.FPREFIX to %s", cls.FPREFIX)
        logger.info("Set Config.ZMIN to %s", str(cls.ZMIN))
        logger.info("Set Config.ZMAX to %s", str(cls.ZMAX))

def parse_filename(fname):
    fname_bckp = fname
    # Strip prefix and postfix
    fname = fname.replace(Config.FPREFIX, "")
    fname = fname.replace(Config.FPOSTFIX, "")
    # remove all remaining alphabetic characters
    for c in "ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvwxyz":
        fname = fname.replace(c, "")
    # Assemble values and return as list
    values = fname.split("_")
    param = {}
    for i in range(len(values)//2):
        param["p"+str(i)] = float(values[2*i] + "." + values[2*i+1])
    logger.info("%s --- extracted params: %s", fname_bckp, str(param))
    return param

def get_files():
    files = os.listdir(Config.FPATH)
    files = [f.lower() for f in files]
    files = [f for f in files if (Config.FPOSTFIX in f and Config.FPREFIX in f)]
    logger.info("Found %d TOU files with matching prefix.", len(files))
    return files

def process_files(fnames):
    res_df = pd.DataFrame()
    max_param = 0
    for i, fn in enumerate(fnames):
        row = {}
        row[DF_FNAME] = fn
        logger.info("Processing file %d / %d --- %s...", i+1, len(fnames), fn)

        param = parse_filename(fn)
        if len(param) > max_param:
            max_param = len(param)
        row.update(param)

        trajs = import_tou.particles_from_tou(os.path.join(Config.FPATH, fn),
                                              zmin=Config.ZMIN, zmax=Config.ZMAX)
        row[DF_NTR] = len(trajs)
        trajs = [tr for tr in trajs if tr.has_data]
        row[DF_NTR_LOST] = row[DF_NTR] - len(trajs)
        logger.info("%s --- valid trajectories found: %d", fn, len(trajs))

        for taskname in MAX_TASK_LIST:
            # the following expression evaluates the task for all trajectories and extracts the
            # largest values and the corresponding value of z
            z0, res = max([getattr(tr, taskname)() for tr in trajs], key=lambda x: x[1])
            row["max_" + taskname + "_z0"] = z0
            row["max_" + taskname] = res
            logger.info("%s --- max_%s: %.3e at z = %.3e", fn, "{:<25}".format(taskname), res, z0)

        res_df = res_df.append(row, ignore_index=True)

    avail_params = ["p"+str(i) for i in range(max_param)]
    res_df = res_df[avail_params + DF_COLS]
    res_df = res_df.sort_values(avail_params)
    return res_df

def main():
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

    ### get list of relevant files
    fnames = get_files()

    ### run computations
    res_df = process_files(fnames)

    # archive results and inputs
    logger.info("----------------------------------------------------")
    logger.info("Finished analysis, starting to save results")
    SAVENAME = Config.FPREFIX_RC + TIMESTAMP
    RESDIRPATH = os.path.join(Config.FPATH, SAVENAME)
    os.mkdir(RESDIRPATH)
    logger.info("Created output directory at %s", RESDIRPATH)

    # dump linear result table
    RESDFFPATH = os.path.join(RESDIRPATH, SAVENAME + "_resultdump.csv")
    res_df.to_csv(RESDFFPATH, index=False)
    logger.info("Saved result table dump to %s", RESDFFPATH)

    piv_task_list = []
    for taskname in MAX_TASK_LIST:
        # create a default pivot task for each max_task that has been evaluated
        piv_task_list.append({"values":"max_" + taskname, "index":"p0", "columns":"p1"})

    # and execute them
    for task in piv_task_list:
        piv = res_df.pivot(index=task["index"], columns=task["columns"], values=task["values"])
        name = SAVENAME + "_" + task["index"]+ "_" + task["columns"]+ "_" + task["values"] + ".csv"
        piv_path = os.path.join(RESDIRPATH, name)
        piv.to_csv(piv_path)
        logger.info("Saved pivot table %s to %s", str(task), piv_path)

    # backup config file
    CFG_BCKP_PATH = os.path.join(RESDIRPATH, SAVENAME + ".cfg")
    shutil.copy2(Config.CFGFILE, CFG_BCKP_PATH)
    logger.info("Saved config file at %s", CFG_BCKP_PATH)

    # close and move logfile and shutdown
    LOGPATH = os.path.join(RESDIRPATH, SAVENAME + ".log")
    logger.info("Unlinking log file. Will be moved to %s", LOGPATH)
    logger_file_handler.close()
    shutil.move(LOGFNAME, LOGPATH)
    input("Press enter to exit...")
    sys.exit()

if __name__ == "__main__":
    main()
