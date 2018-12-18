import os
import sys
import shutil
import math
import logging
import time

import pandas as pd

import import_tou

# Constants and setup
FPATH = FPREFIX = FPREFIX_RC = ZMIN = ZMAX = None # to be instantiated from config file

TIMESTAMP = time.strftime("%Y-%m-%d-%H-%M-%S")
LOGFNAME = "tou_batch_" + TIMESTAMP + ".log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_file_handler = logging.FileHandler(LOGFNAME)
logger_stream_handler = logging.StreamHandler()
logger.addHandler(logger_file_handler)
logger.addHandler(logger_stream_handler)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvwxyz"
FPOSTFIX = ".tou"

CFG_FNAME = "tou-batch.cfg"
CFG_COMMENT_CHAR = "#"
CFG_ASSIGN_CHAR = "="
CFGSTR_FPATH = "path"
CFGSTR_FPREFIX = "prefix"
CFGSTR_ZMIN = "zmin"
CFGSTR_ZMAX = "zmax"
CFGSTR = [CFGSTR_FPATH, CFGSTR_FPREFIX, CFGSTR_ZMIN, CFGSTR_ZMAX]
CFG_HEADER = [CFG_COMMENT_CHAR + " Config File for TOU-ANGLES-BATCH\n",
              CFG_COMMENT_CHAR + " Lines starting with " + CFG_COMMENT_CHAR + " are comments\n",
              CFG_COMMENT_CHAR + " One assignment (" + CFG_ASSIGN_CHAR + ") per line\n",
              CFG_COMMENT_CHAR + " Empty and undefined vars will be assigned defaults if possible\n"]

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


def parse_filename(fname):
    fname_bckp = fname
    # Strip prefix and postfix
    fname = fname.replace(FPREFIX, "")
    fname = fname.replace(FPOSTFIX, "")
    # remove all remaining alphabetic characters
    for c in ALPHABET:
        fname = fname.replace(c, "")
    # Assemble values and return as list
    values = fname.split("_")
    param = {}
    for i in range(len(values)//2):
        param["p"+str(i)] = float(values[2*i] + "." + values[2*i+1])
    logger.info("%s --- extracted params: %s", fname_bckp, str(param))
    return param

def parse_cfgfile():
    cfg = {}
    with open(CFG_FNAME, "r") as f:
        for ln in f:
            ln = ln.strip()
            if ln.startswith(CFG_COMMENT_CHAR) or ln == "":
                continue
            elif any(cfgstr in ln for cfgstr in CFGSTR):
                try:
                    (key, val) = ln.split(CFG_ASSIGN_CHAR)
                    key = key.strip()
                    val = val.strip()
                    cfg[key] = val
                except Exception as e:
                    logger.exception("Error while parsing line: %s", ln)
                    raise e
            else:
                logger.error("Unknown config parameter in line: %s", ln)
                raise ValueError()
    logger.info("Read config from file: %s", str(cfg))
    return cfg

def create_cfgfile():
    if os.path.exists(CFG_FNAME):
        raise FileExistsError("Cannot create cfg file; a file of the same name already exists.")
    with open(CFG_FNAME, "w") as f:
        f.writelines(CFG_HEADER)
        for cfgstr in CFGSTR:
            f.write(cfgstr + CFG_ASSIGN_CHAR + "\n")
    print("Config file created at:")
    print(os.path.abspath(CFG_FNAME))

def get_files():
    files = os.listdir(FPATH)
    files = [f.lower() for f in files]
    files = [f for f in files if (FPOSTFIX in f and FPREFIX in f)]
    logger.info("Found %d TOU files with matching prefix.", len(files))
    return files

def set_global_params(cfg):
    global FPATH, FPREFIX, FPREFIX_RC, ZMIN, ZMAX
    # path
    FPATH = cfg.get(CFGSTR_FPATH, "")
    if FPATH == "":
        FPATH = "./"
    FPATH = os.path.abspath(FPATH)
    # file prefix
    FPREFIX_RC = cfg.get(CFGSTR_FPREFIX, "")
    FPREFIX = FPREFIX_RC.lower() # lower case file prefix
    # zmin and max
    ZMIN = cfg.get(CFGSTR_ZMIN, "")
    if ZMIN == "":
        ZMIN = None
    else:
        try:
            ZMIN = float(ZMIN)
        except Exception as e:
            logger.exception("Error while trying to cast zmin='%s' to float.", ZMIN)
            raise e
    ZMAX = cfg.get(CFGSTR_ZMAX, "")
    if ZMAX == "":
        ZMAX = None
    else:
        try:
            ZMAX = float(ZMAX)
        except Exception as e:
            logger.exception("Error while trying to cast zmax='%s' to float.", ZMAX)
            raise e
    logger.info("Set FPATH to %s", FPATH)
    logger.info("Set FPREFIX_RC to %s", FPREFIX_RC)
    logger.info("Set FPREFIX to %s", FPREFIX)
    logger.info("Set ZMIN to %s", str(ZMIN))
    logger.info("Set ZMAX to %s", str(ZMAX))

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

        trajs = import_tou.particles_from_tou(os.path.join(FPATH, fn), zmin=ZMIN, zmax=ZMAX)
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
    ### check if config exists
    if not os.path.exists(CFG_FNAME):
        print("Could not find config file. Attempting to create one...")
        create_cfgfile()
        print("Please adjust config file and restart.")
        input("Press enter to exit...")
        logger_file_handler.close()
        os.remove(LOGFNAME)
        sys.exit()

    ### set up config values
    cfg = parse_cfgfile()
    set_global_params(cfg)

    ### get list of relevant files
    fnames = get_files()

    ### run computations
    res_df = process_files(fnames)

    # archive results and inputs
    logger.info("----------------------------------------------------")
    logger.info("Finished analysis, starting to save results")
    SAVENAME = FPREFIX_RC + TIMESTAMP
    RESDIRPATH = os.path.join(FPATH, SAVENAME)
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
    shutil.copy2(CFG_FNAME, CFG_BCKP_PATH)
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
