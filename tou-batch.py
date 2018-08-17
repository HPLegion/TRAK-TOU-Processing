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

TIMESTAMP = time.strftime("%Y%m%d%H%M%S")
LOGFNAME = "tou_batch_" + TIMESTAMP + ".log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger_file_handler = logging.FileHandler(LOGFNAME)
logger_stream_handler = logging.StreamHandler()
logger.addHandler(logger_file_handler)
logger.addHandler(logger_stream_handler)
# logger.info("Logger started.")

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
DF_MAX_ANG_WITH_Z = "max_ang_with_z"
DF_MAX_ANG_WITH_Z_Z0 = "max_ang_with_z_z0"
DF_MAX_ETRANS = "max_e_trans"
DF_MAX_ETRANS_Z0 = "max_e_trans_z0"
DF_COLS = [DF_FNAME, DF_NTR, DF_NTR_LOST, DF_MAX_ANG_WITH_Z_Z0, DF_MAX_ANG_WITH_Z,
           DF_MAX_ETRANS_Z0, DF_MAX_ETRANS]

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
    logger.info("Found " + str(len(files)) + " TOU files with matching prefix.")
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
            logger.exception("Error while trying to cast zmin='" + ZMIN + "' to float.")
            raise e
    ZMAX = cfg.get(CFGSTR_ZMAX, "")
    if ZMAX == "":
        ZMAX = None
    else:
        try:
            ZMAX = float(ZMAX)
        except Exception as e:
            logger.exception("Error while trying to cast zmax='" + ZMAX + "' to float.")
            raise e
    logger.info("Set FPATH to %s", FPATH)
    logger.info("Set FPREFIX_RC to %s", FPREFIX_RC)
    logger.info("Set FPREFIX to %s", FPREFIX)
    logger.info("Set ZMIN to %s", str(ZMIN))
    logger.info("Set ZMAX to %s", str(ZMAX))

def max_ang_with_z_per_file(trajs):
    max_ang = math.nan
    max_ang_z0 = math.nan
    for tr in trajs:
        z0, ang = tr.max_ang_with_z()
        if ang > max_ang or math.isnan(max_ang):
            max_ang = ang
            max_ang_z0 = z0
    return (max_ang_z0, max_ang)

def max_kin_energy_trans_per_file(trajs):
    max_et = math.nan
    max_et_z0 = math.nan
    for tr in trajs:
        z0, et = tr.max_kin_energy_trans()
        if et > max_et or math.isnan(max_et):
            max_et = et
            max_et_z0 = z0
    return (max_et_z0, max_et)


def process_files(fnames):
    res_df = pd.DataFrame()
    max_param = 0
    for i, fn in enumerate(fnames):
        row = {}
        row[DF_FNAME] = fn
        logger.info("Processing file %s/%s --- %s...", str(i+1), str(len(fnames)), fn)

        param = parse_filename(fn)
        if len(param) > max_param:
            max_param = len(param)
        row.update(param)

        trajs = import_tou.particles_from_tou(os.path.join(FPATH, fn), zmin=ZMIN, zmax=ZMAX)
        row[DF_NTR] = len(trajs)
        trajs = [tr for tr in trajs if tr.has_data]
        row[DF_NTR_LOST] = row[DF_NTR] - len(trajs)
        logger.info("%s --- valid trajectories found: %s", fn, str(len(trajs)))

        (max_ang_z_z0, max_ang_z) = max_ang_with_z_per_file(trajs)
        row[DF_MAX_ANG_WITH_Z_Z0] = max_ang_z_z0
        row[DF_MAX_ANG_WITH_Z] = max_ang_z
        logger.info("%s --- max angle with z: %s at z = %s", fn, str(max_ang_z), str(max_ang_z_z0))

        (max_et_z0, max_et) = max_kin_energy_trans_per_file(trajs)
        row[DF_MAX_ETRANS_Z0] = max_et_z0
        row[DF_MAX_ETRANS] = max_et
        logger.info("%s --- max Etrans : %s at z = %s", fn, str(max_et), str(max_et_z0))

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

    ### Define Pivot Saving tasks
    PIV_TASK_LIST = [{"name":DF_MAX_ANG_WITH_Z, "index":"p0", "columns":"p1", "values":DF_MAX_ANG_WITH_Z},
                     {"name":DF_MAX_ETRANS, "index":"p0", "columns":"p1", "values":DF_MAX_ETRANS}]

    # archive results and inputs
    SAVENAME = FPREFIX_RC + TIMESTAMP
    RESDIRPATH = os.path.join(FPATH, SAVENAME)
    os.mkdir(RESDIRPATH)
    logger.info("Created output directory at %s", RESDIRPATH)
    
    for task in PIV_TASK_LIST:
        piv = res_df.pivot(index=task["index"], columns=task["columns"], values=task["values"])
        piv_path = os.path.join(RESDIRPATH, SAVENAME + "_" + task["name"] + ".csv")
        piv.to_csv(piv_path)
        logger.info("Saved pivot table %s to %s", str(task), piv_path)

    CFG_BCKP_PATH = os.path.join(RESDIRPATH, SAVENAME + ".cfg")
    shutil.copy2(CFG_FNAME, CFG_BCKP_PATH)
    logger.info("Saved config file at %s", CFG_BCKP_PATH)

    RESDFFPATH = os.path.join(RESDIRPATH, SAVENAME + "_resultdump.csv")
    res_df.to_csv(RESDFFPATH, index=False)
    logger.info("Saved result table dump to %s", RESDFFPATH)

    # shutdown
    LOGPATH = os.path.join(RESDIRPATH, SAVENAME + ".log")
    logger.info("Unlinking log file. Will be moved to %s", LOGPATH)
    logger_file_handler.close()
    shutil.move(LOGFNAME, LOGPATH)
    input("Press enter to exit...")
    sys.exit()

if __name__ == "__main__":
    main()
