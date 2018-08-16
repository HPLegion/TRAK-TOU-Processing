import os

# Input Params
FPATH = os.path.abspath("C:/SAMPLEDATAFROMSASHA")
FPREFIX_RC = "NA53FC1F_" # real case file prefix
FPREFIX = FPREFIX_RC.lower() # lower case file prefix
ZMIN = 0.0
ZMAX = 1.0

# Constants
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVXYZabcdefghijklmnopqrstuvwxyz"
FPOSTFIX = ".tou"

CFG_FNAME = "./tou-angles-batch.cfg"
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

def parse_filename(fname):
    # Strip prefix and postfix
    fname = fname.replace(FPREFIX, "")
    fname = fname.replace(FPOSTFIX, "")
    # remove all remaining alphabetic characters
    for c in ALPHABET:
        fname = fname.replace(c, "")
    # Assemble values and return as list
    values = fname.split("_")
    param = []
    for i in range(len(values)//2):
        param.append(float(values[2*i] + "." + values[2*i+1]))
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
                    print("Encountered error while parsing line: ")
                    print(ln)
                    raise e
            else:
                raise ValueError("Unknown config parameter in line: " + ln)
    print("Read the following config from file:")
    for key, val in cfg.items():
        print(key, "=" ,val)
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
    print("Found " + str(len(files)) + " TOU files with matching prefix.")
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
            print("Error while trying to cast zmin='" + ZMIN + "' to float.")
            raise e
    ZMAX = cfg.get(CFGSTR_ZMAX, "")
    if ZMAX == "":
        ZMAX = None
    else:
        try:
            ZMAX = float(ZMAX)
        except Exception as e:
            print("Error while trying to cast zmax='" + ZMAX + "' to float.")
        raise e
    print("Set FPATH to ", FPATH)
    print("Set FPREFIX_RC to ", FPREFIX_RC)
    print("Set FPREFIX to ", FPREFIX)
    print("Set ZMIN to ", ZMIN)
    print("Set ZMAX to ", ZMAX)

def main():
    ### check if config exists
    if not os.path.exists(CFG_FNAME):
        print("Could not find config file. Attempting to create one...")
        create_cfgfile()
        print("Please adjust config file and restart.")
        input("Press enter to exit...")
        exit()

    ### set up config values
    cfg = parse_cfgfile()
    set_global_params(cfg)

    ### get list of relevant files
    fnames = get_files()

    for fn in fnames:
        pass
    
if __name__ == "__main__":
    main()
