import os

import joblib

from brillouin_analysis import pipeline

ROOTDIR = r"M:\REX_NA_GUN"

def failsafe_pipeline(*args, **kwargs):
    try:
        pipeline(*args, **kwargs)
    except Exception as e:
        print(e)


### core
if __name__ == "__main__":
    jobs = []
    for root, dirs, files in os.walk(ROOTDIR):
        for f in files:
            if os.path.splitext(f)[1] == ".tin":
                jobs.append(os.path.join(root, f))


    pool = joblib.Parallel(n_jobs=6, verbose=51)
    delayed_pipeline = joblib.delayed(failsafe_pipeline)
    pool(delayed_pipeline(j) for j in jobs[:])