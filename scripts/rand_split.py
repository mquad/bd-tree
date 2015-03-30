__author__ = 'massimo'

import argparse
import numpy as np
from os import path


def split(fpath, outdir, perc=.75):
    fname = path.split(fpath)[-1]
    with open(fpath, 'r') as in_file,\
        open(path.join(outdir, fname + '.train'), 'w') as train_file,\
        open(path.join(outdir, fname + '.test'), 'w') as test_file:
        for line in in_file:
            if np.random.rand() <= perc:
                train_file.write(line)
            else:
                test_file.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="fpath", help="input file")
    parser.add_argument("--outdir", help="output dir")
    parser.add_argument("--perc", type=float, default=0.75, help="splitting perc")
    args = vars(parser.parse_args())
    split(**args)