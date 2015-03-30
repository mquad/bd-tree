__author__ = 'massimo'

import numpy as np
import argparse
from os import path
from sys import argv

def build_dict(fname, key_idx=0, delim='\t'):
    mydict = {}
    with open(fname, 'r') as infile:
        for line in infile:
            key = line.split(delim)[key_idx]
            if key not in mydict:
                mydict[key] = []
            mydict[key].append(line)
    return mydict


def extract_keys(fpath, key_id=0, delim='\t'):
    keys = set()
    with open(fpath, 'r') as infile:
        for line in infile:
            key = line.split(delim)[key_id]
            keys.add(key)
    return keys


def split_by_keys(fpath, outdir, train_keys, test_keys, key_idx=0, delim='\t'):
    fname = path.split(fpath)[-1]
    with open(fpath, 'r') as file:
        train_file = open(path.join(outdir, fname + '.train'), 'w')
        test_file = open(path.join(outdir, fname + '.test'), 'w')
        for line in file:
            key = line.split(delim)[key_idx]
            if key in train_keys:
                train_file.write(line)
            if key in test_keys:
                test_file.write(line)
        train_file.close()
        test_file.close()


def split(train_fpath, test_fpath, outdir, key_idx=0, sampling_perc=1, train_perc=0.75, delim='\t'):
    keys = extract_keys(train_fpath, key_idx, delim)
    # shuffle keys
    keys_shuffled = np.asarray(list(keys))
    np.random.shuffle(keys_shuffled)
    # sample keys
    sample_size = int(len(keys_shuffled)*sampling_perc)
    keys_sampled = keys_shuffled[:sample_size]
    # split the sample into training and test keys
    train_size = int(sample_size*train_perc)
    train_keys = keys_sampled[:train_size]
    test_keys = keys_sampled[train_size:]
    # split the train and test files
    split_by_keys(train_fpath, outdir, train_keys, test_keys, key_idx, delim)
    split_by_keys(test_fpath, outdir, train_keys, test_keys, key_idx, delim)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Samples and splits the training and test sets into disjoint sets for a given key field")
    parser.add_argument("--train", dest="train_fpath", help="training file")
    parser.add_argument("--test", dest="test_fpath", help="test file")
    parser.add_argument("--outdir", dest="outdir", help="output directory")
    parser.add_argument("--key", dest="key_idx", type=int, help="index of key field")
    parser.add_argument("--sampling", dest="sampling_perc", type=float, default=1, help="sampling percentage")
    parser.add_argument("--training", dest="train_perc", type=float, default=0.75, help="training percentage")
    parser.add_argument("--delim", dest="delim", default='\t', help="delimiter")
    args = vars(parser.parse_args())
    print args
    split(**args)