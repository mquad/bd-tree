__author__ = 'massimo'

import numpy as np
import argparse
from os import path, listdir, makedirs
import errno


def mkdir_p(dir_path):
    try:
        makedirs(dir_path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and path.isdir(dir_path):
            pass

def import_dataset(fpath, key_idx=0, delim='\t'):
    dataset = {}
    with open(fpath, 'r') as infile:
        for line in infile:
            key = line.split(delim)[key_idx]
            if key not in dataset:
                dataset[key] = []
            dataset[key].append(line)
    return dataset


def generate_folds(data_size, k=5):
    folds = []
    shuffle_idx = np.arange(data_size)
    np.random.shuffle(shuffle_idx)
    # generate fold boundaries
    fold_size = data_size/k     # ceil integer division
    fold_bounds = [(i*fold_size, (i+1)*fold_size) for i in xrange(0, k-1)]
    fold_bounds += [(fold_bounds[-1][1], data_size)]
    # generate train and test splits
    for low, up in fold_bounds:
        train_indices, test_indices = np.concatenate((shuffle_idx[:low], shuffle_idx[up:])), shuffle_idx[low:up]
        folds.append([train_indices, test_indices])
    return folds


def write_split(splitname, outpath, dataset, train_indices, test_indices, split_perc=.75):
    # write the training profiles
    with open(path.join(outpath, splitname + '.train'), 'w') as tfile:
        for idx in train_indices:
            tfile.write(''.join(dataset[idx]))
    # split and write the answer and evaluation profiles
    with open(path.join(outpath, splitname + '.ans'), 'w') as ansfile,\
        open(path.join(outpath, splitname + '.eval'), 'w') as evalfile:
        for idx in test_indices:
            lines = dataset[idx]
            lines_shuffled = np.arange(len(lines))
            np.random.shuffle(lines_shuffled)
            thresh = int(split_perc*len(lines))
            ansfile.write(''.join(lines[i] for i in lines_shuffled[:thresh]))
            evalfile.write(''.join(lines[i] for i in lines_shuffled[thresh:]))


def cv_split(dataset_fpath, outpath, num_folds=4, split_perc=.75, key_idx=0, delim='\t'):
    dataset = import_dataset(dataset_fpath, key_idx, delim)
    keys = dataset.keys()
    print 'Dataset %s successfully imported.' % dataset_fpath
    # k-fold split of the users in the dataset
    folds = generate_folds(data_size=len(dataset), k=num_folds)
    print '%d folds generated.' % num_folds
    for i, fold in enumerate(folds):
        splitname = 's%d' % i
        splitpath = path.join(outpath, splitname)
        mkdir_p(splitpath)
        print 'Writing fold %d to %s' % (i, splitpath)
        train_indices = [keys[idx] for idx in fold[0]]
        test_indices = [keys[idx] for idx in fold[1]]
        write_split(splitname, splitpath, dataset, train_indices, test_indices, split_perc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Cross-Validation split w.r.t. a given field. Lines in the test split are further splitted for a given splitting percentage.")
    parser.add_argument("--in", dest="dataset_fpath", help="dataset path")
    parser.add_argument("--out", dest="outpath", help="output path")
    parser.add_argument("--folds", dest="num_folds", type=int, default=4, help="number of folds")
    parser.add_argument("--split_perc", dest="split_perc", type=float, default=.75, help="splitting percentage in the test file")
    parser.add_argument("--key", dest="key_idx", type=int, default=0, help="index of key field")
    parser.add_argument("--delim", dest="delim", default='\t', help="delimiter")
    args = vars(parser.parse_args())
    cv_split(**args)
