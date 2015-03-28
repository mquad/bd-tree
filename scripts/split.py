__author__ = 'massimo'

import numpy as np
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


def split(train_fname, test_fname, perc=0.75, delim='\t'):
    train_dict = build_dict(train_fname, delim=delim)
    test_dict = build_dict(test_fname, delim=delim)

    train_keys_shuffled = np.asarray(train_dict.keys())
    np.random.shuffle(train_keys_shuffled)
    train_size = int(len(train_keys_shuffled)*perc)
    train_keys = train_keys_shuffled[:train_size]
    test_keys = train_keys_shuffled[train_size:]

    train_train_file = open(train_fname + '.train', 'w')
    train_test_file = open(train_fname + '.test', 'w')
    test_train_file = open(test_fname + '.train', 'w')
    test_test_file = open(test_fname + '.test', 'w')

    for key in train_keys:
        train_train_file.write(''.join(train_dict[key]))
        if key in test_dict:
            test_train_file.write(''.join(test_dict[key]))

    for key in test_keys:
        train_test_file.write(''.join(train_dict[key]))
        if key in test_dict:
            test_test_file.write(''.join(test_dict[key]))

    train_train_file.close()
    train_test_file.close()
    test_train_file.close()
    test_test_file.close()

if __name__ == '__main__':
    split(*argv[1:])