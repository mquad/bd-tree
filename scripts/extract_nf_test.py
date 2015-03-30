__author__ = 'massimo'

from os import path
from sys import argv

def extract(full_fpath, probe_fpath):
    wdir, full_fname = path.split(full_fpath)
    train_fpath = path.join(wdir, path.splitext(full_fname)[0] + '.train')
    test_fpath = path.join(wdir, path.splitext(full_fname)[0] + '.test')
    with open(full_fpath, 'r') as full_file,\
        open(probe_fpath, 'r') as probe_file,\
        open(train_fpath, 'w') as train_file,\
        open(test_fpath, 'w') as test_file:
        for pline in probe_file:
            pline = pline.strip()
            if pline.endswith(':'):
                item = pline.lstrip(':')
            else:
                user = pline
                while True:
                    fline = full_file.readline()
                    if fline.split('\t')[:1] != [user, item]:
                        train_file.write(fline)
                    else:
                        test_file.write(fline)
                        break

if __name__ == "__main__":
    extract(*argv[1:])