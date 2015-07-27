import argparse

def convert(filePath, outfile):
    xLookup = {}
    yLookup = {}
    countx = 0
    county = 0
    row_ind = []
    col_ind = []
    values = []
    f = open(filePath)
    h = f.readline()
    ll = []
    for l in f:
        line_split = l.split(',')
        uidStr = line_split[0]
        iidStr = line_split[1]
        value = float(line_split[2])
        if value > 5:
            value = 5
        if xLookup.has_key(uidStr) is False:
            xLookup[uidStr] = countx
            countx += 1
        uid = xLookup[uidStr]
        if yLookup.has_key(iidStr) is False:
            yLookup[iidStr] = county
            county += 1
        iid = yLookup[iidStr]

        row_ind.append(uid)
        col_ind.append(iid)
        values.append(value)
    f.close()

    with open(outfile, 'w') as ofile:
        for fields in zip(row_ind, col_ind, values):
            ofile.write('\t'.join(map(str, fields)))
            ofile.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="filePath", help="input file")
    parser.add_argument("--out", dest="outfile", help="output dir")
    args = vars(parser.parse_args())
    convert(**args)

