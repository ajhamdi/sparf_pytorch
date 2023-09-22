import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path1', type=str)
parser.add_argument('path2', type=str)
args = parser.parse_args()

z1 = np.load(args.path1)
z2 = np.load(args.path2)

assert z1.keys() == z2.keys()

for k in z1.keys():
    assert k in z2.keys()
    err = np.max(np.abs(z1[k] - z2[k]))
    assert err < 1e-10
