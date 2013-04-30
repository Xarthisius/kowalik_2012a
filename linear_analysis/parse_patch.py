#!/usr/bin/python

import re
import sys
import mpmath as mp
from patch import generate_patch

f = open(sys.argv[1], 'r')

keys = ['Omega_K', 'cs', 'rhog', 'etavk', 'etar', 'tauf', 'eps']

ics = []

for line in f.readlines():
    if (not re.match('#', line)):
        d = line.split()[1:8]
        ic = {}
        for i, key in enumerate(keys):
            ic[key] = mp.mpf(d[i])
        ics.append(ic)
f.close()

for i, item in enumerate(ics):
    generate_patch(i, item, sys.argv[2])
