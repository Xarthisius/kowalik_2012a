#!/usr/bin/env python

import sys
import os
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import h5py as h5
import numpy as np
import cPickle as pickle

matplotlib.rc("axes", linewidth=1.5)
matplotlib.rc("lines", markeredgewidth=2.0)
matplotlib.rc("font", size=18)

x0 = sys.argv[1]
l0 = x0.replace('x', 'l')

fname = '%s.h5' % x0
outname = '%s.eps' % x0

legend = {
    "3d_%s_modes.dat" % x0: r"$150^2$ 3D",
    "high_%s_modes.dat" % x0: r"$300^2$",
    "ultra_%s_modes.dat" % x0: r"$600^2$"
}
color = {
    "3d_%s_modes.dat" % x0: "1",
    "high_%s_modes.dat" % x0: "+",
    "ultra_%s_modes.dat" % x0: "x"
}

locs = {
    "x3_00_50e1": [(0.95, 6.0), (0.7, 4.5), (0.54, 2.3), (0.4, 2.1),
                   (0.31, 1.5), (0.25, 0.9), (0.2, 0.6), (0.16, 0.46),
                   (0.12, 0.3)],
    "x4_50_10e1": [(0.9, 2.), (0.7, 0.9), (0.56, 1.3), (0.44, 1.34),
              (0.36, 1.0), (0.28, 1.0), (0.22, 0.7), (1.13, 2.3),
              (1.4, 5.1), (1.8, 5.1), (2.2, 5.4), (2.85, 6.5),
              (7.4, 4.7)]
}

ic = {}

h5f = h5.File(os.path.join('data',fname))
data = np.log10(h5f['A1'][:].T)

for key in ['Omega_K', 'cs', 'rhog', 'etavk', 'tauf', 'eps']:
    ic[key] = h5f['A1'].attrs[key]

etar = h5f['A1'].attrs['etar']
etavk = h5f['A1'].attrs['etavk']
k_min = h5f['A1'].attrs['k_min']
k_max = h5f['A1'].attrs['k_max']
h5f.close()

x = np.logspace(np.log10(k_min), np.log10(k_max), num=72)
fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
ax.set_yscale('log')
ax.set_xscale('log')
X, Y = np.meshgrid(x, x)

result = []

CS = ax.contour(X, Y, data, np.arange(
    -2.8, 0.0, 0.2), colors='0.2', linestyles='solid')
plt.clabel(CS, fontsize=10, inline=1, manual=locs[x0])
plt.xlabel("$k_x\\,\\eta r$")
plt.ylabel("$k_z\\,\\eta r$")
nn = 12
for rn, fn in enumerate(legend.keys()):
    fname = os.path.join('data', fn)
    if not os.path.isfile(fname):
        continue
    b = np.array(pickle.load(open(fname, "rb")))
    ex = np.array([b[:, 0]-b[:, 1], b[:, 2]-b[:, 0]])
    ey = np.array([b[:, 3]-b[:, 4], b[:, 5]-b[:, 3]])
    plt.scatter(b[:nn, 0], b[
                :nn, 3], 100, c='k', marker=color[fn], label=legend[fn])
plt.axis([1e-1, 1e1, 1e-1, 1e1])
plt.legend(loc=4, prop={'size': 20})
plt.draw()
plt.savefig(outname, facecolor="white", bbox_inches='tight')
