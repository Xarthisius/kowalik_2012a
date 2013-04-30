#!/usr/bin/python

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.ticker
import h5py as h5
import loadct as ct
import argparse
from matplotlib.patches import Rectangle, ConnectionPatch
from yt.mods import *
import numpy as np

#matplotlib.rc("axes", linewidth=1.5, labelsize="large")
matplotlib.rc("axes", linewidth=1.5)
matplotlib.rc("lines", markeredgewidth=1.5)
matplotlib.rc("font", size=28)
matplotlib.rc("xtick.major", pad=10)
matplotlib.rc("ytick.major", pad=10)


def read_data_hdf5(name,dset):
   temp = {}
   h5f = h5.File(name)
   temp[dset] = h5f[dset][:,0,:]
   for f in h5f.attrs.keys():
      temp[f] = h5f.attrs[f]
   h5f.close()
   return temp

def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

usage = "usage: %prog [options] FILES"

parser = argparse.ArgumentParser()
parser.add_argument("-x", nargs=4, default=[4.25, 6.5, 5.5, 9.35])
parser.add_argument("-z", nargs=4, default=[0.0, 0.0, 0.0, -0.25])
parser.add_argument("files", nargs='*')

args = parser.parse_args()
xc = map(np.float64, args.x)
zc = map(np.float64, args.z)

my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', ct.p05)

h  = 0.25
phi = 0.52359878*0.5

c = []
patch = []
for i in range(len(xc)):
    c.append([xc[i], phi, zc[i]])
    patch.append([xc[i] - h, xc[i] + h, zc[i] - h, zc[i] + h])

ticks = {'ltb': [True, True, False, False], 'llr': [True, False, True, False] }
ax_pos = [ (0,0), (0,2), (4,0), (4,2) ]

vmin = 0.0
vmax = 0.5

for fn in parallel_objects(args.files, njobs=1):
#for fn in args.files:

   pf = load(fn)
   field = "dend"

   le = pf.domain_left_edge
   re = pf.domain_right_edge

   s = pf.h.slice(1, phi, fields=["dend","vlxd","vlzd"])

   ext = [ le[0], re[0], le[2], re[2] ]
   fig = plt.figure(0, figsize=(15,16.5))
   fig.clf()
   ax1 = plt.subplot2grid((7,4), (3,0), colspan=4)
   ax1.axis([2.5, 10.0, le[2], re[2]])
   img = s.to_frb(7.5, (int(7.5*1024),1024), center=[6.25, phi, 0.0], height=4*h)
   ax1.imshow(img['dend'], extent=[2.5, 10.0, le[2], re[2]], cmap=my_cmap,
         vmin=vmin, vmax=vmax, origin='lower')
   for f in patch:
      xy = (f[0], f[2])
      rect = Rectangle(xy, f[1]-f[0] , f[3]-f[2], facecolor="#aaaaaa", alpha=0.85)
      ax1.add_patch(rect)
## ax1.set_xlabel("R [AU]")
   ax1.set_ylabel("z [AU]")
   ax1.tick_params(axis="x",labelbottom=False, labeltop=False)

   for i in range(len(xc)):
       ax = plt.subplot2grid((7,4), ax_pos[i], colspan=2, rowspan=3)
       ax.axis(patch[i])
       img = s.to_frb(0.5, (512,512),center=c[i])
       ax.imshow(img['dend'], cmap=my_cmap, extent=patch[i],
                 vmin=vmin, vmax=vmax, interpolation="nearest",
                 origin='lower')
       print i, ticks['llr'][i], (not ticks['llr'][i])
       ax.tick_params(axis="x",labelbottom=(not ticks['ltb'][i]), labeltop=ticks['ltb'][i])
       ax.tick_params(axis="y",labelleft=ticks['llr'][i], labelright=(not ticks['llr'][i]))
       ax.set_aspect(1.0)
       ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
       if i < 2:
           xy = [ (patch[i][0],patch[i][2]), (patch[i][1],patch[i][2]) ]
       else:
           xy = [ (patch[i][0],patch[i][3]), (patch[i][1],patch[i][3]) ]
       for f in xy:
          arrow = ConnectionPatch( xyA=f, xyB=f,
               axesA=ax, axesB=ax1, coordsA="data", coordsB="data",
               arrowstyle="<-", shrinkA=3.0, shrinkB=3.0)
          ax.add_artist(arrow)

   fig.suptitle("Time = %4.0i [yr]" % pf.current_time)
   plt.draw()
   plt.savefig(fn.replace('.h5','_zoom.pdf'))
   print("%s written"%fn.replace('.h5','_zoom.pdf'))
