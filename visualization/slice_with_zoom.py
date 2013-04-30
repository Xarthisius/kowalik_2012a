#!/usr/bin/python

import matplotlib
import numpy as np
import sys
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import h5py as h5
import loadct as ct
import argparse
from matplotlib.patches import Rectangle, ConnectionPatch

matplotlib.rc("axes", linewidth=1.5, labelsize="large")
matplotlib.rc("lines", markeredgewidth=1.5)
matplotlib.rc("font", size=20)

def read_data_hdf5(name,dset):
   temp = {}
   h5f = h5.File(name)
   temp[dset] = h5f[dset][:,0,:]
   for f in h5f.attrs.keys():
      temp[f] = h5f.attrs[f]
   h5f.close()
   return temp

colormap = ct.p05
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',colormap)

def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)


usage = "usage: %prog [options] FILES"

parser = argparse.ArgumentParser()
parser.add_argument("-x", nargs=2, default=[4.25, 8.25])
parser.add_argument("-z", nargs=2, default=[0.0, 0.0])
parser.add_argument("files", nargs='*')

args = parser.parse_args()
xc = map(np.float64,args.x)
zc = map(np.float64,args.z)

my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', ct.p05)

h  = 0.25

patch1 = [xc[0]-h, xc[0]+h, zc[0]-h, zc[0]+h]
patch2 = [xc[1]+h, xc[1]-h, zc[1]-h, zc[1]+h]
vmin = 0.0
vmax = 0.5


for fn in args.files:

   foo = read_data_hdf5(fn, 'dend')

   ext = [ foo['xmin'][0], foo['xmax'][0], foo['zmin'][0], foo['zmax'][0] ]
   fig = plt.figure(0, figsize=(14,10))
   fig.clf()
   ax1 = plt.subplot2grid((4,8), (0,0), colspan=8)
   ax1.axis([2.5, 10.0, foo['zmin'], foo['zmax']])
   ax1.imshow(foo['dend'], extent=ext, vmin=vmin, vmax=vmax, cmap=my_cmap)
   for f in [ patch1, patch2 ]:
      xy = (f[0], f[2])
      rect = Rectangle(xy, f[1]-f[0] , f[3]-f[2], facecolor="#aaaaaa", alpha=0.85)
      ax1.add_patch(rect)
   ax1.set_xlabel("R [AU]")
   ax1.set_ylabel("z [AU]")

   ax2 = plt.subplot2grid((4,8), (1,0), colspan=4, rowspan=3)
   ax2.axis(patch1)
   ax2.imshow(foo['dend'], extent=ext, vmin=vmin, vmax=vmax, cmap=my_cmap)
   ax2.set_aspect(1.0)
   ax2.set_xlabel("R [AU]")
   ax2.set_ylabel("z [AU]")


   ax3 = plt.subplot2grid((4,8), (1,4), colspan=4, rowspan=3)
   ax3.axis(patch2)
   ax3.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
   ax3.imshow(foo['dend'], extent=ext, vmin=vmin, vmax=vmax, cmap=my_cmap)
   ax3.set_aspect(1.0)
   ax3.set_xlabel("R [AU]")
   ax3.tick_params(axis="y",labelright=True, labelleft=False)

   xy = [ [(patch1[0],patch1[3]), (patch1[0],patch1[2])],
          [(patch1[1],patch1[3]), (patch1[1],patch1[2])] ]
   for f in xy:
      arrow = ConnectionPatch( xyA=f[0], xyB=f[1],
              axesA=ax2, axesB=ax1, coordsA="data", coordsB="data",
              arrowstyle="<-", shrinkA=3.0, shrinkB=3.0)
      ax2.add_artist(arrow)

   xy = [ [(patch2[0],patch2[3]), (patch2[0],patch2[2])],
          [(patch2[1],patch2[3]), (patch2[1],patch2[2])] ]
   for f in xy:
      arrow = ConnectionPatch( xyA=f[0], xyB=f[1],
              axesA=ax3, axesB=ax1, coordsA="data", coordsB="data",
              arrowstyle="<-", shrinkA=3.0, shrinkB=3.0)
      ax3.add_artist(arrow)

   fig.suptitle("Time = %4.0i [yr]" % foo['time'][0])
   plt.draw()
   plt.savefig(fn.replace('.h5','_zoom.png'))
   print("%s written"%fn.replace('.h5','_zoom.png'))
