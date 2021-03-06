#!/usr/bin/python

import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import h5py as h5
import loadct as ct
import argparse
from matplotlib.patches import Rectangle, ConnectionPatch
from yt.mods import \
    load, parallel_objects
import numpy as np
from mpi4py import MPI

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

def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

parser = argparse.ArgumentParser()
parser.add_argument("-x", nargs=2, default=[4.2, 5.55])
parser.add_argument("-z", nargs=2, default=[0.0, 0.0])
parser.add_argument("files", nargs='*')

args = parser.parse_args()
xc = map(np.float64,args.x)
zc = map(np.float64,args.z)

my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', ct.p05)

h  = 0.146484375
phi = 0.5

c1 = np.array([xc[0], phi, zc[0]])
c2 = np.array([xc[1], phi, zc[1]])
patch1 = [c1[0] - h, c1[0] + h, c1[2]-h, c1[2]+h]
patch2 = [c2[0] - h, c2[0] + h, c2[2]-h, c2[2]+h]
vmin = 0.0
vmax = 0.1 * 0.2
first_pass = True
for fn in parallel_objects(args.files, njobs=-1):
   pf = load(fn)
   field = "dend"
   le = pf.domain_left_edge * pf['au']
   re = pf.domain_right_edge * pf['au']
   s = pf.h.slice(1, phi, fields=["dend"])
   # = pf.h.proj(1, 'dend')
   #fac = pf['au'] / (2.0 * pf['dend'] * np.pi)
   fac = 1./pf['dend']
   if first_pass:
      c1 /= pf.units['au']
      c2 /= pf.units['au']

   ext = [ le[0], re[0], le[2], re[2] ]
   fig = plt.figure(0, figsize=(14,10))
   fig.clf()
   ax1 = plt.subplot2grid((4,8), (0,0), colspan=8)
   ax1.axis([3.0, 6.0, le[2], re[2]])

   img = s.to_frb(3/pf['au'], (int(0.29296875*1024), 3*1024), center=np.array([4.5, phi, 0.0])/pf.units['au'],
                  height=2.*h/pf['au'])
   ax1.imshow(img['dend'] * fac, extent=[3.0, 6.0, le[2], re[2]], cmap=my_cmap,
         vmin=vmin, vmax=vmax)
   for f in [ patch1, patch2 ]:
      xy = (f[0], f[2])
      rect = Rectangle(xy, f[1]-f[0] , f[3]-f[2], facecolor="#aaaaaa", alpha=0.85)
      ax1.add_patch(rect)
   ax1.set_xlabel("R [AU]")
   ax1.set_ylabel("z [AU]")

   ax2 = plt.subplot2grid((4,8), (1,0), colspan=4, rowspan=3)
   ax2.axis(patch1)
   img = s.to_frb(2*h/pf['au'], (512,512), center=c1)
   ax2.imshow(img['dend'] * fac, cmap=my_cmap, extent=patch1,
         vmin=vmin, vmax=vmax, interpolation="nearest")
   ax2.set_aspect(1.0)
   ax2.set_xlabel("R [AU]")
   ax2.set_ylabel("z [AU]")

   ax3 = plt.subplot2grid((4,8), (1,4), colspan=4, rowspan=3)
   ax3.axis(patch2)
   ax3.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
   img = s.to_frb(2*h/pf['au'], (512,512),center=c2)
   ax3.imshow(img['dend'] * fac, extent=patch2, cmap=my_cmap,
         vmin=vmin, vmax=vmax, interpolation="nearest")
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

   fig.suptitle("Time = %4.0i [yr]" % (pf.current_time / (3600.*24.*365.25)))
   plt.draw()
   plt.savefig(fn.replace('.h5','_zoom.png'))
   print("%s written"%fn.replace('.h5','_zoom.png'))
   del ax1, ax2, ax3, fig, s, pf, img
   first_pass = False
