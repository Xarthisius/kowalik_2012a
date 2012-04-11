#!/usr/bin/python

import matplotlib
import sys
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import h5py as h5
import loadct as ct
from matplotlib.patches import Rectangle, ConnectionPatch
from yt.mods import *
import numpy as np

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

c1 = [4.25,np.pi/2,0.0]
c2 = [8.25,np.pi/2,0.0]
patch1 = [4.0,4.5,-0.25,0.25]
patch2 = [8.0,8.5,-0.25,0.25]
vmin = 0.0
vmax = 0.5


for fn in sys.argv[1:]:

   pf = load(fn)
   field = "dend"

   le = pf.domain_left_edge
   re = pf.domain_right_edge

   s = pf.h.slice(1, np.pi/2.0, fields=["dend"])

   ext = [ le[0], re[0], le[2], re[2] ]
   fig = plt.figure(0, figsize=(14,10))
   fig.clf()
   ax1 = plt.subplot2grid((4,8), (0,0), colspan=8)
   ax1.axis([2.5, 10.0, le[2], re[2]])
   img = s.to_frb(7.5, (int(7.5*512),512), center=[6.25,np.pi/2.,0.0], height=1.0)
   ax1.imshow(img['dend'], extent=[2.5, 10.0, le[2], re[2]], cmap=my_cmap)
   for f in [ patch1, patch2 ]:
      xy = (f[0], f[2])
      rect = Rectangle(xy, f[1]-f[0] , f[3]-f[2], facecolor="#aaaaaa", alpha=0.85)
      ax1.add_patch(rect)
   ax1.set_xlabel("R [AU]")
   ax1.set_ylabel("z [AU]")

   ax2 = plt.subplot2grid((4,8), (1,0), colspan=4, rowspan=3)
   ax2.axis(patch1)
   img = s.to_frb(0.5, (512,512),center=c1)
   ax2.imshow(img['dend'], cmap=my_cmap,  extent=patch1)
   ax2.set_aspect(1.0)
   ax2.set_xlabel("R [AU]")
   ax2.set_ylabel("z [AU]")

   ax3 = plt.subplot2grid((4,8), (1,4), colspan=4, rowspan=3)
   ax3.axis(patch2)
   ax3.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
   img = s.to_frb(0.5, (512,512),center=c2)
   ax3.imshow(img['dend'], extent=patch2, cmap=my_cmap)
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

   fig.suptitle("Time = %4.0i [yr]" % pf.current_time)
   plt.draw()
   plt.savefig(fn.replace('.h5','_zoom.png'))
   print("%s written"%fn.replace('.h5','_zoom.png'))
