#!/usr/bin/python

import matplotlib
import sys
matplotlib.use('cairo')
import matplotlib.pyplot as plt
import h5py as h5
import loadct as ct
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

foo = read_data_hdf5(sys.argv[1], 'dend')

colormap = ct.p05
my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',colormap)

def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

ext = [ foo['xmin'][0], foo['xmax'][0], foo['zmin'][0], foo['zmax'][0] ]
patch1 = [3.75,4.25,-0.25,0.25]
patch2 = [7.75,8.25,-0.25,0.25]
vmin = 0.0
vmax = 0.5

plt.figure(0, figsize=(14,10))
ax1 = plt.subplot2grid((4,8), (0,0), colspan=8)
ax1.axis([2.5, 10.0, foo['zmin'], foo['zmax']])
ax1.imshow(foo['dend'], extent=ext, vmin=vmin, vmax=vmax, cmap=my_cmap)
for f in [ (patch1[0], patch1[2]), (patch2[0], patch2[2])]:
   rect = Rectangle(f, 0.5, 0.5, facecolor="#aaaaaa", alpha=0.8)
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
ax3.imshow(foo['dend'], extent=ext, vmin=vmin, vmax=vmax, cmap=my_cmap)
ax3.set_aspect(1.0)
ax3.set_xlabel("R [AU]")
for tl in ax3.get_yticklabels():
   tl.set_visible(False)

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

plt.suptitle("subplot2grid")
#make_ticklabels_invisible(plt.gcf())
plt.draw()
plt.savefig('demo.png')

