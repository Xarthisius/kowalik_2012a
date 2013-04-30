#!/usr/bin/python

import sys
from yt.mods import *
import argparse
import numpy as np

grain_dens = 1.6
yr = 1./ (3600.*24.*365.25)
cm =  6.68458134e-14
gram =  5.267038870746866e-31
M_jup = 1.8986e30
au = 1/cm
G = 6.674e-8
gram_cm2 = gram/cm**2
ptmass = 1047.7
cs = 0.25

grain_size = 50.0  # !!! HARDCODED !!!!
fields = ['dend', 'denn', 'vlxd', 'vlyd', 'vlzd', 'vlxn', 'vlyn', 'vlzn']

parser = argparse.ArgumentParser()
parser.add_argument("-x", type=float, default=0.0)
parser.add_argument("files", nargs='*')
args = parser.parse_args()

x0 = args.x

def _EffectiveSoundSpeed(field, data):
    """\sqrt{c_s^2 + |\\vec{v} - \\vec{u}|}"""
    return ((data["vlxd"] - data["vlxn"])**2.0 + \
            (data["vlyd"] - data["vlyn"])**2.0 + \
            (data["vlzd"] - data["vlzn"])**2.0 )
add_field("effcs", function=_EffectiveSoundSpeed,
          particle_type=False,
          take_log=False, units=r"\rm{cm}/\rm{s}")

def print_vels(vals):
    tauf = vals[6]
    omega = vals[1]
    etavk = vals[4]
    eps = vals[7]
    taus = tauf * omega
    wx = -2.0 * taus *etavk / ((1.0 + eps) ** 2 + taus ** 2)
    ux = -eps * wx
    wy = (1.0 + eps) / (2.0 * taus) * wx
    uy = (1.0 + eps + taus ** 2) / (2.0 * taus) * wx
    print vals[0]
    print "self.wx = mp.mpf(%f)" % vals[-4][0] , wx
    print "self.ux = mp.mpf(%f)" % vals[-2][0] , ux
    print "self.wy = mp.mpf(%f)" % vals[-3][0], wy
    print "self.uy = mp.mpf(%f)" % vals[-1][0], uy
    print "tau_s = %f" % (taus)

my_storage = {}
for sto, fn in parallel_objects(args.files, njobs=-1, storage = my_storage):
   pf = load(fn)
   c = 0.5 * (pf.domain_left_edge + pf.domain_right_edge)
   c[0] = x0 / pf['au']
   S = pf.domain_right_edge - pf.domain_left_edge
   n_d = pf.domain_dimensions

   slc = pf.h.slice(1, c[1], fields=fields)
   frb = slc.to_frb(S[2], (n_d[2], n_d[2]), height=S[2], center=c)

   z0 = c[2] * pf['au']
   hh = S[2] * pf['au']
   time = pf.current_time * pf['years']
   Omega = np.sqrt(G*ptmass*M_jup / (x0/pf['au'])**3) / yr
   vk    = Omega * x0
   dx    = S[0] / n_d[0] * pf['au']
   denn = frb['denn'] / pf['denn']
   gradr = np.gradient(np.average(denn, axis=0), dx)
   rhog = frb['denn'].mean() / pf['denn']
   rhop = frb['dend'].mean() / pf['dend']
   efcs = np.sqrt(frb['effcs']).mean() / pf['vlxd']
   eps  = np.mean(frb['dend']/frb['denn'])
   eta  = -cs**2*np.mean(gradr) / (2*rhog*Omega**2*x0)
   f_eta= -cs**2*gradr/ (2*rhog*Omega**2*x0)
   etavk = eta*vk
   tauf  = grain_dens*grain_size*gram_cm2 / (rhog*np.sqrt(cs**2 + efcs**2))

   wx = frb['vlxd'].mean() / pf['vlxd']
   wy = frb['vlyd'].mean() / pf['vlyd'] - Omega * x0
   ux = frb['vlxn'].mean() / pf['vlxn'] 
   uy = frb['vlyn'].mean() / pf['vlyn'] - Omega * x0

   out = (time[0], Omega, cs, rhog[0], etavk[0], eta[0]*x0, tauf[0],
         eps, x0, z0, hh, wx, wy, ux, uy)
   sto.result_id = fn
   sto.result = out

if ytcfg.getint("yt", "__topcomm_parallel_rank") == 0:
   fname = "%s_x%4.2f.dat" % (args.files[0].split('/')[0], x0)
   f = open(fname, 'wb')
   f.write("# time, Omega, cs, rhog, etavk, etar, tauf, eps, x0, z0, hh, wx, wy, ux, uy\n")
   #print("# time, Omega, cs, rhog, etavk, etar, tauf, eps, x0, z0, hh, wx, wy, ux, uy\n")
   for fn, vals in sorted(my_storage.items()):
       f.write("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n" % vals)
   #   print("%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n" % vals)
   #   print_vels(vals)
   f.close()
