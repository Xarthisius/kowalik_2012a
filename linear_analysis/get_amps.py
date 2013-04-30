#!/usr/bin/python

import tables as h5
import numpy as np
import matplotlib
#matplotlib.use('Cairo')
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import *
import sys
import mpmath as mp
from eigens import eigen_vals
import cPickle as pickle
from yt.mods import *

matplotlib.rc("axes", linewidth=2.0)
matplotlib.rc("lines", linewidth=3.0)
matplotlib.rc("font", size=24)

Lx = 0.146484375
Lz = 0.146484375
FIELD = 'dend'

parser = argparse.ArgumentParser()
parser.add_argument("-x", type=str, default='x3_50')
parser.add_argument("files", nargs='*')
args = parser.parse_args()

SET = args.x
X0 = eval(SET[1:].replace('_','.')) # !!!! HARDCODED

u4_50 = '''
   200.005723 0.658302 0.250000 0.021034 0.015761 0.023942 0.358613 0.997498
   4.500000 0.000000 0.292969 -0.001847 -0.007386 0.001843 -0.007530
'''

w3_50 = '''
   50.001331 0.959715 0.250000 0.030586 0.014029 0.014618 0.246643 0.199753
   3.500000 0.000000 0.292969 -0.000457 -0.012980 0.000094 -0.013017
'''
w4_50 = '''
   400.008387 0.658302 0.250000 0.021043 0.015763 0.023945 0.358470 0.996586
   4.500000 0.000000 0.292969 -0.001849 -0.007391 0.001843 -0.007535
'''
x3_00 = '''
   70.001444 1.209379 0.250000 0.038282 0.012535 0.010365 0.984135 1.017591
   3.000000 0.000000 0.292969 -0.005945 -0.003903 0.006061 -0.006257
'''
x3_50 = '''
   74.001976 0.959715 0.250000 0.030585 0.013542 0.014110 1.231535 1.003307
   3.500000 0.000000 0.292969 -0.006473 -0.004700 0.006505 -0.007243
'''
x4_00 = '''
   100.002844 0.785514 0.250000 0.025223 0.014325 0.018236 1.493120 0.988094
   4.000000 0.000000 0.292969 -0.006918 -0.005275 0.006816 -0.007964
'''
x4_50 = '''
   100.002844 0.658302 0.250000 0.021299 0.015075 0.022899 1.767893 0.973427
   4.500000 0.000000 0.292969 -0.007328 -0.005761 0.007128 -0.008588
'''
l3_00 = '''
   130.002449 1.209379 0.250000 0.038144 0.012658 0.010467 0.987828
   1.024388 3.000000 0.000000 0.292969 -0.005471 -0.003500 0.005619 -0.006759
'''
l3_50 = '''
   130.002449 0.959715 0.250000 0.030408 0.013691 0.014266 1.238920
   1.014483 3.500000 0.000000 0.292969 -0.005948 -0.004257 0.006052 -0.007783
'''
l4_00 = '''
   130.002449 0.785514 0.250000 0.024979 0.014658 0.018661 1.507892
   1.006486 4.000000 0.000000 0.292969 -0.006394 -0.004850 0.006461 -0.008624
'''
l4_50 = '''
   130.002449 0.658302 0.250000 0.020996 0.015563 0.023642 1.793591
   1.000231 4.500000 0.000000 0.292969 -0.006813 -0.005343 0.006840 -0.009348
'''

def parse_str(line):
    data = map(float, line.split())
    ic = {}
    ic['Omega_K'] = mp.mpf(data[1])
    ic['cs'] = mp.mpf(data[2])
    ic['rhog'] = mp.mpf(data[3])
    ic['etavk'] = mp.mpf(data[4])
    ic['etar'] = mp.mpf(data[5])
    ic['tauf'] = mp.mpf(data[6])
    ic['eps'] = mp.mpf(data[7])
    ic['wx'] = mp.mpf(data[-4])
    ic['wy'] = mp.mpf(data[-3])
    ic['ux'] = mp.mpf(data[-2])
    ic['uy'] = mp.mpf(data[-1])
    ic['etar_min'] = 1.0
    ic['etar_max'] = 1.0
    return ic

def format_my(x, pos=None):
   return "%i" % np.log(x)

def get_means(fn):
   pf = load(fn)
   c = 0.5 * (pf.domain_left_edge + pf.domain_right_edge)
   c[0] = X0 / pf['au']
   S = pf.domain_right_edge - pf.domain_left_edge
   n_d = pf.domain_dimensions

   slc = pf.h.slice(1, c[1], fields=[FIELD])
   frb = slc.to_frb(S[2], (n_d[2], n_d[2]), height=S[2], center=c)

   z0 = c[2] * pf['au']
   hh = S[2] * pf['au']
   time = pf.current_time * pf['years']

   return frb[FIELD] / pf[FIELD], time


if len(args.files) < 1:
   parser.error("I need at least one file")

outname = "%s_%s_%s_modes.pkl" % (args.files[0].split('/')[0], FIELD, SET)

if not os.path.isfile(outname):
   my_storage = {}
   for sto, fn in parallel_objects(args.files, njobs=-1, storage = my_storage):
      dend, time = get_means(fn)
      tt = np.fft.rfft2(dend)
      sto.result_id = fn
      sto.result = (time, np.abs(tt[0:dend.shape[0]/2, :]) / (0.5 * np.product(dend.shape)))

if ytcfg.getint("yt", "__topcomm_parallel_rank") == 0:
   if not os.path.isfile(outname):
      amps = []
      time_sim = []
      for fn, vals in sorted(my_storage.items()):
          time, amp = vals
          amps.append(amp)
          time_sim.append(time)
      del my_storage
      amps = np.array(amps)
      plots = []
      kas   = []
      for i in range(0, amps.shape[1]):
         for j in range(0, amps.shape[2]):
               plots.append(amps[:, i, j])
               kas.append([i, j])
      plots = np.array(plots)
      kas   = np.array(kas)

      mask = (kas[:,0] > 0) & (kas[:,1] > 0)
      plots = plots[mask,:]
      k_max = kas[mask,:]

      N_amps = plots.shape[0]

      p = plots[0:N_amps,:]
      k = k_max[0:N_amps,:]
      time_sim = np.array(time_sim)[:,0]

      output = open(outname, 'wb')
      pickle.dump(k, output)
      pickle.dump(time_sim, output)
      pickle.dump(p, output)
      output.close()

   else:
      output = open(outname, 'rb')
      k = pickle.load(output)
      time_sim = pickle.load(output)
      p = pickle.load(output)
      N_amps = p.shape[0]
      output.close()

   a0 = []
   b0 = []
   c0 = []

   if SET == "x3_00":
      # 3.0
      tmin= np.where(abs(time_sim - 50.)< 2.1)[0][0]
      t_0 = np.where(abs(time_sim - 80.)< 2.1)[0][0]
      tmax= np.where(abs(time_sim - 90.)< 2.1)[0][0]
   elif SET == "x3_50":
      # 3.5
      tmin= np.where(abs(time_sim - 50.)< 2.1)[0][0]
      t_0 = np.where(abs(time_sim - 85.)< 2.1)[0][0]
      tmax= np.where(abs(time_sim - 100.)< 2.1)[0][0]
   elif SET == "x4_00":
      tmin= np.where(abs(time_sim - 75.)< 2.1)[0][0]
      t_0 = np.where(abs(time_sim - 130.)< 2.1)[0][0]
      tmax= np.where(abs(time_sim - 145.)< 2.1)[0][0]
   elif SET == "x4_50":
      tmin= np.where(abs(time_sim - 60.)< 2.1)[0][0]
      t_0 = np.where(abs(time_sim - 100.)< 2.1)[0][0]
      tmax= np.where(abs(time_sim - 140.)< 2.1)[0][0]
   elif SET == "x5_00":
      tmin= np.where(abs(time_sim - 74.)< 2.1)[0][0]
      t_0 = np.where(abs(time_sim - 160.)< 2.1)[0][0]
      tmax= np.where(abs(time_sim - 174.)< 2.1)[0][0]
   elif SET == "x5_50":
      tmin= np.where(abs(time_sim - 100.)< 2.1)[0][0]
      t_0 = np.where(abs(time_sim - 170.)< 2.1)[0][0]
      tmax= np.where(abs(time_sim - 200.)< 2.1)[0][0]
   elif SET == "l3_50" or SET == "l3_00" or SET == "l4_00":
      tmin= np.where(abs(time_sim - 140.)< 5.1)[0][0]
      t_0 = np.where(abs(time_sim - 185.)< 5.1)[0][0]
      tmax= np.where(abs(time_sim - 200.)< 5.1)[0][0]
   elif SET == "l4_50":
      tmin= np.where(abs(time_sim - 150.)< 5.1)[0][0]
      t_0 = np.where(abs(time_sim - 200.)< 5.1)[0][0]
      tmax= np.where(abs(time_sim - 220.)< 5.1)[0][0]
   elif SET == "w4_50":
      tmin= np.where(abs(time_sim - 400.) < 2.1)[0][0]
      t_0 = np.where(abs(time_sim - 590.) < 2.1)[0][0]
      tmax= np.where(abs(time_sim - 600.) < 2.1)[0][0]
   elif SET == "u4_50":
      tmin= np.where(abs(time_sim - 250.) < 2.1)[0][0]
      t_0 = np.where(abs(time_sim - 390.) < 2.1)[0][0]
      tmax= np.where(abs(time_sim - 400.) < 2.1)[0][0]
   else:
      sys.exit("SET=%s undefined" % SET)

   for i in range(N_amps):
      x = time_sim[tmin:tmax]
      y = p[i,tmin:tmax]
      A = np.vstack([x, np.ones(len(x))]).T
      a1, b1 = np.linalg.lstsq(A, np.log(y+1e-25))[0]
      y0 = np.exp(b1)*np.exp(a1*x)
      c0.append(p[i, t_0])
      a0.append(a1)
      b0.append(b1)

   ic = parse_str(eval(SET))

   Nmodes = 20

   c_sorted = np.flipud(np.argsort(np.array(c0)))
   c0 = np.array(c0)
   maxes = c_sorted[:Nmodes]

   if False:
      for i in range(6):
        plt.semilogy(time_sim, p[maxes[i], :], lw=1)
      plt.axis([15,250,1e-8,5e-2])
      plt.xlabel("t [yr]")
      plt.ylabel(r"$\tilde{\rho}_d$")
      plt.tight_layout()
      plt.show()

   res = []

   for ind in maxes: #[::-1]:
      kx = mp.mpf(k[ind,1]*2.0*np.pi/(2.*Lx))*(ic['etar'])
      kz = mp.mpf(k[ind,0]*2.0*np.pi/(2.*Lz))*(ic['etar'])
      my = eigen_vals(ic=ic, kx=kx, kz=kz)
      ds = np.abs(float(a0[ind])/ic['Omega_K'] - float(my.omega.imag)) / np.abs(float(my.omega.imag))
      print float(a0[ind]/ic['Omega_K']), float(my.omega.imag), int(ds*100.0), kx, k[ind,0], kz, k[ind,1], c0[ind]
#      ds = np.abs(float(a0[ind]) - float(my.omega.imag)) / np.abs(float(my.omega.imag))
#      print float(a0[ind]), float(my.omega.imag), int(ds*100.0), kx, k[ind,0], kz, k[ind,1], c0[ind]
      res.append( (int(ds*100.0), float(kx), float(kz), float(a0[ind]), float(my.omega.imag), ind) )

   if True:
      fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(16,12))
      save_me = []
      for i, item in enumerate(res[:6]):
         ax = axs.ravel()[i]
         ind = item[-1]
         func_m = np.exp(b0[ind]) * np.exp(a0[ind] * np.array(time_sim))
         func_o = np.exp(b0[ind]) * np.exp(item[-2] * np.array(time_sim))
         save_me.append((func_m, func_o, p[ind,:], time_sim))
         print a0[ind], item[-2], item[0], item[1], item[2]
         ax.plot(time_sim, func_m, 'r-')
         ax.plot(time_sim, func_o, 'c--')
         ax.plot(time_sim, p[ind, :])
         ax.set_ylim([np.e**(-16), np.e**(-4.5)])
         # txt = r"$\left|\frac{\Delta s}{s_0}\right| = %i" % (item[0])
         # txt += ' \%$'
         # ax.text(0.5, 0.1, txt, transform = ax.transAxes, fontsize='large')
         # ax.text(0.3335, 0.1, txt, transform = ax.transAxes)
         ax.set_yscale('log', basey=np.e)
         ax.yaxis.set_major_formatter(FuncFormatter(format_my))
         # ax.xaxis.set_major_locator(FixedLocator([35,45,55,65]))
         if i > 2:
            ax.set_xlabel("t [yr]")
         if i % 3 == 0:
            ax.set_ylabel(r"$\ln \left(\delta w_z\right)$")
      pickle.dump(save_me, open('save.pkl', 'wb'))
      plt.draw()
      plt.savefig(outname.replace('modes.pkl', 'grow.png'),
                  facecolor="white", bbox_inches='tight')

   output = []
   for i, item in enumerate(res[:Nmodes]):
      output.append([abs(item[1]), abs(item[1])*ic['etar_min'],
                     abs(item[1])*ic['etar_max'],
                     abs(item[2]), abs(item[2])*ic['etar_min'],
                     abs(item[2])*ic['etar_max']])

   pickle.dump(output, open(outname.replace('.pkl', '.dat'), 'wb'))
