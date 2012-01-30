#!/usr/bin/python

from streaming import dispersion_eq
from mpi4py import MPI
import mpmath as mp
import numpy as np

#Omega, cs, rhog, etavk, etar tauf, eps, nxd, nzd
#0.785525711295 0.25 0.0890996054549 -0.01047792055 -0.013338736593 0.42331988635 1.63946012091

ic = {'Omega_K' : mp.mpf('1.0'),
      'cs'      : mp.mpf('1.0'),
      'rhog'    : mp.mpf('1.0'),
      'etavk'   : mp.mpf('0.05'),
      'etar'    : mp.mpf('0.05'),
      'tauf'    : mp.mpf('1.0'),
      'eps'     : mp.mpf('3.0')}

# stream_new_50cm 300  - 400 yr
#0.0897128 -0.0104597 -0.0133156 0.420626 1.61967
#ic = {'Omega_K' : mp.mpf('0.785525711295'),
#      'cs'      : mp.mpf('0.25'),
#      'rhog'    : mp.mpf('0.0897128'),
#      'etavk'   : mp.mpf('-0.0104597'),
#      'etar'    : mp.mpf('-0.0133156'),
#      'tauf'    : mp.mpf('0.420626'),
#      'eps'     : mp.mpf('1.61967')
#      }

N = 64

psize = np.array([2,2])
k = np.array([N,N])
n_b = k/psize
result = []

x = 10.0**np.linspace(-1.0, 3.0, N)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
cart_comm = comm.Create_cart(psize)

if rank == 0:
   print "min(kx) = %f, max(kx) = %f"  % (1*ic['etar'], 100*ic['etar'])

pcoords = cart_comm.topo[2]
for kx in range(pcoords[0]*n_b[0], (pcoords[0]+1)*n_b[0]):
   for kz in  range(pcoords[1]*n_b[1], (pcoords[1]+1)*n_b[1]):
#     k_x = kx*ic['etar']
#     k_z = kz*ic['etar']
      k_x = x[kx]
      k_z = x[kz]
      my = dispersion_eq(Omega_K=ic['Omega_K'],cs=ic['cs'],rhog=ic['rhog'],etavk=ic['etavk'],
                         tauf=ic['tauf'],eps=ic['eps'],kx=mp.mpf(k_x),kz=mp.mpf(k_z))
      result.append([k_x,k_z,my.solve_eq()[0]])

result = cart_comm.gather(result, root=0)
#rint sorted(result, key=lambda x: x[2].imag, reverse=True)

if rank == 0:
   output = []
#  tab = sorted(np.reshape(np.array(result), (np.product(k),3)) , key=lambda x: x[2].imag, reverse=True)
   tab = np.reshape(np.array(result), (np.product(k),3))
   for item in tab:
   #  if (item[2].imag > 0.0):
      print "kx = %f, kz = %f, s/Omega = %f" % (np.abs(item[0]), np.abs(item[1]), item[2].imag/ic['Omega_K'])
      output.append([np.abs(item[0]), np.abs(item[1]), item[2].imag/ic['Omega_K']])
   output = np.array(output)
   print np.shape(output)
   A = np.zeros((N,N))
   for kx in range(0,N):
      for kz in range(0,N):
         A[kx,kz] = np.max(output[np.where((output[:,0] == x[kx]) & (output[:,1]==x[kz])),2],1.0e-9)
   import matplotlib.pyplot as plt
   from matplotlib.colors import LogNorm
   fig = plt.figure()
   ax = fig.add_subplot(1,1,1)
   ax.set_yscale('log')
   ax.set_xscale('log')
   X,Y = np.meshgrid(x,x)

   CS = ax.contour(X,Y,np.log10(A.T),8 ,colors='k', linestyles='solid')
   plt.clabel(CS, fontsize=9, inline=1)
   ax.plot([0.11],[1.0],'o')
   plt.draw()
   plt.savefig('test.png')

