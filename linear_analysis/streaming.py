#!/usr/bin/python

import mpmath as mp
import numpy as np

class dispersion_eq:
   """Class for solving dispersion equation for streaming instability"""
   omega = []
   def __init__(self,Omega_K,cs,rhog,etavk,tauf,eps,kx,kz):
      self.Omega_K = Omega_K
      self.etavk = etavk
      self.cs = cs
      self.rhog = rhog
      self.tauf = tauf
      self.eps = eps
      self.kx = kx / self.etavk
      self.kz = kz / self.etavk
      self.set_derived()

   def set_derived(self):
      self.rhop = self.eps*self.rhog
      self.taus = self.Omega_K*self.tauf
      self.wx = mp.mpf(-2)*self.taus*self.etavk/( (mp.mpf(1)+self.eps)**2+self.taus**2)
      self.ux = -self.eps*self.wx
      self.wy = (mp.mpf(1)+self.eps)/(mp.mpf(2)*self.taus)*self.wx
      self.uy = (mp.mpf(1)+self.eps+self.taus**2)/(mp.mpf(2)*self.taus)*self.wx

   def get_vars(self):
      return self.Omega_K, self.cs, self.rhog, self.etavk, self.tauf, self.eps, self.kx, self.kz, self.rhop, self.taus, self.wx, self.ux, self.wy, self.uy

   def _c0(self):
      Omega_K, cs, rhog, etavk, tauf, eps, kx, kz, rhop, taus, wx, ux, wy, uy = self.get_vars()
      c0 = -( (( (2*1j*Omega_K*tauf**2*rhop*cs**2*kx**3*ux+2*eps*Omega_K*tauf*rhop*cs**2*kx**2)*wx + 
         2*Omega_K*tauf*rhop*cs**2*kx**2*ux+2*1j*Omega_K**3*tauf**2*rhop*cs**2*kx)*wy + 
         ((-2*1j*Omega_K*tauf**2*rhop*cs**2*kx**3*ux-2*eps*Omega_K*tauf*rhop*cs**2*kx**2)*wx - 
         2*Omega_K*tauf*rhop*cs**2*kx**2*ux-2*1j*Omega_K**3*tauf**2*rhop*cs**2*kx)*uy+
         (tauf**4*rhog*cs**2*kx**6*ux**2-2*1j*eps*tauf**3*rhog*cs**2*kx**5*ux+(-Omega_K**2*tauf**4-eps**2*tauf**2)*rhog*cs**2*kx**4)*wx**4+
         (-3*1j*tauf**3*rhog*cs**2*kx**5*ux**2-4*eps*tauf**2*rhog*cs**2*kx**4*ux+(3*1j*Omega_K**2*tauf**3+1j*eps**2*tauf)*rhog*cs**2*kx**3)*wx**3+
         ((-Omega_K**2*tauf**4-3*tauf**2)*rhog*cs**2*kx**4*ux**2+(2*1j*eps*Omega_K**2*tauf**3+2*1j*eps*tauf)*rhog*cs**2*kx**3*ux+
         ((Omega_K**4*tauf**4+(eps**2+2*eps+3)*Omega_K**2*tauf**2)*rhog-Omega_K**2*tauf**2*rhop)*cs**2*kx**2)*wx**2+
         ((1j*Omega_K**2*tauf**3+1j*tauf)*rhog*cs**2*kx**3*ux**2+2*eps*Omega_K**2*tauf**2*rhog*cs**2*kx**2*ux+((1j*eps+1j)*Omega_K**2*tauf*rhop+
         ((-1j*eps**2-2*1j*eps-1j)*Omega_K**2*tauf-1j*Omega_K**4*tauf**3)*rhog)*cs**2*kx)*wx+
         Omega_K**2*tauf**2*rhop*cs**2*kx**2*ux**2+(-1j*eps-1j)*Omega_K**2*tauf*rhop*cs**2*kx*ux)*kz**2
         +((-2*1j*Omega_K*tauf**2*rhop*kx**5*ux**2-2*eps*Omega_K*tauf*rhop*kx**4*ux)*wx**2+
         (-2*1j*Omega_K*tauf**2*rhop*kx**5*ux**3+(-4*eps-4)*Omega_K*tauf*rhop*kx**4*ux**2+
         (2*1j*Omega_K*tauf**2*rhop*cs**2*kx**5+(2*1j*eps**2+2*1j*eps)*Omega_K*rhop*kx**3)*ux+
         2*eps*Omega_K*tauf*rhop*cs**2*kx**4)*wx-
         2*Omega_K*tauf*rhop*kx**4*ux**3+(2*1j*eps+2*1j)*Omega_K*rhop*kx**3*ux**2+2*Omega_K*tauf*rhop*cs**2*kx**4*ux)*wy
         +((2*1j*Omega_K*tauf**2*rhop*kx**5*ux**2+2*eps*Omega_K*tauf*rhop*kx**4*ux)*wx**2+
         (2*1j*Omega_K*tauf**2*rhop*kx**5*ux**3+(4*eps+4)*Omega_K*tauf*rhop*kx**4*ux**2+
         ((-2*1j*eps**2-2*1j*eps)*Omega_K*rhop*kx**3-2*1j*Omega_K*tauf**2*rhop*cs**2*kx**5)*ux-2*eps*Omega_K*tauf*rhop*cs**2*kx**4)*wx+
         2*Omega_K*tauf*rhop*kx**4*ux**3+(-2*1j*eps-2*1j)*Omega_K*rhop*kx**3*ux**2-2*Omega_K*tauf*rhop*cs**2*kx**4*ux)*uy
         +(-tauf**4*rhog*kx**8*ux**4+3*1j*eps*tauf**3*rhog*kx**7*ux**3+(tauf**4*rhog*cs**2*kx**8+(Omega_K**2*tauf**4+3*eps**2*tauf**2)*rhog*kx**6)*ux**2+
         ((-1j*eps*Omega_K**2*tauf**3-1j*eps**3*tauf)*rhog*kx**5-2*1j*eps*tauf**3*rhog*cs**2*kx**7)*ux-
         eps**2*tauf**2*rhog*cs**2*kx**6)*wx**4
         +(3*1j*tauf**3*rhog*kx**7*ux**4+(tauf**2*rhop+6*eps*tauf**2*rhog)*kx**6*ux**3+
         (((-3*1j*Omega_K**2*tauf**3-3*1j*eps**2*tauf)*rhog-2*1j*eps*tauf*rhop)*kx**5-3*1j*tauf**3*rhog*cs**2*kx**7)*ux**2+
         ((-eps**2*rhop-2*eps*Omega_K**2*tauf**2*rhog)*kx**4-4*eps*tauf**2*rhog*cs**2*kx**6)*ux+1j*eps**2*tauf*rhog*cs**2*kx**5)*wx**3
         +(((Omega_K**2*tauf**4+3*tauf**2)*rhog-tauf**2*rhop)*kx**6*ux**4+((2*1j*eps-2*1j)*tauf*rhop+(-3*1j*eps*Omega_K**2*tauf**3-3*1j*eps*tauf)*rhog)*kx**5*ux**3+
         ((-Omega_K**2*tauf**4-3*tauf**2)*rhog*cs**2*kx**6+((Omega_K**2*tauf**2+eps**2-2*eps)*rhop+
         ((-3*eps**2-2*eps-3)*Omega_K**2*tauf**2-Omega_K**4*tauf**4)*rhog)*kx**4)*ux**2+((2*1j*eps*Omega_K**2*tauf**3+2*1j*eps*tauf)*rhog*cs**2*kx**5+
         ((1j*eps*Omega_K**4*tauf**3+(1j*eps**3+2*1j*eps**2+1j*eps)*Omega_K**2*tauf)*rhog-1j*eps*Omega_K**2*tauf*rhop)*kx**3)*ux+
         eps**2*Omega_K**2*tauf**2*rhog*cs**2*kx**4)*wx**2
         +((2*1j*tauf*rhop+(-1j*Omega_K**2*tauf**3-1j*tauf)*rhog)*kx**5*ux**4+((-Omega_K**2*tauf**2+2*eps-1)*rhop-2*eps*Omega_K**2*tauf**2*rhog)*kx**4*ux**3+
         ((1j*Omega_K**2*tauf**3+1j*tauf)*rhog*cs**2*kx**5+((1j*eps-1j)*Omega_K**2*tauf*rhop+(1j*Omega_K**4*tauf**3+(1j*eps**2+2*1j*eps+1j)*Omega_K**2*tauf)*rhog)*kx**3)*ux**2+
         eps*Omega_K**2*tauf**2*rhog*cs**2*kx**4*ux)*wx
         +rhop*kx**4*ux**4+1j*Omega_K**2*tauf*rhop*kx**3*ux**3)/(tauf**4*rhog)
      return c0

   def _c1(self):
      Omega_K, cs, rhog, etavk, tauf, eps, kx, kz, rhop, taus, wx, ux, wy, uy = self.get_vars()
      c1 = (((2*1j*Omega_K*tauf**2*rhop*cs**2*kx**2*wx+2*1j*Omega_K*tauf**2*rhop*cs**2*kx**2*ux+(2*eps+2)*Omega_K*tauf*rhop*cs**2*kx)*wy+
         (-2*1j*Omega_K*tauf**2*rhop*cs**2*kx**2*wx-2*1j*Omega_K*tauf**2*rhop*cs**2*kx**2*ux+(-2*eps-2)*Omega_K*tauf*rhop*cs**2*kx)*uy+
         (2*tauf**4*rhog*cs**2*kx**5*ux-2*1j*eps*tauf**3*rhog*cs**2*kx**4)*wx**4+
         (4*tauf**4*rhog*cs**2*kx**5*ux**2+(-8*1j*eps-6*1j)*tauf**3*rhog*cs**2*kx**4*ux+((-4*eps**2-4*eps)*tauf**2-4*Omega_K**2*tauf**4)*rhog*cs**2*kx**3)*wx**3+
         (-9*1j*tauf**3*rhog*cs**2*kx**4*ux**2+((-12*eps-6)*tauf**2-2*Omega_K**2*tauf**4)*rhog*cs**2*kx**3*ux+((2*1j*eps+9*1j)*Omega_K**2*tauf**3+
         (3*1j*eps**2+2*1j*eps)*tauf)*rhog*cs**2*kx**2)*wx**2+((-2*Omega_K**2*tauf**4-6*tauf**2)*rhog*cs**2*kx**3*ux**2+((4*1j*eps+2*1j)*Omega_K**2*tauf**3+
         (4*1j*eps+2*1j)*tauf)*rhog*cs**2*kx**2*ux+((2*Omega_K**4*tauf**4+(2*eps**2+6*eps+6)*Omega_K**2*tauf**2)*rhog-2*Omega_K**2*tauf**2*rhop)*cs**2*kx)*wx+
         (1j*Omega_K**2*tauf**3+1j*tauf)*rhog*cs**2*kx**2*ux**2+(2*Omega_K**2*tauf**2*rhop+2*eps*Omega_K**2*tauf**2*rhog)*cs**2*kx*ux+
         ((-1j*eps**2-2*1j*eps-1j)*Omega_K**2*tauf-1j*Omega_K**4*tauf**3)*rhog*cs**2)*kz**2+
         ((-4*1j*Omega_K*tauf**2*rhop*kx**4*ux-2*eps*Omega_K*tauf*rhop*kx**3)*wx**2+
         (-10*1j*Omega_K*tauf**2*rhop*kx**4*ux**2+(-12*eps-8)*Omega_K*tauf*rhop*kx**3*ux+2*1j*Omega_K*tauf**2*rhop*cs**2*kx**4+(2*1j*eps**2+2*1j*eps)*Omega_K*rhop*kx**2)*wx-
         2*1j*Omega_K*tauf**2*rhop*kx**4*ux**3+(-4*eps-10)*Omega_K*tauf*rhop*kx**3*ux**2+
         (2*1j*Omega_K*tauf**2*rhop*cs**2*kx**4+(2*1j*eps**2+6*1j*eps+4*1j)*Omega_K*rhop*kx**2)*ux+(2*eps+2)*Omega_K*tauf*rhop*cs**2*kx**3)*wy+
         ((4*1j*Omega_K*tauf**2*rhop*kx**4*ux+2*eps*Omega_K*tauf*rhop*kx**3)*wx**2+(10*1j*Omega_K*tauf**2*rhop*kx**4*ux**2+(12*eps+8)*Omega_K*tauf*rhop*kx**3*ux-
         2*1j*Omega_K*tauf**2*rhop*cs**2*kx**4+(-2*1j*eps**2-2*1j*eps)*Omega_K*rhop*kx**2)*wx+2*1j*Omega_K*tauf**2*rhop*kx**4*ux**3+(4*eps+10)*Omega_K*tauf*rhop*kx**3*ux**2+
         ((-2*1j*eps**2-6*1j*eps-4*1j)*Omega_K*rhop*kx**2-2*1j*Omega_K*tauf**2*rhop*cs**2*kx**4)*ux+(-2*eps-2)*Omega_K*tauf*rhop*cs**2*kx**3)*uy+
         (-4*tauf**4*rhog*kx**7*ux**3+9*1j*eps*tauf**3*rhog*kx**6*ux**2+(2*tauf**4*rhog*cs**2*kx**7+(2*Omega_K**2*tauf**4+6*eps**2*tauf**2)*rhog*kx**5)*ux-
         2*1j*eps*tauf**3*rhog*cs**2*kx**6+(-1j*eps*Omega_K**2*tauf**3-1j*eps**3*tauf)*rhog*kx**4)*wx**4+(-4*tauf**4*rhog*kx**7*ux**4+(12*1j*eps+12*1j)*tauf**3*rhog*kx**6*ux**3+
         (4*tauf**4*rhog*cs**2*kx**7+(3*tauf**2*rhop+(4*Omega_K**2*tauf**4+(12*eps**2+18*eps)*tauf**2)*rhog)*kx**5)*ux**2+((-8*1j*eps-6*1j)*tauf**3*rhog*cs**2*kx**6+
         (((-4*1j*eps-6*1j)*Omega_K**2*tauf**3+(-4*1j*eps**3-6*1j*eps**2)*tauf)*rhog-4*1j*eps*tauf*rhop)*kx**4)*ux+(-4*eps**2-4*eps)*tauf**2*rhog*cs**2*kx**5+
         (-eps**2*rhop-2*eps*Omega_K**2*tauf**2*rhog)*kx**3)*wx**3+(9*1j*tauf**3*rhog*kx**6*ux**4+((4*Omega_K**2*tauf**4+(18*eps+12)*tauf**2)*rhog-tauf**2*rhop)*kx**5*ux**3+
         ((((-9*1j*eps-9*1j)*Omega_K**2*tauf**3+(-9*1j*eps**2-9*1j*eps)*tauf)*rhog-6*1j*tauf*rhop)*kx**4-9*1j*tauf**3*rhog*cs**2*kx**6)*ux**2+
         (((-12*eps-6)*tauf**2-2*Omega_K**2*tauf**4)*rhog*cs**2*kx**5+((2*Omega_K**2*tauf**2-eps**2-4*eps)*rhop+((-6*eps**2-10*eps-6)*Omega_K**2*tauf**2-
         2*Omega_K**4*tauf**4)*rhog)*kx**3)*ux+(2*1j*eps*Omega_K**2*tauf**3+(3*1j*eps**2+2*1j*eps)*tauf)*rhog*cs**2*kx**4+((1j*eps*Omega_K**4*tauf**3+
         (1j*eps**3+2*1j*eps**2+1j*eps)*Omega_K**2*tauf)*rhog-1j*eps*Omega_K**2*tauf*rhop)*kx**2)*wx**2+(((2*Omega_K**2*tauf**4+6*tauf**2)*rhog-
         2*tauf**2*rhop)*kx**5*ux**4+((4*1j*eps+4*1j)*tauf*rhop+((-6*1j*eps-4*1j)*Omega_K**2*tauf**3+(-6*1j*eps-4*1j)*tauf)*rhog)*kx**4*ux**3+
         ((-2*Omega_K**2*tauf**4-6*tauf**2)*rhog*cs**2*kx**5+((-Omega_K**2*tauf**2+2*eps**2+2*eps-3)*rhop+((-6*eps**2-10*eps-6)*Omega_K**2*tauf**2-
         2*Omega_K**4*tauf**4)*rhog)*kx**3)*ux**2+(((4*1j*eps+2*1j)*Omega_K**2*tauf**3+(4*1j*eps+2*1j)*tauf)*rhog*cs**2*kx**4+(((2*1j*eps+2*1j)*Omega_K**4*tauf**3+
         (2*1j*eps**3+6*1j*eps**2+6*1j*eps+2*1j)*Omega_K**2*tauf)*rhog-2*1j*Omega_K**2*tauf*rhop)*kx**2)*ux+(2*eps**2+eps)*Omega_K**2*tauf**2*rhog*cs**2*kx**3)*wx+
         (2*1j*tauf*rhop+(-1j*Omega_K**2*tauf**3-1j*tauf)*rhog)*kx**4*ux**4+((-Omega_K**2*tauf**2+2*eps+3)*rhop-2*eps*Omega_K**2*tauf**2*rhog)*kx**3*ux**3+
         ((1j*Omega_K**2*tauf**3+1j*tauf)*rhog*cs**2*kx**4+((1j*eps+2*1j)*Omega_K**2*tauf*rhop+(1j*Omega_K**4*tauf**3+
         (1j*eps**2+2*1j*eps+1j)*Omega_K**2*tauf)*rhog)*kx**2)*ux**2+eps*Omega_K**2*tauf**2*rhog*cs**2*kx**3*ux)/(tauf**4*rhog)
      return c1

   def _c2(self):
      Omega_K, cs, rhog, etavk, tauf, eps, kx, kz, rhop, taus, wx, ux, wy, uy = self.get_vars()
      c2 = -((2*1j*Omega_K*tauf**2*rhop*cs**2*kx*wy-2*1j*Omega_K*tauf**2*rhop*cs**2*kx*uy+tauf**4*rhog*cs**2*kx**4*wx**4+(8*tauf**4*rhog*cs**2*kx**4*ux+
         (-8*1j*eps-3*1j)*tauf**3*rhog*cs**2*kx**3)*wx**3+(6*tauf**4*rhog*cs**2*kx**4*ux**2+(-12*1j*eps-18*1j)*tauf**3*rhog*cs**2*kx**3*ux+
         ((-6*eps**2-12*eps-3)*tauf**2-7*Omega_K**2*tauf**4)*rhog*cs**2*kx**2)*wx**2+(-9*1j*tauf**3*rhog*cs**2*kx**3*ux**2+((-12*eps-12)*tauf**2-
         4*Omega_K**2*tauf**4)*rhog*cs**2*kx**2*ux+((4*1j*eps+10*1j)*Omega_K**2*tauf**3+(3*1j*eps**2+4*1j*eps+1j)*tauf)*rhog*cs**2*kx)*wx+(-Omega_K**2*tauf**4-
         3*tauf**2)*rhog*cs**2*kx**2*ux**2+((2*1j*eps+2*1j)*Omega_K**2*tauf**3+(2*1j*eps+2*1j)*tauf)*rhog*cs**2*kx*ux+
         (Omega_K**4*tauf**4+(eps**2+4*eps+3)*Omega_K**2*tauf**2)*rhog*cs**2)*kz**2+(-2*1j*Omega_K*tauf**2*rhop*kx**3*wx**2+((-8*eps-4)*Omega_K*tauf*rhop*kx**2-
         14*1j*Omega_K*tauf**2*rhop*kx**3*ux)*wx-8*1j*Omega_K*tauf**2*rhop*kx**3*ux**2+(-10*eps-14)*Omega_K*tauf*rhop*kx**2*ux+2*1j*Omega_K*tauf**2*rhop*cs**2*kx**3+
         (2*1j*eps**2+4*1j*eps+2*1j)*Omega_K*rhop*kx)*wy+(2*1j*Omega_K*tauf**2*rhop*kx**3*wx**2+(14*1j*Omega_K*tauf**2*rhop*kx**3*ux+
         (8*eps+4)*Omega_K*tauf*rhop*kx**2)*wx+8*1j*Omega_K*tauf**2*rhop*kx**3*ux**2+(10*eps+14)*Omega_K*tauf*rhop*kx**2*ux-2*1j*Omega_K*tauf**2*rhop*cs**2*kx**3+
         (-2*1j*eps**2-4*1j*eps-2*1j)*Omega_K*rhop*kx)*uy+(-6*tauf**4*rhog*kx**6*ux**2+9*1j*eps*tauf**3*rhog*kx**5*ux+tauf**4*rhog*cs**2*kx**6+(Omega_K**2*tauf**4+
         3*eps**2*tauf**2)*rhog*kx**4)*wx**4+(-16*tauf**4*rhog*kx**6*ux**3+(36*1j*eps+18*1j)*tauf**3*rhog*kx**5*ux**2+(8*tauf**4*rhog*cs**2*kx**6+
         (3*tauf**2*rhop+(8*Omega_K**2*tauf**4+(24*eps**2+18*eps)*tauf**2)*rhog)*kx**4)*ux+(-8*1j*eps-3*1j)*tauf**3*rhog*cs**2*kx**5+(((-4*1j*eps-3*1j)*Omega_K**2*tauf**3+
         (-4*1j*eps**3-3*1j*eps**2)*tauf)*rhog-2*1j*eps*tauf*rhop)*kx**3)*wx**3+(-6*tauf**4*rhog*kx**6*ux**4+
         (18*1j*eps+36*1j)*tauf**3*rhog*kx**5*ux**3+(6*tauf**4*rhog*cs**2*kx**6+(3*tauf**2*rhop+(12*Omega_K**2*tauf**4+(18*eps**2+54*eps+18)*tauf**2)*rhog)*kx**4)*ux**2+
         ((-12*1j*eps-18*1j)*tauf**3*rhog*cs**2*kx**5+((-6*1j*eps-6*1j)*tauf*rhop+((-15*1j*eps-18*1j)*Omega_K**2*tauf**3+(-6*1j*eps**3-18*1j*eps**2-9*1j*eps)*tauf)*rhog)*kx**3)*ux+
         ((-6*eps**2-12*eps-3)*tauf**2-Omega_K**2*tauf**4)*rhog*cs**2*kx**4+((Omega_K**2*tauf**2-2*eps**2-2*eps)*rhop+((-3*eps**2-8*eps-3)*Omega_K**2*tauf**2-
         Omega_K**4*tauf**4)*rhog)*kx**2)*wx**2+(9*1j*tauf**3*rhog*kx**5*ux**4+((8*Omega_K**2*tauf**4+(18*eps+24)*tauf**2)*rhog-5*tauf**2*rhop)*kx**4*ux**3+
         ((6*1j*eps*tauf*rhop+((-18*1j*eps-15*1j)*Omega_K**2*tauf**3+(-9*1j*eps**2-18*1j*eps-6*1j)*tauf)*rhog)*kx**3-9*1j*tauf**3*rhog*cs**2*kx**5)*ux**2+
         (((-12*eps-12)*tauf**2-4*Omega_K**2*tauf**4)*rhog*cs**2*kx**4+((Omega_K**2*tauf**2+eps**2-2*eps-3)*rhop+((-12*eps**2-20*eps-12)*Omega_K**2*tauf**2-
         4*Omega_K**4*tauf**4)*rhog)*kx**2)*ux+
         ((4*1j*eps+1j)*Omega_K**2*tauf**3+(3*1j*eps**2+4*1j*eps+1j)*tauf)*rhog*cs**2*kx**3+((-1j*eps-1j)*Omega_K**2*tauf*rhop+((2*1j*eps+1j)*Omega_K**4*tauf**3+
         (2*1j*eps**3+5*1j*eps**2+4*1j*eps+1j)*Omega_K**2*tauf)*rhog)*kx)*wx+((Omega_K**2*tauf**4+3*tauf**2)*rhog-tauf**2*rhop)*kx**4*ux**4+
         ((2*1j*eps+6*1j)*tauf*rhop+((-3*1j*eps-4*1j)*Omega_K**2*tauf**3+(-3*1j*eps-4*1j)*tauf)*rhog)*kx**3*ux**3+((-Omega_K**2*tauf**4-3*tauf**2)*rhog*cs**2*kx**4+
         ((-2*Omega_K**2*tauf**2+eps**2+4*eps+3)*rhop+((-3*eps**2-8*eps-3)*Omega_K**2*tauf**2-Omega_K**4*tauf**4)*rhog)*kx**2)*ux**2+
         (((2*1j*eps+2*1j)*Omega_K**2*tauf**3+(2*1j*eps+2*1j)*tauf)*rhog*cs**2*kx**3+((1j*eps+1j)*Omega_K**2*tauf*rhop+((1j*eps+2*1j)*Omega_K**4*tauf**3+
         (1j*eps**3+4*1j*eps**2+5*1j*eps+2*1j)*Omega_K**2*tauf)*rhog)*kx)*ux+(eps**2+eps)*Omega_K**2*tauf**2*rhog*cs**2*kx**2)/(tauf**4*rhog) 
      return c2

   def _c3(self):
      Omega_K, cs, rhog, etavk, tauf, eps, kx, kz, rhop, taus, wx, ux, wy, uy = self.get_vars()
      c3 = ((4*tauf**4*rhog*cs**2*kx**3*wx**3+(12*tauf**4*rhog*cs**2*kx**3*ux+(-12*1j*eps-9*1j)*tauf**3*rhog*cs**2*kx**2)*wx**2+
         (4*tauf**4*rhog*cs**2*kx**3*ux**2+(-8*1j*eps-18*1j)*tauf**3*rhog*cs**2*kx**2*ux+((-4*eps**2-12*eps-6)*tauf**2-6*Omega_K**2*tauf**4)*rhog*cs**2*kx)*wx-
         3*1j*tauf**3*rhog*cs**2*kx**2*ux**2+((-4*eps-6)*tauf**2-2*Omega_K**2*tauf**4)*rhog*cs**2*kx*ux+((2*1j*eps+4*1j)*Omega_K**2*tauf**3+
         (1j*eps**2+2*1j*eps+1j)*tauf)*rhog*cs**2)*kz**2+(-6*1j*Omega_K*tauf**2*rhop*kx**2*wx-10*1j*Omega_K*tauf**2*rhop*kx**2*ux+(-6*eps-6)*Omega_K*tauf*rhop*kx)*wy+
         (6*1j*Omega_K*tauf**2*rhop*kx**2*wx+10*1j*Omega_K*tauf**2*rhop*kx**2*ux+(6*eps+6)*Omega_K*tauf*rhop*kx)*uy+(3*1j*eps*tauf**3*rhog*kx**4-4*tauf**4*rhog*kx**5*ux)*wx**4+
         (-24*tauf**4*rhog*kx**5*ux**2+(36*1j*eps+12*1j)*tauf**3*rhog*kx**4*ux+4*tauf**4*rhog*cs**2*kx**5+(tauf**2*rhop+(4*Omega_K**2*tauf**4+
         (12*eps**2+6*eps)*tauf**2)*rhog)*kx**3)*wx**3+(-24*tauf**4*rhog*kx**5*ux**3+(54*1j*eps+54*1j)*tauf**3*rhog*kx**4*ux**2+(12*tauf**4*rhog*cs**2*kx**5+
         (5*tauf**2*rhop+(16*Omega_K**2*tauf**4+(36*eps**2+54*eps+12)*tauf**2)*rhog)*kx**3)*ux+
         (-12*1j*eps-9*1j)*tauf**3*rhog*cs**2*kx**4+((-4*1j*eps-2*1j)*tauf*rhop+((-9*1j*eps-9*1j)*Omega_K**2*tauf**3+(-6*1j*eps**3-9*1j*eps**2-3*1j*eps)*tauf)*rhog)*kx**2)*wx**2+
         (-4*tauf**4*rhog*kx**5*ux**4+(12*1j*eps+36*1j)*tauf**3*rhog*kx**4*ux**3+
         (4*tauf**4*rhog*cs**2*kx**5+((16*Omega_K**2*tauf**4+(12*eps**2+54*eps+36)*tauf**2)*rhog-3*tauf**2*rhop)*kx**3)*ux**2+((-8*1j*eps-18*1j)*tauf**3*rhog*cs**2*kx**4+
         (((-22*1j*eps-22*1j)*Omega_K**2*tauf**3+(-4*1j*eps**3-18*1j*eps**2-18*1j*eps-4*1j)*tauf)*rhog-4*1j*tauf*rhop)*kx**2)*ux+((-4*eps**2-12*eps-6)*tauf**2-
         2*Omega_K**2*tauf**4)*rhog*cs**2*kx**3+((Omega_K**2*tauf**2-eps**2-2*eps-1)*rhop+((-6*eps**2-12*eps-6)*Omega_K**2*tauf**2-2*Omega_K**4*tauf**4)*rhog)*kx)*wx+
         3*1j*tauf**3*rhog*kx**4*ux**4+((4*Omega_K**2*tauf**4+(6*eps+12)*tauf**2)*rhog-3*tauf**2*rhop)*kx**3*ux**3+
         (((4*1j*eps+6*1j)*tauf*rhop+((-9*1j*eps-9*1j)*Omega_K**2*tauf**3+(-3*1j*eps**2-9*1j*eps-6*1j)*tauf)*rhog)*kx**2-3*1j*tauf**3*rhog*cs**2*kx**4)*ux**2+
         (((-4*eps-6)*tauf**2-2*Omega_K**2*tauf**4)*rhog*cs**2*kx**3+((-Omega_K**2*tauf**2+eps**2+2*eps+1)*rhop+((-6*eps**2-12*eps-6)*Omega_K**2*tauf**2-2*Omega_K**4*tauf**4)*rhog)*kx)*ux+
         ((2*1j*eps+1j)*Omega_K**2*tauf**3+(1j*eps**2+2*1j*eps+1j)*tauf)*rhog*cs**2*kx**2+((1j*eps+1j)*Omega_K**4*tauf**3+
         (1j*eps**3+3*1j*eps**2+3*1j*eps+1j)*Omega_K**2*tauf)*rhog)/(tauf**4*rhog)
      return c3

   def _c4(self):
      Omega_K, cs, rhog, etavk, tauf, eps, kx, kz, rhop, taus, wx, ux, wy, uy = self.get_vars()
      c4 = -((6*tauf**3*rhog*cs**2*kx**2*wx**2+(8*tauf**3*rhog*cs**2*kx**2*ux+(-8*1j*eps-9*1j)*tauf**2*rhog*cs**2*kx)*wx+tauf**3*rhog*cs**2*kx**2*ux**2+
         (-2*1j*eps-6*1j)*tauf**2*rhog*cs**2*kx*ux+((-eps**2-4*eps-3)*tauf-2*Omega_K**2*tauf**3)*rhog*cs**2)*kz**2-4*1j*Omega_K*tauf*rhop*kx*wy+
         4*1j*Omega_K*tauf*rhop*kx*uy-tauf**3*rhog*kx**4*wx**4+((12*1j*eps+3*1j)*tauf**2*rhog*kx**3-16*tauf**3*rhog*kx**4*ux)*wx**3+
         (-36*tauf**3*rhog*kx**4*ux**2+(54*1j*eps+36*1j)*tauf**2*rhog*kx**3*ux+6*tauf**3*rhog*cs**2*kx**4+(2*tauf*rhop+(7*Omega_K**2*tauf**3+(18*eps**2+18*eps+3)*tauf)*rhog)*kx**2)*wx**2+
         (-16*tauf**3*rhog*kx**4*ux**3+(36*1j*eps+54*1j)*tauf**2*rhog*kx**3*ux**2+(8*tauf**3*rhog*cs**2*kx**4+(tauf*rhop+(16*Omega_K**2*tauf**3+
         (24*eps**2+54*eps+24)*tauf)*rhog)*kx**2)*ux+(-8*1j*eps-9*1j)*tauf**2*rhog*cs**2*kx**3+((-2*1j*eps-2*1j)*rhop+((-10*1j*eps-10*1j)*Omega_K**2*tauf**2-
         4*1j*eps**3-9*1j*eps**2-6*1j*eps-1j)*rhog)*kx)*wx-tauf**3*rhog*kx**4*ux**4+(3*1j*eps+12*1j)*tauf**2*rhog*kx**3*ux**3+(tauf**3*rhog*cs**2*kx**4+
         ((7*Omega_K**2*tauf**3+(3*eps**2+18*eps+18)*tauf)*rhog-3*tauf*rhop)*kx**2)*ux**2+((-2*1j*eps-6*1j)*tauf**2*rhog*cs**2*kx**3+((2*1j*eps+2*1j)*rhop+
         ((-10*1j*eps-10*1j)*Omega_K**2*tauf**2-1j*eps**3-6*1j*eps**2-9*1j*eps-4*1j)*rhog)*kx)*ux+
         ((-eps**2-4*eps-3)*tauf-Omega_K**2*tauf**3)*rhog*cs**2*kx**2+((-3*eps**2-6*eps-3)*Omega_K**2*tauf-Omega_K**4*tauf**3)*rhog)/(tauf**3*rhog)
      return c4

   def _c5(self):
      Omega_K, cs, rhog, etavk, tauf, eps, kx, kz, rhop, taus, wx, ux, wy, uy = self.get_vars()
      c5 = ((4*tauf**3*rhog*cs**2*kx*wx+2*tauf**3*rhog*cs**2*kx*ux+(-2*1j*eps-3*1j)*tauf**2*rhog*cs**2)*kz**2-
         4*tauf**3*rhog*kx**3*wx**3+((18*1j*eps+9*1j)*tauf**2*rhog*kx**2-24*tauf**3*rhog*kx**3*ux)*wx**2+
         (-24*tauf**3*rhog*kx**3*ux**2+(36*1j*eps+36*1j)*tauf**2*rhog*kx**2*ux+4*tauf**3*rhog*cs**2*kx**3+(tauf*rhop+(6*Omega_K**2*tauf**3+(12*eps**2+18*eps+6)*tauf)*rhog)*kx)*wx-
         4*tauf**3*rhog*kx**3*ux**3+(9*1j*eps+18*1j)*tauf**2*rhog*kx**2*ux**2+(2*tauf**3*rhog*cs**2*kx**3+((6*Omega_K**2*tauf**3+(6*eps**2+18*eps+12)*tauf)*rhog-tauf*rhop)*kx)*ux+
         (-2*1j*eps-3*1j)*tauf**2*rhog*cs**2*kx**2+((-4*1j*eps-4*1j)*Omega_K**2*tauf**2-1j*eps**3-3*1j*eps**2-3*1j*eps-1j)*rhog)/(tauf**3*rhog)
      return c5

   def _c6(self):
      Omega_K, cs, rhog, etavk, tauf, eps, kx, kz, rhop, taus, wx, ux, wy, uy = self.get_vars()
      c6 = -(tauf**2*cs**2*kz**2-6*tauf**2*kx**2*wx**2+((12*1j*eps+9*1j)*tauf*kx-16*tauf**2*kx**2*ux)*wx-6*tauf**2*kx**2*ux**2+(9*1j*eps+12*1j)*tauf*kx*ux+
         tauf**2*cs**2*kx**2+2*Omega_K**2*tauf**2+3*eps**2+6*eps+3)/tauf**2
      return c6

   def _c7(self):
      Omega_K, cs, rhog, etavk, tauf, eps, kx, kz, rhop, taus, wx, ux, wy, uy = self.get_vars()
      c7 = -(4.*tauf*kx*wx+4.*tauf*kx*ux-3.*1j*eps-3.*1j)/tauf
      return c7

   def _c8(self):
      return mp.mpf(1.0)

   def solve_eq(self):
      c = [self._c8(),self._c7(),self._c6(),self._c5(),self._c4(),self._c3(),self._c2(),self._c1(),self._c0()]
      self.omega_tab = mp.polyroots(c)
      self.omega = sorted(self.omega_tab,key=lambda x: x.imag,reverse=True)[0]
      return sorted(self.omega_tab,key=lambda x: x.imag,reverse=True)

   def get_eigenvalues(self):
      # since for given 'omega' |A|=0, we remove one row and one column by fixing perturbation amplitude of dust density
      # then we're able to express other amplitudes as functions of amp_rp
      amp_rp = np.float64(1.0)

      # array A with first row and first column removed
      B=np.array([[-1j*(self.omega-self.kx*self.wx) + 1/self.tauf, -2*self.Omega_K, 0, 0, -1/self.tauf, 0 ,0 ],
             [0.5*self.Omega_K, -1j*(self.omega-self.kx*self.wx) + 1/self.tauf, 0 ,0, 0, -1/self.tauf, 0],
             [0, 0, -1j*(self.omega-self.kx*self.wx) + 1/self.tauf, 0 ,0, 0,   -1/self.tauf],
             [0, 0, 0, -1j*(self.omega-self.kx*self.ux), 1j*self.kx*self.rhog, 0, 1j*self.kz*self.rhog],
             [-self.eps/self.tauf, 0, 0, self.cs**2/self.rhog*1j*self.kx, -1j*(self.omega-self.kx*self.ux)+self.eps/self.tauf, -2*self.Omega_K, 0],
             [0, -self.eps/self.tauf, 0, 0, 0.5*self.Omega_K, -1j*(self.omega-self.kx*self.ux)+self.eps/self.tauf, 0],
             [0, 0, -self.eps/self.tauf, self.cs**2/self.rhog*1j*self.kz, 0, 0, -1j*(self.omega-self.kx*self.ux)+self.eps/self.tauf]],dtype=np.complex128)

      # some expressions from removed column appear as constants on the r.h.s of A.x = b
      RHS = np.array( [0,0,0,0,-(self.ux-self.wx)/(self.tauf*self.rhog)*amp_rp,-(self.uy-self.wy)/(self.tauf*self.rhog)*amp_rp,0], dtype=np.complex128)

      # we solve B.x = rhs by finding inverse of B, x = B^{-1}.rhs
      invB = np.linalg.inv(B)
      return np.dot(invB,RHS)    # amplitudes [ wx, wy, wz, rg, ux, uy, uz ]

if __name__ == "__main__":
   linA = dispersion_eq(Omega_K=mp.mpf('1.0'),cs=mp.mpf('1.0'),rhog=mp.mpf('1.0'),etavk=mp.mpf('0.05'),
         tauf=mp.mpf('0.1'),eps=mp.mpf('3.0'),kx=mp.mpf('30.0'),kz=mp.mpf('30.0'))
   linB = dispersion_eq(Omega_K=mp.mpf('1.0'),cs=mp.mpf('1.0'),rhog=mp.mpf('1.0'),etavk=mp.mpf('0.05'),
         tauf=mp.mpf('0.1'),eps=mp.mpf('0.2'),kx=mp.mpf('6.0'),kz=mp.mpf('6.0'))

# reference from Y&J07 (linA)
   w0 = mp.mpc(0.3480127,0.4190204)
   print "LinA omega=", linA.solve_eq()[0] , w0, abs(linA.omega - w0)
   ux0 = np.complex128(-0.1691398+0.0361553*1j)
   uy0 = np.complex128(+0.1336704+0.0591695*1j)
   uz0 = np.complex128(+0.1691389-0.0361555*1j)
   rg0 = np.complex128(+0.0000224+0.0000212*1j)
   wx0 = np.complex128(-0.1398623+0.0372951*1j)
   wy0 = np.complex128(+0.1305628+0.0640574*1j)
   wz0 = np.complex128(+0.1639549-0.0233277*1j)

   x = linA.get_eigenvalues()
   etavk = linA.etavk
   eps = linA.eps
   mean_rhog = linA.rhog
# Verification of results (notice normalization)
   print "ux = ", x[4]*(eps/etavk), ux0, abs(x[4]*(eps/etavk) - ux0)
   print "uy = ", x[5]*(eps/etavk), uy0, abs(x[5]*(eps/etavk) - uy0)
   print "uz = ", x[6]*(eps/etavk), uz0, abs(x[6]*(eps/etavk) - uz0)
   print "rg0= ", x[3]*(eps/mean_rhog),  rg0, abs(x[3]*(eps/mean_rhog) - rg0)
   print "wx = ", x[0]*(eps/etavk), wx0, abs(x[0]*(eps/etavk) - wx0)
   print "wy = ", x[1]*(eps/etavk), wy0, abs(x[1]*(eps/etavk) - wy0)
   print "wz = ", x[2]*(eps/etavk), wz0, abs(x[2]*(eps/etavk) - wz0)

# reference from Y&J07 (linB)
   w0 = mp.mpc(0.4998786,0.0154764)
   print "LinB omega=", linB.solve_eq()[0] , w0, abs(linB.omega - w0)
   ux0 = np.complex128(-0.0174121-0.2770347*1j)
   uy0 = np.complex128(+0.2767976-0.0187568*1j)
   uz0 = np.complex128(+0.0174130+0.2770423*1j)
   rg0 = np.complex128(-0.0000067-0.0000691*1j)
   wx0 = np.complex128(+0.0462916-0.2743072*1j)
   wy0 = np.complex128(+0.2739304+0.0039293*1j)
   wz0 = np.complex128(+0.0083263+0.2768866*1j)

   x = linB.get_eigenvalues()
   etavk = linB.etavk
   eps = linB.eps
   mean_rhog = linB.rhog
# Verification of results (notice normalization)
   print "ux = ", x[4]*(eps/etavk), ux0, abs(x[4]*(eps/etavk) - ux0)
   print "uy = ", x[5]*(eps/etavk), uy0, abs(x[5]*(eps/etavk) - uy0)
   print "uz = ", x[6]*(eps/etavk), uz0, abs(x[6]*(eps/etavk) - uz0)
   print "rg0= ", x[3]*(eps/mean_rhog),  rg0, abs(x[3]*(eps/mean_rhog) - rg0)
   print "wx = ", x[0]*(eps/etavk), wx0, abs(x[0]*(eps/etavk) - wx0)
   print "wy = ", x[1]*(eps/etavk), wy0, abs(x[1]*(eps/etavk) - wy0)
   print "wz = ", x[2]*(eps/etavk), wz0, abs(x[2]*(eps/etavk) - wz0)

