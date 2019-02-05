# Chebyshev-Gauss Quadrature
# Bryan Kaiser
# 12/12/17

import h5py
import numpy as np
import math as ma
import matplotlib.pyplot as plt


# =============================================================================
# functions

def cgq(u,z,Lz,Nz):
  # Chebyshev-Gauss Quadrature on a grid z = (0,Lz)
  w = np.pi/Nz # Chebyshev weights
  U = 0.
  for n in range(0,Nz-1):  
    U = w*u[Nz-1-n]*np.sqrt(1.-z[n]**2.) + U
  U = U*Lz/2.
  return U


# =============================================================================
# grid (centered at 0)

# y-dimensions:

Ly = 30.0; # m, domain size
Ny = 128; 
dy = Ly/Ny; # m, uniform longitudinal grid spacing
y = np.linspace(0.5*dy, dy*Ny-dy*0.5, num=Ny) - Ly/2.

# z-dimensions:

# Chebyshev points in z 
# (points z are the roots of Chebyshev polynomials of the first kind)
Nz = 256
k = np.linspace(1., Nz, num=Nz)
# example 1:
Lz1 = 2.0
z1 = np.cos((k*2.-1.)/(2.*Nz)*np.pi) 
# example 2:
Lz2 = 60.0
z2 = -np.cos((k*2.-1.)/(2.*Nz)*np.pi)*Lz2/2.+Lz2/2.
#print(Lz2/2.*z1[0]+Lz2/2.-z2[Nz-1]) # = 0
#print(Lz2/2.*z1[Nz-1]+Lz2/2.-z2[0]) # = 0

# a 2d grid:
Y,Z = np.meshgrid(y,z2) # shape = Nz, Ny
Y = np.transpose(Y); Z = np.transpose(Z); # shape = Ny, Nz


# =============================================================================
# test signals

# example 1: z = (-1,1)
u1 = np.power(z1,2.) # signal
U1soln = (z1[0]**3.)/3.-(z1[Nz-1]**3.)/3. # z integral of signal in (-1,1)
U1true = 2./3. # z integral of signal in [-1,1]

# example 2: z = (0,Lz)
u2 = np.power(z2,2.) # signal
U2soln = (z2[Nz-1]**3.)/3.-(z2[0]**3.)/3. # z integral of signal in (0,Lz)
U2true = (Lz2**3.)/3. # z integral of signal in [0,Lz]

# example 3: repeated 1d integration over z = (0,Lz)
u3 = np.power(Z,2.) # signal
U3soln = np.ones([Ny])*U2soln
U3true = np.ones([Ny])*U2true


# =============================================================================
# integration by Chebyshev-Gauss quadrature 
# (Chebyshev polynomials of the first kind) 

# example 1: z = (-1,1)
U1 = 0.
w = np.pi/Nz # Chebyshev weights
for n in range(0,Nz-1):
  U1 = w*u1[n]*np.sqrt(1.-z1[n]**2.) + U1
error1 = 'Example 1 error of int z^2 dz for z = (-1,1) = %.8f' %(abs(U1-U1soln)/abs(U1soln))
print(error1)

# example 2: z = (0,Lz)
U2 = cgq(u2,z1,Lz2,Nz)
error2 = 'Example 2 error of int z^2 dz for z = (0,Lz) = %.8f' %(abs(U2-U2soln)/abs(U2soln))
print(error2)
"""
U2 = 0.
for n in range(0,Nz-1):  
  U2 = w*u2[Nz-1-n]*np.sqrt(1.-z1[n]**2.) + U2
U2 = U2*Lz2/2.
"""

# example 3: z = (0,Lz) repeated
U3 = np.zeros([Ny])
for j in range(0,Ny):
  U3[j] = cgq(u3[j,:],z1,Lz2,Nz)
U3m = np.mean(U3,axis=0)
error3 = 'Example 3 error of int z^2 dz for z = (0,Lz) = %.8f' %(abs(U3m-U2soln)/abs(U2soln))
print(error3)
"""
for n in range(0,Nz-1):
  U3 = w*u3[:,Nz-1-n]*np.sqrt(1.-z1[n]**2.) + U3
U3 = U3*Lz2/2.
"""

# errors relative to the intervals including end points:
error4 = 'Example 1 error of int z^2 dz for z = [-1,1] = %.8f' %(abs(U1-U1true)/abs(U1true))
print(error4)
error5 = 'Example 2 error of int z^2 dz for z = [0,Lz] = %.8f' %(abs(U2-U2true)/abs(U2true))
print(error5)
error6 = 'Example 3 error of int z^2 dz for z = [0,Lz] = %.8f' %(abs(U3m-U2true)/abs(U2true))
print(error6)
