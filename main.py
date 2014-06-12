__author__ = 'mpopovic'

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from operator import mul
from scipy.fftpack import fft, ifft
from scipy.linalg import solve
from numpy import matrix
from numpy import linalg
from scipy.interpolate import BarycentricInterpolator as BI

def fill_zeros(s,k):
    while len(s)<k:
        s='0'+s
    return s

def fcglt(A): # Modal Coefficients to Lobatto Nodal
    """
    Fast Chebyshev-Gauss-Lobatto transformation from
    Chebyshev expansion coefficients (modal) to point
    space values (nodal). If I=numpy.identity(n), then
    T=chebyshev.fcglt(I) will be the Chebyshev
    Vandermonde matrix on the Lobatto nodes
    """
    size = A.shape
    m = size[0]
    k = m-2-np.arange(m-2)
    if len(size) == 2: # Multiple vectors
        V = np.vstack((2*A[0,:],A[1:m-1,:],2*A[m-1,:],A[k,:]))
        F = fft(V, n=None, axis=0)
        B = 0.5*F[0:m,:]
    else:  # Single vector
        V = np.hstack((2*A[0],A[1:m-1],2*A[m-1],A[k]))
        F = fft(V, n=None)
        B = 0.5*F[0:m]
    if A.dtype!='complex':
        return np.real(B)
    else:
        return B

def ifcglt(A): # Lobatto Nodal to Modal Coefficients
    """
    Fast Chebyshev-Gauss-Lobatto transformation from
    point space values (nodal) to Chebyshev expansion
    coefficients (modal). If I=numpy.identity(n), then
    Ti=chebyshev.ifcglt(I) will be the inverse of the
    Chebyshev Vandermonde matrix on the Lobatto nodes
    """
    size = A.shape
    m = size[0]
    k = m-1-np.arange(m-1)
    if len(size) == 2: # Multiple vectors
        V = np.vstack((A[0:m-1,:],A[k,:]))
        F = ifft(V, n=None, axis=0)
        B = np.vstack((F[0,:],2*F[1:m-1,:],F[m-1,:]))
    else:  # Single vector
        V = np.hstack((A[0:m-1],A[k]))
        F = ifft(V, n=None)
        B = np.hstack((F[0],2*F[1:m-1],F[m-1]))
    if A.dtype!='complex':
        return np.real(B)
    else:
        return B


def diffmat(x):
    n= sp.size(x)
    e= sp.ones((n,1))
    Xdiff= sp.outer(x,e)-sp.outer(e,x)+sp.identity(n)
    xprod= -reduce(mul, Xdiff)
    W= sp.outer(1/xprod,e)
    D= W/sp.multiply(W.T,Xdiff)
    d= 1-sum(D)
    for k in range(0,n):
        D[k,k] = d[k]
    return -D.T

N = 20
x= -np.cos(np.pi*np.arange(N+1)/N) #collocation points

D= diffmat(x)  #derivative matrices
D2= np.dot(D,D)

dt= 0.0001       #time step
g= 1.           #diffusion constant

E= linalg.inv(np.identity(N+1)-g*dt*D2)  #evolution matrix for implicit diffusion calculation

sigma= 0.1
mu= 0.5
#f0= np.array(map(lambda z: 1/np.sqrt(20*np.pi*sigma**2)*np.exp(-(z)*(z)/(2*sigma**2)),x))
#f0= np.array(map(lambda z: 1/(1+np.exp(-(z-mu)/sigma)),x))
f0= np.array(map(lambda z: 0.*z, x))
#force= np.array(map(lambda z: (z-1)*(z+1),x))
#f0= np.array(map(lambda z: 1/(np.sqrt(2*np.pi/30))*np.exp(-(z-0.5)*(z-0.5)*30)-1/(np.sqrt(2*np.pi/30))*np.exp(-(z+0.5)*(z+0.5)*30), x))
#f0= np.array(map(lambda z: np.sin(z*np.pi), x))

f= f0
f[0]= np.sin(0)
f[-1]= -np.sin(0)

t0= 1.
mu= 0.
sigma= 0.02
sigma_t= 0.3
# boundary_a= lambda x: np.sin(x*dt*np.pi/5.)
# boundary_b= lambda x: np.sin(2*x*dt*np.pi/300.)
active= np.array(map(lambda x: 1./(1+np.exp((x-mu)/sigma)),x)+)*1./(1+np.exp(t0/sigma_t))
p= f


s= BI(x, p)
print s
rng= np.linspace(-1,1,1000)
# plt.figure()
# plt.plot(rng, map(s,rng))
# plt.show()
for i in np.arange(300000):
    # A= np.identity(N+1)-g*dt*D2
    # A[-1,:]= D[-1,:]
    # v= p - dt*p*D.dot(p)
    # v[-1]= 0.
    # p= sp.linalg.solve(A, v)
    ac_factor= np.array(map(lambda x: x*(1+np.exp(-(i*dt-t0)/sigma_t)),active))
    active_new= active+ dt*(1./sigma_t*(active-active*active/ac_factor)-1* p*D.dot(active))
    p= E.dot(p-dt*p*D.dot(p)+ 1*D.dot(active_new-active))
    if i%3000 == 0:
        print i*dt, np.min(p), np.max(active), np.max(ac_factor)
    active= active_new#/np.max(ac_factor)
    p[0]= 0.#boundary_a(i)
    p[-1]= 0.#boundary_b(i)
    if i%3000 == 0:
        f= p
        s= BI(x, f)
        sa= BI(x, active)
        rng= np.linspace(-1,1,1000)
        plt.figure()
        plt.plot(rng, map(s,rng))
        plt.plot(rng, np.array(map(sa, rng))/10.)
        plt.grid()
        plt.ylim(-0.5,0.11)
        #plt.show()
        plt.savefig('/Users/mpopovic/Documents/Work/Projects/drosophila_wing_analysis/pseudospectral_integrator/test'+ fill_zeros(str(i/3000),4)+'.png')
        plt.close()



#active= new_active
# two_P= np.zeros(2*N+1)
# P= ifcglt(p)
# two_P[:21]= P
# two_p= fcglt(two_P)
# two_nonlin= -two_p*two_D.dot(two_p)
# two_nonlin_tran= ifcglt(two_nonlin)
# nonlin_tran= two_nonlin_tran[:21]
# nonlin= fcglt(nonlin_tran)


