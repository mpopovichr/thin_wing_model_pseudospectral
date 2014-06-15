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

N= 20
x= -np.cos(np.pi*np.arange(N+1)/N) #collocation points

D= diffmat(x)  #derivative matrices
D2= np.dot(D,D)

dt= 0.00001       #time step

L= 0.4      #HB interface
tau= 1.9    #
tauT1= 3.9 #
h0= 0.3  #relative to total length of wing
k= 30.
K1= 2.
K2= 4.
Gamma= 300.
lambda1= 1.
lambda2_h= 0.
lambda2_b= -.07
zeta_h= 0.
zeta_b= 0.
zetabar_h= 5.
zetabar_b=0.

Q_plus= np.zeros(N+1)
Q_minus= np.zeros(N+1)
v_x= np.zeros(N+1)
v_yy= np.zeros(N+1)
h= np.zeros(N+1)+h0
T_minus= np.zeros(N+1)
sigma_xx= np.zeros(N+1)
sigma_yy= np.zeros(N+1)


t0= 3.
mu= 0.
sigma= 0.02
sigma_t= 0.5
# boundary_a= lambda x: np.sin(x*dt*np.pi/5.)
# boundary_b= lambda x: np.sin(2*x*dt*np.pi/300.)
active= np.array(map(lambda x: 1./(1+np.exp((x-mu)/sigma)),x))*1./(1+np.exp(t0/sigma_t))

rng= np.linspace(-1,1,1000)

for i in np.arange(2000000):
    zeta= zeta_h*active + zeta_b
    zetabar= zetabar_h*active + zetabar_b
    lambda2= lambda2_h*active + lambda2_b
    ac_factor= np.array(map(lambda x: x*(1+np.exp(-(i*dt-t0)/sigma_t)),active))
    active_new= active+ dt*(1./sigma_t*(active-active*active/ac_factor)-1* v_x*D.dot(active))
    dxvx= D.dot(v_x)
    Q_plus= Q_plus + dt*(dxvx + v_yy-v_x*D.dot(Q_plus))
    Q_minus= Q_minus + dt*(dxvx - v_yy - T_minus-v_x*D.dot(Q_minus))
    T_minus = T_minus + dt/tauT1*(-T_minus+lambda1*0.5*(sigma_xx-sigma_yy)+lambda2*active-v_x*D.dot(T_minus))
    sigma_xx= K1*Q_minus + K2*Q_plus + zeta + zetabar
    sigma_yy= -1*K1*Q_minus + K2*Q_plus - zeta + zetabar
    H= np.log(h/h0)
    v_x= 1./Gamma*(sigma_xx*D.dot(H) + D.dot(sigma_xx))
    h_new= sigma_yy/k + h0
    v_yy= 1./h*(h_new-h)/dt+1./h*v_x*D.dot(h)
    h= h_new
    if i%10000 == 0:
        print i*dt, np.min(v_x), np.max(active), np.max(ac_factor), np.min(h), np.min(T_minus), np.max(T_minus)
    active= active_new#/np.max(ac_factor)
    v_x[0]= 0.#boundary_a(i)
    v_x[-1]= 0.#boundary_b(i)
    if i%10000 == 0:
        s= BI(x, v_x)
        sa= BI(x, active)
        sh= BI(x, h)
        sQ= BI(x,Q_minus)
        sT= BI(x,T_minus)
        rng= np.linspace(-1,1,1000)
        plt.figure()
        plt.plot(rng, map(s,rng))
        plt.plot(rng, np.array(map(sa, rng))/10.)
        plt.plot(rng, np.array(map(sh, rng))/10.)
        plt.plot(rng, np.array(map(sQ,rng))/10.)
        plt.plot(rng, np.array(map(sT, rng))/10.)
        plt.grid()
        plt.ylim(-0.5,0.11)
        #plt.show()
        plt.savefig('/home/mpopovic/Documents/Work/Projects/drosophila_wing_analysis/thin_wing_model_pseudospectral/test'+ fill_zeros(str(i/10000),4)+'.png')
        plt.close()