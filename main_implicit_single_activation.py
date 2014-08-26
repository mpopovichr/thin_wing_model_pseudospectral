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
import matplotlib.gridspec as gridspec

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

dt= 0.001       #time step

L= -0.1      #HB interface
tauT1= 3.7 #
h0= 1./2.4  #relative to total length of wing
k= 25.
K1= 1.
K2= 10.
Gamma= 300.0
tau= 1.7 ## TODO - check!!!
lambda1=1./2./tau/K1
print lambda1
#tau= 1.
lambda2_h= -.05
lambda2_b= -.055
zeta_h= .04
zeta_b= .025
zetabar_h= 5.
zetabar_b= 2.
#xi_h= 0.655
#xi_b= 0.29

M= np.array([[1.,0.,0.],[0.,1.,dt],[0.,-dt/tau,tauT1+dt]])
M_inv= np.linalg.inv(M)

Q_plus= np.zeros(N+1)
Q_minus= np.zeros(N+1)
v_x= np.zeros(N+1)
v_yy= np.zeros(N+1)
h= np.zeros(N+1)+h0
T_minus= np.zeros(N+1)

t0_a= 10.8
sigma_t_a= 0.4
sigma= 0.04
active_space_a= np.array(map(lambda x: 1./(1+np.exp((x-L)/sigma)),x))*1./(1+np.exp(t0_a/sigma_t_a))
active_const_a= np.array(map(lambda x: 1.,x))*1./(1+np.exp(t0_a/sigma_t_a))
t0_i= 11.6
sigma_t_i= .6
sigma= 0.04
active_space_i= np.array(map(lambda x: 1./(1+np.exp((x-L)/sigma)),x))*1./(1+np.exp(t0_i/sigma_t_i))
active_const_i= np.array(map(lambda x: 1.,x))*1./(1+np.exp(t0_i/sigma_t_i))

rng= np.linspace(-1,1,1000)
x_step= 2./1000
interpolator= BI(x)
avg_h_blade= []
avg_Q_minus_blade= []
avg_T_minus_blade= []
avg_shear_blade= []

zeta= zeta_h*active_space_a + zeta_b*active_const_a
zetabar= zetabar_h*active_space_i + zetabar_b*active_const_i
lambda2= lambda2_h*active_space_a + lambda2_b*active_const_a
#xi = xi_h*active_space_a + xi_b*active_const_a

n_images= 35
image_step= 1*1000

for i in np.arange(n_images*image_step):
    if i%image_step == 0:
        fig= plt.figure(figsize=(20,16))
        gs= gridspec.GridSpec(3,3)
        ax0= fig.add_subplot(gs[1:,1:])
        axh= fig.add_subplot(gs[1,0])
        axh.set_title('h')
        axQ_minus= fig.add_subplot(gs[0,1])
        axQ_minus.set_title('Q_minus')
        axT_minuse=fig.add_subplot(gs[0,2])
        axT_minuse.set_title('T_minus')
        axshear= fig.add_subplot(gs[2,0])
        axshear.set_title('shear')
        interpolator.set_yi(active_space_a)
        ac_space_a_plot= np.array(map(interpolator, rng))
        HB_int= np.argmin(np.abs(ac_space_a_plot- 0.5*(np.max(ac_space_a_plot)-np.min(ac_space_a_plot))))
        print HB_int
        ax0.plot(rng, np.array(map(interpolator, rng))/10.+ np.max(active_const_a)/10.,label= 'active_a')
        interpolator.set_yi(active_space_i)
        ax0.plot(rng, np.array(map(interpolator, rng))/10.+ np.max(active_const_i)/10.,label= 'active_i')
        interpolator.set_yi(v_x)
        ax0.plot(rng, np.array(map(interpolator,rng))*3., label= 'vx')
        interpolator.set_yi(h)
        h_plot=np.array(map(interpolator,rng))
        avg_h_blade.append(np.sum(h_plot[HB_int:])*x_step/(rng[-1]-rng[HB_int]))
        axh.plot(range(len(avg_h_blade)),avg_h_blade)
        axh.set_xlim([0,n_images])
        ax0.plot(rng, np.array(map(interpolator,rng))/5., label='h')
        interpolator.set_yi(Q_minus)
        Q_minus_plot= np.array(map(interpolator,rng))
        ax0.plot(rng, Q_minus_plot/5., label='Q_-')
        avg_Q_minus_blade.append(np.sum(Q_minus_plot[HB_int:])*x_step/(rng[-1]-rng[HB_int]))
        axQ_minus.plot(range(len(avg_h_blade)),avg_Q_minus_blade)
        axQ_minus.set_xlim([0,n_images])
        interpolator.set_yi(T_minus)
        T_minus_plot= np.array(map(interpolator, rng))
        ax0.plot(rng, T_minus_plot,label='T_-')
        avg_T_minus_blade.append(np.sum(T_minus_plot[HB_int:])*x_step/(rng[-1]-rng[HB_int]))
        axT_minuse.plot(range(len(avg_h_blade)),avg_T_minus_blade)
        axT_minuse.set_xlim([0,n_images])
        interpolator.set_yi(D.dot(v_x))
        vxx_plot= np.array(map(interpolator,rng))
        ax0.plot(rng, vxx_plot, label='v_xx')
        interpolator.set_yi(v_yy)
        vyy_plot= np.array(map(interpolator, rng))
        ax0.plot(rng, vyy_plot, label='v_yy', color='black', linestyle='-.')
        avg_shear_blade.append(np.sum(vxx_plot[HB_int:]-vyy_plot[HB_int:])*x_step/(rng[-1]-rng[HB_int]))
        axshear.plot(range(len(avg_h_blade)),avg_shear_blade)
        axshear.set_xlim([0,n_images])
        ax= plt.gca()
        ax0.text(0.1,0.9,str(dt*i-10.), transform=ax.transAxes, fontsize= 20)
        ax0.grid()
        axh.grid()
        axT_minuse.grid()
        axQ_minus.grid()
        axshear.grid()
        ax0.legend()
        ax0.set_ylim([-.16,.3])
        plt.savefig('/home/mpopovic/Documents/Work/Projects/drosophila_wing_analysis/thin_wing_model_pseudospectral/implicit/test'+ fill_zeros(str(i/image_step),4)+'.png')
        plt.close()
    dxvx= D.dot(v_x)
    a0= dt*dxvx + dt*v_yy + Q_plus - dt*v_x*D.dot(Q_plus)
    a1= dt*dxvx - dt*v_yy + Q_minus - dt*v_x*D.dot(Q_minus)
    a2= tauT1*T_minus + 2*dt*(zeta*lambda1 + lambda2) #xi
    a= np.array([a0,a1,a2])
    Q_plus, Q_minus, T_minus= np.linalg.solve(M,a)
    ac_factor_const_a= np.array(map(lambda x: x*(1+np.exp(-(i*dt-t0_a)/sigma_t_a)),active_const_a))
    ac_factor_space_a= np.array(map(lambda x: x*(1+np.exp(-(i*dt-t0_a)/sigma_t_a)),active_space_a))
    ac_factor_const_i= np.array(map(lambda x: x*(1+np.exp(-(i*dt-t0_i)/sigma_t_i)),active_const_i))
    ac_factor_space_i= np.array(map(lambda x: x*(1+np.exp(-(i*dt-t0_i)/sigma_t_i)),active_space_i))
    active_const_new_a= active_const_a+ dt*(1./sigma_t_a*(active_const_a-active_const_a*active_const_a/ac_factor_const_a))#-1* v_x*D.dot(active_const))
    active_space_new_a= active_space_a+ dt*(1./sigma_t_a*(active_space_a-active_space_a*active_space_a/ac_factor_space_a)-1* v_x*D.dot(active_space_a))
    active_const_new_i= active_const_i+ dt*(1./sigma_t_i*(active_const_i-active_const_i*active_const_i/ac_factor_const_i))#-1* v_x*D.dot(active_const))
    active_space_new_i= active_space_i+ dt*(1./sigma_t_i*(active_space_i-active_space_i*active_space_i/ac_factor_space_i)-1* v_x*D.dot(active_space_i))
    active_const_a= active_const_new_a
    active_space_a= active_space_new_a
    active_const_i= active_const_new_i
    active_space_i= active_space_new_i
    zeta= zeta_h*active_space_a + zeta_b*active_const_a
    zetabar= zetabar_h*active_space_i + zetabar_b*active_const_i
    #xi = xi_h*active_space_a + xi_b*active_const_a
    lambda2= lambda2_h*active_space_a + lambda2_b*active_const_a
    h_new= -1./k*(K2*Q_plus - K1*Q_minus) + h0 + (zeta - zetabar)/k
    v_yy= (h_new - h)/dt + v_x*D.dot(h)
    h= h_new
    H= np.log(h/h0)
    v_x= 1./Gamma*((K1*Q_minus + K2*Q_plus + zetabar + zeta)*D.dot(H) + D.dot(K1*Q_minus + K2*Q_plus + zeta + zetabar))
    if i%image_step == 0:
        print i*dt, np.min(v_x), np.max(active_space_a), np.max(ac_factor_space_a),np.max(active_space_i), np.min(h), np.min(T_minus), np.max(T_minus)
    v_x[0]= 0.#boundary_a(i)
    v_x[-1]= 0.#boundary_b(i)