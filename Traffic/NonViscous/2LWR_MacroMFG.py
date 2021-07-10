#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
from scipy import integrate
import numdifftools as nd
from scipy.optimize.nonlin import newton_krylov
import scipy.sparse.linalg as spla
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time


# In[8]:


''' inputs '''
''' population 1 : cars
    population 2 : trucks '''
T=5.0 # horizon length 
# number of vehicles : 2N
N=5
# average length of the vehicles in the j-th population
l1=1; l2=3; l=l1+l2
# free flow speed
u1_max=1.0; u2_max=0.6
# jam density
rho1_jam=1.0; rho2_jam=0.6 
L=2*N # road length
CFL=0.75    # CFL<1
rho_a=0.2; rho_b=0.8
beta=1
# """ Non-viscous solution"""
ep1=0.0  # rho
# ep2=0.0  # V
# """ Viscous coeff. """
mu=0.1 

def rho1_int(s): # initial density
    if 0<=s<=(1/5)*N or (2/5)*N<=s<=(3/5)*N or (4/5)*N<=s<=N or (6/5)*N<=s<=(7/5)*N or (8/5)*N<=s<=(9/5)*N:
        return rho_a
    else: 
        return rho_b
def rho2_int(s): # initial density
    if 0<=s<=(1/5)*N or (2/5)*N<=s<=(3/5)*N or (4/5)*N<=s<=N or (6/5)*N<=s<=(7/5)*N or (8/5)*N<=s<=(9/5)*N:
        return rho_b
    else: 
        return rho_a

X= np.linspace(0,L,81) # creating the space grid    
U1 = np.zeros(81)
U2 = np.zeros(81)
for i in range(len(X)):
    U1[i] = rho1_int(X[i]) 
    U2[i] = rho2_int(X[i])
plt.plot(X,U1)
plt.plot(X,U2)
plt.grid()
plt.xlabel('x')


# In[9]:


''' functions '''
# Greenshields desired speed
def U(u_max,rho1,rho2): # u_max= u1_max or u2_max
    return u_max*(1-(rho1/rho1_jam)*(l1/l)-(rho2/rho2_jam)*(l2/l))

# Cost functional
def f_mfg(u,rho1,rho2): # u= u1 or u2
    return 0.5*((U(rho1,rho2)-u)**2) # 2-MFG-LWR

def f_star_p(u_max,p,rho1,rho2): # # u_max= u1_max or u2_max
    return U(u_max,rho1,rho2)-p 
    
def f_star(u_max,p,rho1,rho2): # p=Vx
    return -0.5*(p**2)+U(u_max,rho1,rho2)*p 

def integral(a,b): 
    x2 = lambda x: rho_int(x)
    I=integrate.quad(x2, a, b)
    return I[0]

def rho_int(s): # initial density
    if 0<=s<=(1/5)*N or (2/5)*N<=s<=(3/5)*N or (4/5)*N<=s<=N or (6/5)*N<=s<=(7/5)*N or (8/5)*N<=s<=(9/5)*N:
        return rho_a
    else: 
        return rho_b


def VT(a): # Terminal cost
    return 0.0

def F(w):
    FF=np.zeros(3*Nt*2*Nx+2*2*Nx)
    for n in range(0,Nt): 
        # F_rho1 , F[0]->F[Nt-1] ************** 1 (1)
        FF[n]=w[n+1]-0.5*w[n+Nt+1]+(0.5*dt/dx)*w[n+Nt+1]*w[n+(Nt+1)*2*Nx+Nt]+ep1*(w[n+Nt+1]-2*w[n])
        # F_rho1 , F[Nt*Nx-Nt]->F[Nt*Nx-1] ********** 3 (1)
        FF[Nt*(Nx-1)+n]=w[(Nt+1)*(Nx-1)+n+1]-0.5*w[(Nt+1)*(Nx-2)+n]-(0.5*dt/dx)*w[(Nt+1)*(Nx-2)+n]*w[(Nt+1)*2*Nx+(Nx-2)*Nt+n]+ep1*(-2*w[(Nt+1)*(Nx-1)+n]+w[(Nt+1)*(Nx-2)+n])
        # F_rho2 , F[Nt*Nx]->F[Nt*Nx+Nt-1] ************** 4 (2)
        FF[Nt*Nx+n]=w[Nx*(Nt+1)+n+1]-0.5*w[(Nx+1)*(Nt+1)+n]+(0.5*dt/dx)*w[(Nx+1)*(Nt+1)+n]*w[n+(Nt+1)*2*Nx+(Nx+1)*Nt]+ep1*(w[(Nx+1)*(Nt+1)+n]-2*w[Nx*(Nt+1)+n])
        # F_rho2 , F[2*Nt*Nx-Nt]->F[2*Nt*Nx-1] ********** 6 (2)
        FF[Nt*(2*Nx-1)+n]=w[(Nt+1)*(2*Nx-1)+n+1]-0.5*w[(Nt+1)*(2*Nx-2)+n]-(0.5*dt/dx)*w[(Nt+1)*(2*Nx-2)+n]*w[(Nt+1)*2*Nx+(2*Nx-2)*Nt+n]+ep1*(-2*w[(Nt+1)*(2*Nx-1)+n]+w[(Nt+1)*(2*Nx-2)+n])
        # F_u1 , F[2*Nt*Nx]->F[2*Nt*Nx+Nt-1] *********** 7 (1)
        FF[Nt*2*Nx+n]=w[(Nt+1)*2*Nx+n]-beta*f_star_p(u1_max,w[(2*Nt+1)*2*Nx+n+1]/dx,w[n],w[Nx*(Nt+1)+n])
        # F_u1 , F[3*Nt*Nx-Nt]->F[3*Nt*Nx-1] ********* 9 (1)
        FF[(Nx-1)*Nt+Nt*2*Nx+n]=w[(Nt+1)*2*Nx+(Nx-1)*Nt+n]-beta*f_star_p(u1_max,(w[(2*Nt+1)*2*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(Nx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n],w[(2*Nx-1)*(Nt+1)+n])
        # F_u2 , F[3*Nt*Nx]->F[3*Nt*Nx+Nt-1] *********** 10 (2)
        FF[Nt*Nx+Nt*2*Nx+n]=w[(Nt+1)*2*Nx+Nt*Nx+n]-beta*f_star_p(u2_max,w[(2*Nt+1)*2*Nx+Nx*(Nt+1)+n+1]/dx,w[n],w[Nx*(Nt+1)+n])
        # F_u2 , F[4*Nt*Nx-Nt]->F[4*Nt*Nx-1] ********* 12 (2)
        FF[(2*Nx-1)*Nt+Nt*2*Nx+n]=w[(Nt+1)*2*Nx+(2*Nx-1)*Nt+n]-beta*f_star_p(u2_max,(w[(2*Nt+1)*2*Nx+(2*Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(2*Nx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n],w[(2*Nx-1)*(Nt+1)+n])
        # F_V1 , F[4*Nt*Nx]->F[4*Nt*Nx+Nt-1] *********** 13 (1)
        FF[2*Nt*2*Nx+n]=w[(2*Nt+1)*2*Nx+n+1]-w[(2*Nt+1)*2*Nx+n]+beta*dt*f_star(u1_max,w[(2*Nt+1)*2*Nx+n+1]/dx,w[n],w[Nx*(Nt+1)+n])+ep2*(w[(2*Nt+1)*2*Nx+Nt+n+2]-2*w[(2*Nt+1)*2*Nx+n+1])
        # F_V1 , F[5*Nt*Nx-Nt]->F[5*Nt*Nx-1] ********** 15 (1)
        FF[5*Nt*Nx-Nt+n]=w[(2*Nt+1)*2*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(Nx-1)*(Nt+1)+n]+beta*dt*f_star(u1_max,(w[(2*Nt+1)*2*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(Nx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n],w[(2*Nx-1)*(Nt+1)+n])+ep2*(-2*w[(2*Nt+1)*2*Nx+(Nx-1)*(Nt+1)+n+1]+w[(2*Nt+1)*2*Nx+(Nx-2)*(Nt+1)+n+1])
        # F_V2 , F[5*Nt*Nx]->F[5*Nt*Nx+Nt-1] *********** 16 (2)
        FF[Nt*Nx+2*Nt*2*Nx+n]=w[(2*Nt+1)*2*Nx+Nx*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+Nx*(Nt+1)+n]+beta*dt*f_star(u2_max,w[(2*Nt+1)*2*Nx+Nx*(Nt+1)+n+1]/dx,w[n],w[Nx*(Nt+1)+n])+ep2*(w[(2*Nt+1)*2*Nx+(Nx+1)*(Nt+1)+n+1]-2*w[(2*Nt+1)*2*Nx+Nx*(Nt+1)+n+1])
        # F_V2 , F[6*Nt*Nx-Nt]->F[6*Nt*Nx-1] ********** 18 (2)
        FF[2*Nt*2*Nx+(2*Nx-1)*Nt+n]=w[(2*Nt+1)*2*Nx+(2*Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(2*Nx-1)*(Nt+1)+n]+beta*dt*f_star(u2_max,(w[(2*Nt+1)*2*Nx+(2*Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(2*Nx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n],w[(2*Nx-1)*(Nt+1)+n])+ep2*(-2*w[(2*Nt+1)*2*Nx+(2*Nx-1)*(Nt+1)+n+1]+w[(2*Nt+1)*2*Nx+(2*Nx-2)*(Nt+1)+n+1])
        
        
    for j in range(2,Nx):
        for n in range(0,Nt):
            # F_rho1 , F[Nt]->F[Nt*Nx-Nt-1] ************ 2 (1)
            FF[(j-1)*Nt+n]=w[(j-1)*(Nt+1)+n+1]-0.5*(w[(j-2)*(Nt+1)+n]+w[j*(Nt+1)+n])+(0.5*dt/dx)*(w[j*(Nt+1)+n]*w[(Nt+1)*2*Nx+j*Nt+n]-w[(j-2)*(Nt+1)+n]*w[(Nt+1)*2*Nx+(j-2)*Nt+n])+ep1*(w[j*(Nt+1)+n]-2*w[(j-1)*(Nt+1)+n]+w[(j-2)*(Nt+1)+n])      
            # F_u1 , F[2*Nt*Nx+Nt]->F[3*Nt*Nx-Nt-1] *********** 8 (1)
            FF[(j-1)*Nt+Nt*2*Nx+n]=w[(Nt+1)*2*Nx+(j-1)*Nt+n]-beta*f_star_p(u1_max,(w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(j-2)*(Nt+1)+n+1])/dx,w[(j-1)*(Nt+1)+n],w[(Nx+j-1)*(Nt+1)+n])
            # F_V1 , F[4*Nt*Nx+Nt]->F[5*Nt*Nx-Nt-1] ********* 14 (1)
            FF[(j-1)*Nt+2*Nt*2*Nx+n]=w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n]+beta*dt*f_star(u1_max,(w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(j-2)*(Nt+1)+n+1])/dx,w[(j-1)*(Nt+1)+n],w[(Nx+j-1)*(Nt+1)+n])+ep2*(w[(2*Nt+1)*2*Nx+j*(Nt+1)+n+1]-2*w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n+1]+w[(2*Nt+1)*2*Nx+(j-2)*(Nt+1)+n+1])
        # F_rho1_int , F[6*Nt*Nx+1]->F[6*Nt*Nx+Nx-2] ********** 20 (1)
        FF[6*Nt*Nx+j-1]=w[(j-1)*(Nt+1)]-(1/dx)*integral(x[j-1],x[j])
        # F_V1_ter , F[6*Nt*Nx+2*Nx+1]->F[6*Nt*Nx+3*Nx-2] ********* 26 (1)
        FF[6*Nt*Nx+2*Nx+j-1]=w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+Nt]-VT(x[j])
    # F_rho1_int , F[6*Nt*Nx] ********* 19 (1)
    FF[6*Nt*Nx]=w[0]-(1/dx)*integral(x[0],x[1])
    # F_rho1_int , F[6*Nt*Nx+Nx-1] ********* 21 (1)
    FF[6*Nt*Nx+Nx-1]=w[(Nx-1)*(Nt+1)]-(1/dx)*integral(x[Nx-1],x[Nx])
    # F_V1_ter , F[6*Nt*Nx+2*Nx] *********** 25 (1)
    FF[6*Nt*Nx+2*Nx]=w[(2*Nt+1)*2*Nx+Nt]-VT(x[1])
    # F_V1_ter , F[6*Nt*Nx+3*Nx-1] ************** 27 (1)
    FF[6*Nt*Nx+3*Nx-1]=w[(2*Nt+1)*2*Nx+(Nx-1)*(Nt+1)+Nt]-VT(x[Nx])
        
    for j in range(Nx+2,2*Nx):
        for n in range(0,Nt):
            # F_rho2 , F[Nt*Nx+Nt]->F[2*Nt*Nx-Nt-1] ************ 5 (2)
            FF[(j-1)*Nt+n]=w[(j-1)*(Nt+1)+n+1]-0.5*(w[(j-2)*(Nt+1)+n]+w[j*(Nt+1)+n])+(0.5*dt/dx)*(w[j*(Nt+1)+n]*w[(Nt+1)*2*Nx+j*Nt+n]-w[(j-2)*(Nt+1)+n]*w[(Nt+1)*2*Nx+(j-2)*Nt+n])+ep1*(w[j*(Nt+1)+n]-2*w[(j-1)*(Nt+1)+n]+w[(j-2)*(Nt+1)+n])      
            # F_u2 , F[3*Nt*Nx+Nt]->F[4*Nt*Nx-Nt-1] *********** 11 (2)
            FF[(j-1)*Nt+Nt*2*Nx+n]=w[(Nt+1)*2*Nx+(j-1)*Nt+n]-beta*f_star_p(u2_max,(w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(j-2)*(Nt+1)+n+1])/dx,w[(j-Nx-1)*(Nt+1)+n],w[(j-1)*(Nt+1)+n])
            # F_V2 , F[5*Nt*Nx+Nt]->F[6*Nt*Nx-Nt-1] ********* 17 (2)
            FF[(j-1)*Nt+2*Nt*2*Nx+n]=w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n]+beta*dt*f_star(u2_max,(w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(j-2)*(Nt+1)+n+1])/dx,w[(j-Nx-1)*(Nt+1)+n],w[(j-1)*(Nt+1)+n])+ep2*(w[(2*Nt+1)*2*Nx+j*(Nt+1)+n+1]-2*w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n+1]+w[(2*Nt+1)*2*Nx+(j-2)*(Nt+1)+n+1])
        # F_rho2_int , F[6*Nt*Nx+Nx+1]->F[6*Nt*Nx+2*Nx-2] ********** 23 (2)
        FF[6*Nt*Nx+j-1]=w[(j-1)*(Nt+1)]-(1/dx)*integral(x[j-1],x[j])
        # F_V2_ter , F[6*Nt*Nx+3*Nx+1]->F[6*Nt*Nx+4*Nx-2] ********* 29 (2)
        FF[6*Nt*Nx+2*Nx+j-1]=w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+Nt]-VT(x[j])
    # F_rho2_int , F[6*Nt*Nx+Nx] ********* 22 (2)
    FF[6*Nt*Nx+Nx]=w[Nx*(Nt+1)]-(1/dx)*integral(x[Nx],x[Nx+1])
    # F_rho2_int , F[6*Nt*Nx+2*Nx-1] ********* 24 (2)
    FF[6*Nt*Nx+2*Nx-1]=w[(2*Nx-1)*(Nt+1)]-(1/dx)*integral(x[2*Nx-1],x[2*Nx])
    # F_V2_ter , F[6*Nt*Nx+3*Nx] *********** 28 (2) 
    FF[6*Nt*Nx+3*Nx]=w[(2*Nt+1)*2*Nx+Nx*(Nt+1)+Nt]-VT(x[Nx+1])
    # F_V2_ter , F[6*Nt*Nx+4*Nx-1] ************** 30 (2)
    FF[6*Nt*Nx+4*Nx-1]=w[(2*Nt+1)*2*Nx+(2*Nx-1)*(Nt+1)+Nt]-VT(x[2*Nx])
    
    return FF


def get_preconditioner(a):
    beta=0 # Ignoring the forward-backward coupling  parts
    Jac=nd.Jacobian(F)
    J1=Jac(a)
    # the *incomplete LU* decomposition
    J1_ilu = spla.spilu(J1)
    # matrix-vector product -> LinearOperator 
    M_x = lambda r: J1_ilu.solve(r)
    M = spla.LinearOperator(J1.shape, M_x)

    return M
    

def interpol(Nt,Nt_mul,Nx,Nx_mul,w): # 1D interpolation
    
    """" Go from a coarse grid Nt*Nx to a finer grid spacing (Nt_mul*Nt)*(Nx_mul*Nx) """""

    n=w.shape[0] # n=3Nt*Nx+2Nx
    i = np.indices(w.shape)[0]/(n-1)  # [0, ..., 1]
    new_n = 3*(Nt_mul*Nt)*(Nx_mul*Nx)+2*(Nx_mul*Nx)
    print('n={n}, new_n={new_n}'.format(n=n,new_n=new_n))
    new_i = np.linspace(0, 1, new_n)
    new_w=griddata(i, w, new_i, method="cubic")  # method{‘linear’, ‘nearest’, ‘cubic’}
    
    return Nt_mul*Nt, Nx_mul*Nx, new_w

def solution(sol,rho1,u1,V1,Q1,rho2,u2,V2,Q2):
    for j in range(1,Nx+1):
        for n in range(0,Nt):
            rho1[j,n]=sol[(j-1)*(Nt+1)+n]
            u1[j,n]=sol[(Nt+1)*2*Nx+(j-1)*Nt+n]
            V1[j,n]=sol[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n]
            Q1[j,n]=rho1[j,n]*u1[j,n]
        rho1[j,Nt]=sol[(j-1)*(Nt+1)+Nt]
        V1[j,Nt]=sol[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+Nt]
    for j in range(Nx+1,2*Nx+1):
        for n in range(0,Nt):
            rho2[j-Nx,n]=sol[(j-1)*(Nt+1)+n]
            u2[j-Nx,n]=sol[(Nt+1)*2*Nx+(j-1)*Nt+n]
            V2[j-Nx,n]=sol[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n]
            Q2[j-Nx,n]=rho2[j-Nx,n]*u2[j-Nx,n]
        rho2[j-Nx,Nt]=sol[(j-1)*(Nt+1)+Nt]
        V2[j-Nx,Nt]=sol[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+Nt]
    for n in range(0,Nt+1): # periodic boundary conditions
        rho1[0,n]=rho1[Nx,n]
        V1[0,n]=V1[Nx,n]
        rho2[0,n]=rho2[Nx,n]
        V2[0,n]=V2[Nx,n]
    for n in range(0,Nt):
        u1[0,n]=f_star_p(u1_max,V1[0,n+1]/dx,rho1[0,n],rho2[0,n])
        Q1[0,n]=rho1[0,n]*u1[0,n]
        u2[0,n]=f_star_p(u2_max,V2[0,n+1]/dx,rho1[0,n],rho2[0,n])
        Q2[0,n]=rho2[0,n]*u2[0,n]
    return 0

def plotting(text,t,x,rho1,u1,V1,Q1,rho2,u2,V2,Q2,Nx_list,Error1_list,Error2_list,fig1,fig2,fig3,fig4):
    tt, xx = np.meshgrid(t, x)
    fig1 = plt.figure(figsize=(6, 5), dpi=100)
    ax1 = fig1.gca(projection='3d')
    ax1.plot_surface(xx, tt, rho1, cmap=cm.viridis)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$t$')
    ax1.set_zlabel('density-1')
    ax1.invert_xaxis()
#     ax.text2D(0.05, 0.95, text, transform=ax1.transAxes)
#     plt.savefig(fig1)
    
    fig2 = plt.figure(figsize=(6, 5), dpi=100)
    ax2 = fig2.gca(projection='3d')
    ax2.plot_surface(xx, tt, rho2, cmap=cm.viridis)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$t$')
    ax2.set_zlabel('density-2')
    ax2.invert_xaxis()
#     ax.text2D(0.05, 0.95, text, transform=ax2.transAxes)
#     plt.savefig(fig2)

    plt.figure(figsize=(20, 8))
    plt.subplot(2,3,1)
    plt.plot(x,rho1[:,0],'b',label='density-1')
#     plt.plot(x,u1[:,0],'b',label='speed-1')
#     plt.plot(x,V1[:,0],'g',label='Optimal cost-1')
    plt.plot(x,rho2[:,0],'r',label='density-2')
#     plt.plot(x,u2[:,0],r,label='speed-2')
#     plt.plot(x,V2[:,0],label='Optimal cost-2')
    plt.legend()
    plt.grid()
    plt.title("t=0.0")
#     plt.xlabel('x')
    plt.subplot(2,3,2)
    plt.plot(x,rho1[:,int(Nt/5)],'b',label='density-1')
#     plt.plot(x,u1[:,int(Nt/5)],'b',label='speed-1')
#     plt.plot(x,V1[:,int(Nt/5)],'g',label='Optimal cost-1')
    plt.plot(x,rho2[:,int(Nt/5)],'r',label='density-2')
#     plt.plot(x,u2[:,int(Nt/5)],'b',label='speed-2')
#     plt.plot(x,V2[:,int(Nt/5)],label='Optimal cost-2')
    plt.grid()
    plt.legend()
    plt.title("t={t}".format(t=round(t[int(Nt/5)],3)))
#     plt.xlabel('x')
    plt.subplot(2,3,3)
    plt.plot(x,rho1[:,int(2*Nt/5)],'b',label='density-1')
#     plt.plot(x,u1[:,int(Nt/2)],'g',label='speed-1')
#     plt.plot(x,V1[:,int(Nt/2)],'r',label='Optimal cost-1')
    plt.plot(x,rho2[:,int(2*Nt/5)],'r',label='density-2')
#     plt.plot(x,u2[:,int(Nt/2)],label='speed-2')
#     plt.plot(x,V2[:,int(Nt/2)],label='Optimal cost-2')
    plt.grid()
    plt.legend()
    plt.title("t={t}".format(t=round(t[int(2*Nt/5)],3)))
#     plt.xlabel('x')
    plt.subplot(2,3,4)
    plt.plot(x,rho1[:,int(3*Nt/5)],'b',label='density-1')
#     plt.plot(x,u1[:,int(3*Nt/5)],'b',label='speed-1')
#     plt.plot(x,V1[:,int(3*Nt/5)],'g',label='Optimal cost-1')
    plt.plot(x,rho2[:,int(3*Nt/5)],'r',label='density-2')
#     plt.plot(x,u2[:,int(3*Nt/5)],label='speed-2')
#     plt.plot(x,V2[:,int(3*Nt/5)],label='Optimal cost-2')
    plt.grid()
    plt.legend()
    plt.title("t={t}".format(t=round(t[int(3*Nt/5)],3)))
    plt.xlabel('x')
    plt.subplot(2,3,5)
    plt.plot(x,rho1[:,int(4*Nt/5)],'b',label='density-1')
#     plt.plot(x,u1[:,int(4*Nt/5)],'b',label='speed-1')
#     plt.plot(x,V1[:,int(4*Nt/5)],'g',label='Optimal cost-1')
    plt.plot(x,rho2[:,int(4*Nt/5)],'r',label='density-2')
#     plt.plot(x,u2[:,int(4*Nt/5)],label='speed-2')
#     plt.plot(x,V2[:,int(4*Nt/5)],label='Optimal cost-2')
    plt.grid()
    plt.legend()
    plt.title("t={t}".format(t=round(t[int(4*Nt/5)],3)))
    plt.xlabel('x')
    plt.subplot(2,3,6)
    plt.plot(x,rho1[:,Nt-1],'b',label='density-1')
#     plt.plot(x,u1[:,Nt-1],'b',label='speed-1')
#     plt.plot(x,V1[:,Nt-1],'g',label='Optimal cost-1')
    plt.plot(x,rho2[:,Nt-1],'r',label='density-2')
#     plt.plot(x,u2[:,Nt-1],label='speed-2')
#     plt.plot(x,V2[:,Nt-1],label='Optimal cost-2')
    plt.grid()
    plt.legend()
    plt.title("t={t}".format(t=round(t[Nt-1],3)))
    plt.xlabel('x')
#     plt.savefig(fig3)

    plt.figure(figsize=(20, 8))
    plt.subplot(2,3,1)
    plt.plot(x,u1[:,0],'b',label='speed-1')
    plt.plot(x,u2[:,0],'r',label='speed-2')
    plt.legend()
    plt.grid()
    plt.title("t=0.0")
#     plt.xlabel('x')
    plt.subplot(2,3,2)
    plt.plot(x,u1[:,int(Nt/5)],'b',label='speed-1')
    plt.plot(x,u2[:,int(Nt/5)],'r',label='speed-2')
    plt.grid()
    plt.legend()
    plt.title("t={t}".format(t=round(t[int(Nt/5)],3)))
#     plt.xlabel('x')
    plt.subplot(2,3,3)
    plt.plot(x,u1[:,int(2*Nt/5)],'b',label='speed-1')
    plt.plot(x,u2[:,int(2*Nt/5)],'r',label='speed-2')
    plt.grid()
    plt.legend()
    plt.title("t={t}".format(t=round(t[int(2*Nt/5)],3)))
#     plt.xlabel('x')
    plt.subplot(2,3,4)
    plt.plot(x,u1[:,int(3*Nt/5)],'b',label='speed-1')
    plt.plot(x,u2[:,int(3*Nt/5)],'r',label='speed-2')
    plt.grid()
    plt.legend()
    plt.title("t={t}".format(t=round(t[int(3*Nt/5)],3)))
    plt.xlabel('x')
    plt.subplot(2,3,5)
    plt.plot(x,u1[:,int(4*Nt/5)],'b',label='speed-1')
    plt.plot(x,u2[:,int(4*Nt/5)],'r',label='speed-2')
    plt.grid()
    plt.legend()
    plt.title("t={t}".format(t=round(t[int(4*Nt/5)],3)))
    plt.xlabel('x')
    plt.subplot(2,3,6)
    plt.plot(x,u1[:,Nt-1],'b',label='speed-1')
    plt.plot(x,u2[:,Nt-1],'r',label='speed-2')
    plt.grid()
    plt.legend()
    plt.title("t={t}".format(t=round(t[Nt-1],3)))
    plt.xlabel('x')
    
#     plt.figure()
#     plt.plot(Nx_list,Error1_list,'*',label='1')
#     plt.plot(Nx_list,Error2_list,'*',label='2')
#     plt.xlabel('Spatial grid size')
#     plt.ylabel('error')
#     plt.grid()
#     plt.title("convergence of solution algorithm")
    
    plt.figure()
    plt.plot(rho1[:,Nt-1],Q1[:,Nt-1],label='flow-density-1')
    plt.plot(rho2[:,Nt-1],Q2[:,Nt-1],label='flow-density-2')
    plt.xlabel('density')
    plt.ylabel('Flow')
    plt.grid()
    plt.title("Fundamental diagram (T={T})".format(T=T))
#     plt.savefig(fig4)
    return 0

def convergence(guess,sol,o):
    rho1=np.zeros((Nx+1,Nt+1))
    u1=np.zeros((Nx+1,Nt))
    V1=np.zeros((Nx+1,Nt+1))
    Q1=np.zeros((Nx+1,Nt))
    rho2=np.zeros((Nx+1,Nt+1))
    u2=np.zeros((Nx+1,Nt))
    V2=np.zeros((Nx+1,Nt+1))
    Q2=np.zeros((Nx+1,Nt))
    solution(guess,rho1,u1,V1,Q1,rho2,u2,V2,Q2)
    rho1_mfg=np.zeros((Nx+1,Nt+1))
    u1_mfg=np.zeros((Nx+1,Nt))
    V1_mfg=np.zeros((Nx+1,Nt+1))
    Q1_mfg=np.zeros((Nx+1,Nt))
    rho2_mfg=np.zeros((Nx+1,Nt+1))
    u2_mfg=np.zeros((Nx+1,Nt))
    V2_mfg=np.zeros((Nx+1,Nt+1))
    Q2_mfg=np.zeros((Nx+1,Nt))
    solution(sol,rho1_mfg,u1_mfg,V1_mfg,Q1_mfg,rho2_mfg,u2_mfg,V2_mfg,Q2_mfg)
    error1=np.linalg.norm(rho1_mfg-rho1,ord=o)/np.linalg.norm(rho1_mfg,ord=o)+np.linalg.norm(u1_mfg-u1,ord=o)/np.linalg.norm(u1_mfg,ord=o)
    error2=np.linalg.norm(rho2_mfg-rho2,ord=o)/np.linalg.norm(rho2_mfg,ord=o)+np.linalg.norm(u2_mfg-u2,ord=o)/np.linalg.norm(u2_mfg,ord=o)
    return error1,error2


# In[ ]:


""" solve in coarse grid """
Nx=24; Nt=20 # spatial-temporal grid sizes
print(3*Nt*2*Nx+2*2*Nx)
dx=L/(2*Nx) # spatial step size
# dt=min(T/Nt,CFL*dx/abs(u1_max)) # temporal step size
dt=min(T/Nt,CFL*dx/abs(u1_max),CFL*(dx**2)/(2*mu)) # temporal step size
# ep1=-mu*dt/(dx**2)  # rho
ep2=mu*dt/(dx**2) # V
print('dx={dx}, dt={dt}'.format(dx=round(dx,3),dt=round(dt,3)))
x=np.linspace(0,L,2*Nx+1)
t=np.linspace(0,T,Nt+1)
guess0 = np.zeros(3*Nt*2*Nx+2*2*Nx)
t0 = time.process_time()   ###
sol0 = newton_krylov(F, guess0, method='lgmres', verbose=1, inner_M=get_preconditioner(guess0))
t1 = time.process_time()   ###
print("Time spent :",t1-t0)
# np.savetxt('Sol_NonSep_T1_N151.dat', sol0)
np.savetxt('Sol0_2LWR_T1_N5.dat', sol0)
# print('sol0=',sol0)
# """ Error 0 """
# Error1_list=[]
# Error2_list=[]
# Nx_list=[]
# error01,error02=convergence(guess0,sol0,1)
# Error1_list.append(error01)
# Error2_list.append(error02)
# Nx_list.append(Nx)
# data = np.loadtxt('2LWR-Sol0.dat')


# In[144]:


print(sol0.shape)
# N,Nx,Nt,dt=np.loadtxt('list_NonSep_T1_N21.dat')
# L=N
# sol=np.loadtxt('sol_NonSep_T1_N21.dat')
""" MFG Solution """
rho1_mfg=np.zeros((Nx+1,Nt+1))
u1_mfg=np.zeros((Nx+1,Nt))
V1_mfg=np.zeros((Nx+1,Nt+1))
Q1_mfg=np.zeros((Nx+1,Nt))
rho2_mfg=np.zeros((Nx+1,Nt+1))
u2_mfg=np.zeros((Nx+1,Nt))
V2_mfg=np.zeros((Nx+1,Nt+1))
Q2_mfg=np.zeros((Nx+1,Nt))
solution(sol0,rho1_mfg,u1_mfg,V1_mfg,Q1_mfg,rho2_mfg,u2_mfg,V2_mfg,Q2_mfg)
x_mfg=np.linspace(0,N,Nx+1)
t_mfg=np.linspace(0,T,Nt+1)


# In[1]:


print(x)
print(rho1_mfg[:,0])
print(u1_mfg[:,0])
print(rho2_mfg[:,0])
print(u2_mfg[:,0])


# In[146]:


title1="\n Non Viscous MFG-2LWR"
fig1= 'Sol0_2LWR_T1_N5_1.png'
fig2= 'Sol0_2LWR_T1_N5_2.png'
fig3= 'Sol0_2LWR_T1_N5_3.png'
fig4= 'Sol0_2LWR_T1_N5_4.png'
plotting(title1,t_mfg,x_mfg,rho1_mfg,u1_mfg,V1_mfg,Q1_mfg,rho2_mfg,u2_mfg,V2_mfg,Q2_mfg,Nx_list,Error1_list,Error2_list,fig1,fig2,fig3,fig4)


# In[ ]:




