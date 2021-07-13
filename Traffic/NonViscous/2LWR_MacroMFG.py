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
ep2=0.0  # V
# """ Viscous coeff. """
mu=0.1 


''' functions '''
''' 2-MFG-LWR '''
# Greenshields desired speed
def U(u_max,rho1,rho2): # u_max= u1_max or u2_max
    return u_max*(1-(rho1/rho1_jam)*(l1/l)-(rho2/rho2_jam)*(l2/l))

# Cost functional
# Greenshields desired speed
def U(u_max,rho1,rho2): # u_max= u1_max or u2_max
    return u_max*(1-(rho1/rho1_jam)*(l1/(l1+l2))-(rho2/rho2_jam)*(l2/(l1+l2)))
#     return u_max*(1-max(min(rho1*l1+rho2*l2,1),0))

# Cost functional
def f_mfg(u,u_max,rho1,rho2): # u= u1 or u2
    return 0.5*((U(u_max,rho1,rho2)-u)**2) # 2-MFG-LWR

def f_star_p(u_max,p,rho1,rho2): # # u_max= u1_max or u2_max
#     return max(min(U(u_max,rho1,rho2)-p,u_max),0)
    return U(u_max,rho1,rho2)-p 
    
def f_star(u_max,p,rho1,rho2): # p=Vx
    return -0.5*(p**2)+U(u_max,rho1,rho2)*p 

def integral(k,a,b):
    if k==1 : 
        x2 = lambda x: rho1_int(x)
        I=integrate.quad(x2, a, b)
    elif k==2 :
        x2 = lambda x: rho2_int(x)
        I=integrate.quad(x2, a, b)
#     I=integrate.quad(x2, a, b)
    return I[0]

def VT(a): # Terminal cost
    return 0.0

''' Integrated initial density'''
def rho2_int(s): # initial density
    if 0<=s<=(1/5)*L or (2/5)*L<=s<=(3/5)*L or (4/5)*L<=s<=L:
        return rho_a
    else: 
        return rho_b
def rho1_int(s): # initial density
    if 0<=s<=(1/5)*L or (2/5)*L<=s<=(3/5)*L or (4/5)*L<=s<=L:
        return rho_b
    else: 
        return rho_a

''' Fully segregated initial density'''
# def rho1_int(s): # initial density
#     if 0<=s<=(1/2)*L :
#         return rho_a
#     else: 
#         return rho_a+(rho_b-rho_a)*(s-0.5*L)*np.exp(-0.2*((s-L/2)**2))
# def rho2_int(s): # initial density
#     if 0<=s<=(1/2)*L :
#         return rho_a+(rho_b-rho_a)*(0.5*L-s)*np.exp(-0.2*((s-L/2)**2))
#     else: 
#         return rho_a

def VT(a): # Terminal cost
    return 0.0

beta=1
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
        FF[Nt*2*Nx+n]=w[(Nt+1)*2*Nx+n]-beta*f_star_p(u1_max,w[(2*Nt+1)*2*Nx+n+1]/dx,w[Nx*(Nt+1)+n],w[n])
        # F_u1 , F[3*Nt*Nx-Nt]->F[3*Nt*Nx-1] ********* 9 (1)
        FF[(Nx-1)*Nt+Nt*2*Nx+n]=w[(Nt+1)*2*Nx+(Nx-1)*Nt+n]-beta*f_star_p(u1_max,(w[(2*Nt+1)*2*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(Nx-2)*(Nt+1)+n+1])/dx,w[(2*Nx-1)*(Nt+1)+n],w[(Nx-1)*(Nt+1)+n])
        # F_u2 , F[3*Nt*Nx]->F[3*Nt*Nx+Nt-1] *********** 10 (2)
        FF[Nt*Nx+Nt*2*Nx+n]=w[(Nt+1)*2*Nx+Nt*Nx+n]-beta*f_star_p(u2_max,w[(2*Nt+1)*2*Nx+Nx*(Nt+1)+n+1]/dx,w[n],w[Nx*(Nt+1)+n])
        # F_u2 , F[4*Nt*Nx-Nt]->F[4*Nt*Nx-1] ********* 12 (2)
        FF[(2*Nx-1)*Nt+Nt*2*Nx+n]=w[(Nt+1)*2*Nx+(2*Nx-1)*Nt+n]-beta*f_star_p(u2_max,(w[(2*Nt+1)*2*Nx+(2*Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(2*Nx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n],w[(2*Nx-1)*(Nt+1)+n])
        # F_V1 , F[4*Nt*Nx]->F[4*Nt*Nx+Nt-1] *********** 13 (1)
        FF[2*Nt*2*Nx+n]=w[(2*Nt+1)*2*Nx+n+1]-w[(2*Nt+1)*2*Nx+n]+beta*dt*f_star(u1_max,w[(2*Nt+1)*2*Nx+n+1]/dx,w[Nx*(Nt+1)+n],w[n])+ep2*(w[(2*Nt+1)*2*Nx+Nt+n+2]-2*w[(2*Nt+1)*2*Nx+n+1])
        # F_V1 , F[5*Nt*Nx-Nt]->F[5*Nt*Nx-1] ********** 15 (1)
        FF[5*Nt*Nx-Nt+n]=w[(2*Nt+1)*2*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(Nx-1)*(Nt+1)+n]+beta*dt*f_star(u1_max,(w[(2*Nt+1)*2*Nx+(Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(Nx-2)*(Nt+1)+n+1])/dx,w[(2*Nx-1)*(Nt+1)+n],w[(Nx-1)*(Nt+1)+n])+ep2*(-2*w[(2*Nt+1)*2*Nx+(Nx-1)*(Nt+1)+n+1]+w[(2*Nt+1)*2*Nx+(Nx-2)*(Nt+1)+n+1])
        # F_V2 , F[5*Nt*Nx]->F[5*Nt*Nx+Nt-1] *********** 16 (2)
        FF[Nt*Nx+2*Nt*2*Nx+n]=w[(2*Nt+1)*2*Nx+Nx*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+Nx*(Nt+1)+n]+beta*dt*f_star(u2_max,w[(2*Nt+1)*2*Nx+Nx*(Nt+1)+n+1]/dx,w[n],w[Nx*(Nt+1)+n])+ep2*(w[(2*Nt+1)*2*Nx+(Nx+1)*(Nt+1)+n+1]-2*w[(2*Nt+1)*2*Nx+Nx*(Nt+1)+n+1])
        # F_V2 , F[6*Nt*Nx-Nt]->F[6*Nt*Nx-1] ********** 18 (2)
        FF[2*Nt*2*Nx+(2*Nx-1)*Nt+n]=w[(2*Nt+1)*2*Nx+(2*Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(2*Nx-1)*(Nt+1)+n]+beta*dt*f_star(u2_max,(w[(2*Nt+1)*2*Nx+(2*Nx-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(2*Nx-2)*(Nt+1)+n+1])/dx,w[(Nx-1)*(Nt+1)+n],w[(2*Nx-1)*(Nt+1)+n])+ep2*(-2*w[(2*Nt+1)*2*Nx+(2*Nx-1)*(Nt+1)+n+1]+w[(2*Nt+1)*2*Nx+(2*Nx-2)*(Nt+1)+n+1])
        
        
    for j in range(2,Nx):
        for n in range(0,Nt):
            # F_rho1 , F[Nt]->F[Nt*Nx-Nt-1] ************ 2 (1)
            FF[(j-1)*Nt+n]=w[(j-1)*(Nt+1)+n+1]-0.5*(w[(j-2)*(Nt+1)+n]+w[j*(Nt+1)+n])+(0.5*dt/dx)*(w[j*(Nt+1)+n]*w[(Nt+1)*2*Nx+j*Nt+n]-w[(j-2)*(Nt+1)+n]*w[(Nt+1)*2*Nx+(j-2)*Nt+n])+ep1*(w[j*(Nt+1)+n]-2*w[(j-1)*(Nt+1)+n]+w[(j-2)*(Nt+1)+n])      
            # F_u1 , F[2*Nt*Nx+Nt]->F[3*Nt*Nx-Nt-1] *********** 8 (1)
            FF[(j-1)*Nt+Nt*2*Nx+n]=w[(Nt+1)*2*Nx+(j-1)*Nt+n]-beta*f_star_p(u1_max,(w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(j-2)*(Nt+1)+n+1])/dx,w[(Nx+j-1)*(Nt+1)+n],w[(j-1)*(Nt+1)+n])
            # F_V1 , F[4*Nt*Nx+Nt]->F[5*Nt*Nx-Nt-1] ********* 14 (1)
            FF[(j-1)*Nt+2*Nt*2*Nx+n]=w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n]+beta*dt*f_star(u1_max,(w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n+1]-w[(2*Nt+1)*2*Nx+(j-2)*(Nt+1)+n+1])/dx,w[(Nx+j-1)*(Nt+1)+n],w[(j-1)*(Nt+1)+n])+ep2*(w[(2*Nt+1)*2*Nx+j*(Nt+1)+n+1]-2*w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+n+1]+w[(2*Nt+1)*2*Nx+(j-2)*(Nt+1)+n+1])
        # F_rho1_int , F[6*Nt*Nx+1]->F[6*Nt*Nx+Nx-2] ********** 20 (1)
        FF[6*Nt*Nx+j-1]=w[(j-1)*(Nt+1)]-(1/dx)*integral(1,x[j-1],x[j])
        # F_V1_ter , F[6*Nt*Nx+2*Nx+1]->F[6*Nt*Nx+3*Nx-2] ********* 26 (1)
        FF[6*Nt*Nx+2*Nx+j-1]=w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+Nt]-VT(x[j])
    # F_rho1_int , F[6*Nt*Nx] ********* 19 (1)
    FF[6*Nt*Nx]=w[0]-(1/dx)*integral(1,x[0],x[1])
    # F_rho1_int , F[6*Nt*Nx+Nx-1] ********* 21 (1)
    FF[6*Nt*Nx+Nx-1]=w[(Nx-1)*(Nt+1)]-(1/dx)*integral(1,x[Nx-1],x[Nx])
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
        FF[6*Nt*Nx+j-1]=w[(j-1)*(Nt+1)]-(1/dx)*integral(2,x[j-1-Nx],x[j-Nx])
        # F_V2_ter , F[6*Nt*Nx+3*Nx+1]->F[6*Nt*Nx+4*Nx-2] ********* 29 (2)
        FF[6*Nt*Nx+2*Nx+j-1]=w[(2*Nt+1)*2*Nx+(j-1)*(Nt+1)+Nt]-VT(x[j-Nx])
    # F_rho2_int , F[6*Nt*Nx+Nx] ********* 22 (2)
    FF[6*Nt*Nx+Nx]=w[Nx*(Nt+1)]-(1/dx)*integral(2,x[0],x[1])
    # F_rho2_int , F[6*Nt*Nx+2*Nx-1] ********* 24 (2)
    FF[6*Nt*Nx+2*Nx-1]=w[(2*Nx-1)*(Nt+1)]-(1/dx)*integral(2,x[Nx-1],x[Nx])
    # F_V2_ter , F[6*Nt*Nx+3*Nx] *********** 28 (2) 
    FF[6*Nt*Nx+3*Nx]=w[(2*Nt+1)*2*Nx+Nx*(Nt+1)+Nt]-VT(x[1])
    # F_V2_ter , F[6*Nt*Nx+4*Nx-1] ************** 30 (2)
    FF[6*Nt*Nx+4*Nx-1]=w[(2*Nt+1)*2*Nx+(2*Nx-1)*(Nt+1)+Nt]-VT(x[Nx])
    
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

""" solve in coarse grid """
Nx=30; Nt=15 # spatial-temporal grid sizes
print(3*Nt*2*Nx+2*2*Nx)
dx=L/Nx # spatial step size
dt=min(T/Nt,CFL*dx/abs(u1_max)) # temporal step size
# dt=min(T/Nt,CFL*dx/abs(u1_max),CFL*(dx**2)/(2*mu)) # temporal step size
# ep1=-mu*dt/(dx**2)  # rho
# ep2=mu*dt/(dx**2) # V
print('dx={dx}, dt={dt}'.format(dx=round(dx,3),dt=round(dt,3)))
x=np.linspace(0,L,Nx+1)
t=np.linspace(0,T,Nt+1)
guess0 = np.zeros(3*Nt*2*Nx+2*2*Nx)
t0 = time.process_time()   ###
sol0 = newton_krylov(F, guess0, method='lgmres', verbose=1, inner_M=get_preconditioner(guess0))
t1 = time.process_time()   ###
print("Time spent :",t1-t0)
# np.savetxt('Sol_NonSep_T1_N151.dat', sol0)
np.savetxt('Sol0_integ_2LWR_T15_N5.dat', sol0)


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
x_mfg=np.linspace(0,L,Nx+1)
t_mfg=np.linspace(0,T,Nt+1)

''' Plots '''
tt, xx = np.meshgrid(t_mfg,x_mfg)
fig1 = plt.figure(figsize=(6, 5), dpi=100)
ax1 = fig1.gca(projection='3d')
ax1.plot_surface(xx, tt, rho1_mfg, cmap=cm.viridis)
ax1.set_xlabel('$x$')
ax1.set_ylabel('$t$')
ax1.set_zlabel('cars density')
ax1.invert_xaxis()
#     ax.text2D(0.05, 0.95, text, transform=ax1.transAxes)
plt.savefig('2LWR_integ_T15_N5_1.png')

fig2 = plt.figure(figsize=(6, 5), dpi=100)
ax2 = fig2.gca(projection='3d')
ax2.plot_surface(xx, tt, rho2_mfg, cmap=cm.viridis)
ax2.set_xlabel('$x$')
ax2.set_ylabel('$t$')
ax2.set_zlabel('trucks density')
ax2.invert_xaxis()
#     ax.text2D(0.05, 0.95, text, transform=ax2.transAxes)
plt.savefig('2LWR_integ_T15_N5_2.png')

plt.figure(figsize=(20, 8))
plt.subplot(2,3,1)
plt.plot(x_mfg,rho1_mfg[:,0],'b',label='cars density')
plt.plot(x_mfg,rho2_mfg[:,0],'r',label='trucks density')
plt.plot(x_mfg,u1_mfg[:,0],'b-.',label='cars speed')
plt.plot(x_mfg,u2_mfg[:,0],'r-.',label='Trucks speed')
plt.plot(x_mfg,V1_mfg[:,0],'g',label='Optimal cost')
plt.plot(x_mfg,V2_mfg[:,0],'g-.')
plt.legend()
plt.grid()
plt.title("t=0.0")
plt.ylim(-0.1,1.0)
plt.subplot(2,3,2)
plt.plot(x_mfg,rho1_mfg[:,int(Nt/5)],'b')
plt.plot(x_mfg,rho2_mfg[:,int(Nt/5)],'r')
plt.plot(x_mfg,u1_mfg[:,int(Nt/5)],'b-.')
plt.plot(x_mfg,u2_mfg[:,int(Nt/5)],'r-.')
plt.plot(x_mfg,V1_mfg[:,int(Nt/5)],'g')
plt.plot(x_mfg,V2_mfg[:,int(Nt/5)],'g-.')
plt.grid()
#     plt.legend()
plt.title("t={t}".format(t=round(t[int(Nt/5)],3)))
#     plt.xlabel('x')
plt.ylim(-0.1,1.0)
plt.subplot(2,3,3)
plt.plot(x_mfg,rho1_mfg[:,int(2*Nt/5)],'b')
plt.plot(x_mfg,rho2_mfg[:,int(2*Nt/5)],'r')
plt.plot(x_mfg,u1_mfg[:,int(2*Nt/5)],'b-.')
plt.plot(x_mfg,u2_mfg[:,int(2*Nt/5)],'r-.')
plt.plot(x_mfg,V1_mfg[:,int(2*Nt/5)],'g')
plt.plot(x_mfg,V2_mfg[:,int(2*Nt/5)],'g-.')
plt.grid()
#     plt.legend()
plt.title("t={t}".format(t=round(t[int(2*Nt/5)],3)))
#     plt.xlabel('x')
plt.ylim(-0.1,1.0)
plt.subplot(2,3,4)
plt.plot(x_mfg,rho1_mfg[:,int(3*Nt/5)],'b')
plt.plot(x_mfg,rho2_mfg[:,int(3*Nt/5)],'r')
plt.plot(x_mfg,u1_mfg[:,int(3*Nt/5)],'b-.')
plt.plot(x_mfg,u2_mfg[:,int(3*Nt/5)],'r-.')
plt.plot(x_mfg,V1_mfg[:,int(3*Nt/5)],'g')
plt.plot(x_mfg,V2_mfg[:,int(3*Nt/5)],'g-.')
plt.grid()
#     plt.legend()
plt.title("t={t}".format(t=round(t[int(3*Nt/5)],3)))
plt.xlabel('x')
plt.ylim(-0.1,1.0)
plt.subplot(2,3,5)
plt.plot(x_mfg,rho1_mfg[:,int(4*Nt/5)],'b')
plt.plot(x_mfg,rho2_mfg[:,int(4*Nt/5)],'r')
plt.plot(x_mfg,u1_mfg[:,int(4*Nt/5)],'b-.')
plt.plot(x_mfg,u2_mfg[:,int(4*Nt/5)],'r-.')
plt.plot(x_mfg,V1_mfg[:,int(4*Nt/5)],'g')
plt.plot(x_mfg,V2_mfg[:,int(4*Nt/5)],'g-.')
plt.grid()
#     plt.legend()
plt.title("t={t}".format(t=round(t[int(4*Nt/5)],3)))
plt.xlabel('x')
plt.ylim(-0.1,1.0)
plt.subplot(2,3,6)
plt.plot(x_mfg,rho1_mfg[:,Nt-1],'b')
plt.plot(x_mfg,rho2_mfg[:,Nt-1],'r')
plt.plot(x_mfg,u1_mfg[:,Nt-1],'b-.')
plt.plot(x_mfg,u2_mfg[:,Nt-1],'r-.')
plt.plot(x_mfg,V1_mfg[:,Nt-1],'g')
plt.plot(x_mfg,V2_mfg[:,Nt-1],'g-.')
plt.grid()
#     plt.legend()
plt.title("t={t}".format(t=round(t[Nt-1],3)))
plt.xlabel('x')
plt.ylim(-0.1,1.0)
plt.savefig('2LWR_integ_T15_N5_3.png')

plt.figure()
plt.scatter(rho1_mfg[:,Nt-1],Q1_mfg[:,Nt-1],s=10,label='cars')
plt.scatter(rho2_mfg[:,Nt-1],Q2_mfg[:,Nt-1],s=10,label='trucks')
plt.xlabel('density')
plt.ylabel('Flow')
plt.grid()
plt.title("Fundamental diagram (T={T})".format(T=round(t[Nt-1],3)))
plt.legend()
plt.savefig('2LWR_integ_T15_N5_4.png')




