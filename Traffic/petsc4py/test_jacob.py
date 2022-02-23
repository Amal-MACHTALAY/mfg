#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:50:41 2022

@author: amal
"""

import numpy as np
from scipy import integrate
# from scipy.optimize.nonlin import newton_krylov
# import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix
# import time
# import os

''' inputs '''
T=3.0 # horizon length
L=1  #L=N # road length 
u_max=1.0 # free flow speed
rho_jam=1.0 # jam density
CFL=0.75    # CFL<1
rho_a=0.05; rho_b=0.95; gama=0.1
# rho_a=0.2; rho_b=0.8; gama=0.15*L
# """ Viscosity condition"""
EPS=0.45 

''' functions '''

######################LWR
def U(rho): # Greenshields desired speed
    return u_max*(1-rho/rho_jam)

def f_mfg_LWR(u,r):
    return 0.5*((U(r)-u)**2) # MFG-LWR

def f_star_p(p,r): # 0<=u<=u_max
    return U(r)-p # MFG-LWR

def f_star(p,r): # p=Vx
    return -0.5*(p**2)+U(r)*p # MFG-LWR

def f_star_p_der_arg1(P,r):
    return -1
    
def f_star_p_der_arg2(P,r):   
    return -(u_max/rho_jam)

def f_star_der_arg1(P,r):
    return -P+U(r)

def f_star_der_arg2(P,r):
    return -(u_max/rho_jam)*P


#######################################
def integral(a,b): 
    x2 = lambda x: rho_int(x)
    I=integrate.quad(x2, a, b)
    return I[0]

def rho_int(s): # initial density
    return rho_a+(rho_b-rho_a)*np.exp(-0.5*((s-0.5*L)/gama)**2) # 0<=rho<=rho_jam

def VT(a): # Terminal cost
    return 0.0

def r_idx(j,n):
    return (j-1)*(Nt+1)+n

def u_idx(j,n):
    return (Nt+1)*Nx+(j-1)*Nt+n

def V_idx(j,n):
    return (2*Nt+1)*Nx+(j-1)*(Nt+1)+n

def Fr_idx(j,n):
    return (j-1)*Nt+n

def Fu_idx(j,n):
    return Nt*Nx+(j-1)*Nt+n

def FV_idx(j,n):
    return 2*Nt*Nx+(j-1)*Nt+n

def Frint_idx(j):
    return 3*Nt*Nx+(j-1)

def FVter_idx(j):
    return 3*Nt*Nx+Nx+(j-1)


def Fction(w):
    # FF=[F_rho,F_u,F_V,F_rho_int,F_V_ter], F_rho:0->Nt*Nx-1, F_u:Nt*Nx->2*Nt*Nx-1, F_V:2*Nt*Nx->3*Nt*Nx-1, 
    # F_rho_int:3*Nt*Nx->3*Nt*Nx+Nx-1, F_V_ter:3*Nt*Nx+Nx->3*Nt*Nx+2*Nx-1
    FF=np.zeros(3*Nt*Nx+2*Nx)
    for n in range(0,Nt):
        # F_rho , F[0]->F[Nt-1] ************** 1  
        FF[Fr_idx(1,n)]=w[r_idx(1,n+1)]-0.5*(w[r_idx(Nx,n)]+w[r_idx(2,n)])\
            +(0.5*dt/dx)*(w[r_idx(2,n)]*w[u_idx(2,n)]-w[r_idx(Nx,n)]*w[u_idx(Nx,n)])
        # F_rho , F[Nt*Nx-Nt]->F[Nt*Nx-1] ********** 3 
        FF[Fr_idx(Nx,n)]=w[r_idx(Nx,n+1)]-0.5*(w[r_idx(Nx-1,n)]+w[r_idx(1,n)])\
            +(0.5*dt/dx)*(w[r_idx(1,n)]*w[u_idx(1,n)]-w[r_idx(Nx-1,n)]*w[u_idx(Nx-1,n)])
        # F_u , F[Nt*Nx]->F[Nt*Nx+Nt-1] *********** 4 
        FF[Fu_idx(1,n)]=w[u_idx(1,n)]-f_star_p((w[V_idx(1,n+1)]-w[V_idx(Nx,n+1)])/dx,w[r_idx(1,n)])
        # F_u , F[2*Nt*Nx-Nt]->F[2*Nt*Nx-1] ********* 6 
        FF[Fu_idx(Nx,n)]=w[u_idx(Nx,n)]-f_star_p((w[V_idx(Nx,n+1)]-w[V_idx(Nx-1,n+1)])/dx,w[r_idx(Nx,n)])
        # F_V , F[2*Nt*Nx]->F[2*Nt*Nx+Nt-1] *********** 7 
        FF[FV_idx(1,n)]=w[V_idx(1,n+1)]-w[V_idx(1,n)]+dt*f_star((w[V_idx(1,n+1)]-w[V_idx(Nx,n+1)])/dx,w[r_idx(1,n)])\
            +eps*(w[V_idx(2,n+1)]-2*w[V_idx(1,n+1)]+w[V_idx(Nx,n+1)])
        # F_V , F[3*Nt*Nx-Nt]->F[3*Nt*Nx-1] ********** 9 
        FF[FV_idx(Nx,n)]=w[V_idx(Nx,n+1)]-w[V_idx(Nx,n)]+dt*f_star((w[V_idx(Nx,n+1)]-w[V_idx(Nx-1,n+1)])/dx,w[r_idx(Nx,n)])\
            +eps*(w[V_idx(1,n+1)]-2*w[V_idx(Nx,n+1)]+w[V_idx(Nx-1,n+1)])
    for j in range(2,Nx):
        for n in range(0,Nt):
            # F_rho , F[Nt]->F[Nt*Nx-Nt-1] ************ 2 
            FF[Fr_idx(j,n)]=w[r_idx(j,n+1)]-0.5*(w[r_idx(j-1,n)]+w[r_idx(j+1,n)])\
                +(0.5*dt/dx)*(w[r_idx(j+1,n)]*w[u_idx(j+1,n)]-w[r_idx(j-1,n)]*w[u_idx(j-1,n)])
            # F_u , F[Nt*Nx+Nt]->F[2*Nt*Nx-Nt-1] *********** 5 
            FF[Fu_idx(j,n)]=w[u_idx(j,n)]-f_star_p((w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)])
            # F_V , F[2*Nt*Nx+Nt]->F[3*Nt*Nx-Nt-1] ********* 8 
            FF[FV_idx(j,n)]=w[V_idx(j,n+1)]-w[V_idx(j,n)]+dt*f_star((w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)])\
                +eps*(w[V_idx(j+1,n+1)]-2*w[V_idx(j,n+1)]+w[V_idx(j-1,n+1)])
        # F_rho_int , F[3*Nt*Nx+1]->F[3*Nt*Nx+Nx-2] ********** 11
        FF[Frint_idx(j)]=w[r_idx(j,0)]-(1/dx)*integral(x[j-1],x[j])
        # F_V_ter , F[3*Nt*Nx+Nx+1]->F[3*Nt*Nx+2*Nx-2] ********* 14
        FF[FVter_idx(j)]=w[V_idx(j,Nt)]-VT(x[j])
    # F_rho_int , F[3*Nt*Nx] ********* 10
    FF[Frint_idx(1)]=w[r_idx(1,0)]-(1/dx)*integral(x[0],x[1])
    # F_rho_int , F[3*Nt*Nx+Nx-1] ********* 12
    FF[Frint_idx(Nx)]=w[r_idx(Nx,0)]-(1/dx)*integral(x[Nx-1],x[Nx])
    # F_V_ter , F[3*Nt*Nx+Nx] *********** 13 
    FF[FVter_idx(1)]=w[V_idx(1,Nt)]-VT(x[1])
    # F_V_ter , F[3*Nt*Nx+2*Nx-1] ************** 15
    FF[FVter_idx(Nx)]=w[V_idx(Nx,Nt)]-VT(x[Nx])
    
    return FF




def jacobian_mat(w,row,col,data): # Ignoring the forward-backward coupling  parts
    # J=np.zeros((3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx))
    cmpt = 0
    row[:] = 0; col[:] = 0; data[:] = 0.
    # row = []; col = []; data = []
    for n in range(0,Nt):
        for j in range(1,Nx+1): # 1,Nx-1
            # J[Fr_idx(j,n),r_idx(j,n+1)]=1 # F_rho - rho  ## Ok
            row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(j,n+1); data[cmpt]=1
            cmpt +=1
            # J[Fu_idx(j,n),u_idx(j,n)]=1 # F_u - u  ## Ok
            row[cmpt]=Fu_idx(j,n); col[cmpt]=u_idx(j,n); data[cmpt]=1
            cmpt +=1
            # J[FV_idx(j,n),V_idx(j,n)]=-1 # F_V - V  ## Ok
            row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(j,n); data[cmpt]=-1
            cmpt +=1
            # J[FV_idx(j,n),V_idx(j,n+1)]=1-2*eps # F_V - V  ## Ok
            row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(j,n+1); data[cmpt]=1-2*eps+(dt/dx)*f_star_der_arg1((w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)])
            cmpt +=1
            if j!=1:
                # J[Fr_idx(j,n),r_idx(j-1,n)]=-(0.5*dt/dx)*w[u_idx(j-1,n)]-0.5 # F_rho -rho  ## Ok
                row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(j-1,n); data[cmpt]=-(0.5*dt/dx)*w[u_idx(j-1,n)]-0.5
                cmpt +=1
                # J[Fr_idx(j,n),u_idx(j-1,n)]=-(0.5*dt/dx)*w[r_idx(j-1,n)] # F_rho - u  ## Ok
                row[cmpt]=Fr_idx(j,n); col[cmpt]=u_idx(j-1,n); data[cmpt]=-(0.5*dt/dx)*w[r_idx(j-1,n)]
                cmpt +=1
                # J[FV_idx(j,n),V_idx(j-1,n+1)]=eps # F_V - V   ## Ok
                row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(j-1,n+1); data[cmpt]=eps-(dt/dx)*f_star_der_arg1((w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)])
                cmpt +=1
                #
                row[cmpt]=FV_idx(j,n); col[cmpt]=r_idx(j,n); data[cmpt]=dt*f_star_der_arg2((w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)])
                cmpt +=1
                #
                row[cmpt]=Fu_idx(j,n); col[cmpt]=V_idx(j,n+1); data[cmpt]=-(1/dx)*f_star_p_der_arg1((w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)])
                cmpt +=1
                #
                row[cmpt]=Fu_idx(j,n); col[cmpt]=V_idx(j-1,n+1); data[cmpt]=(1/dx)*f_star_p_der_arg1((w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)])
                cmpt +=1
                #
                row[cmpt]=Fu_idx(j,n); col[cmpt]=r_idx(j,n); data[cmpt]=-f_star_p_der_arg2((w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)])
                cmpt +=1
            if j==1:
                # J[Fr_idx(j,n),r_idx(Nx,n)]=(0.5*dt/dx)*w[u_idx(Nx,n)]-0.5 # F_rho - rho
                row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(Nx,n); data[cmpt]=(0.5*dt/dx)*w[u_idx(Nx,n)]-0.5
                cmpt +=1
                # J[Fr_idx(j,n),u_idx(Nx,n)]=-(0.5*dt/dx)*w[r_idx(Nx,n)] # F_rho - u
                row[cmpt]=Fr_idx(j,n); col[cmpt]=u_idx(Nx,n); data[cmpt]=-(0.5*dt/dx)*w[r_idx(Nx,n)]
                cmpt +=1
                # J[FV_idx(j,n),V_idx(Nx,n+1)]=eps # F_V - V
                row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(Nx,n+1); data[cmpt]=eps-(dt/dx)*f_star_der_arg1((w[V_idx(j,n+1)]-w[V_idx(Nx,n+1)])/dx,w[r_idx(j,n)])
                cmpt +=1
                #
                row[cmpt]=FV_idx(j,n); col[cmpt]=r_idx(j,n); data[cmpt]=dt*f_star_der_arg2((w[V_idx(j,n+1)]-w[V_idx(Nx,n+1)])/dx,w[r_idx(j,n)])
                cmpt +=1
                #
                row[cmpt]=Fu_idx(j,n); col[cmpt]=V_idx(j,n+1); data[cmpt]=-(1/dx)*f_star_p_der_arg1((w[V_idx(j,n+1)]-w[V_idx(Nx,n+1)])/dx,w[r_idx(j,n)])
                cmpt +=1
                #
                row[cmpt]=Fu_idx(j,n); col[cmpt]=V_idx(Nx,n+1); data[cmpt]=(1/dx)*f_star_p_der_arg1((w[V_idx(j,n+1)]-w[V_idx(Nx,n+1)])/dx,w[r_idx(j,n)])
                cmpt +=1
                #
                row[cmpt]=Fu_idx(j,n); col[cmpt]=r_idx(j,n); data[cmpt]=-f_star_p_der_arg2((w[V_idx(j,n+1)]-w[V_idx(Nx,n+1)])/dx,w[r_idx(j,n)])
                cmpt +=1
            if j!=Nx:
                # J[Fr_idx(j,n),r_idx(j+1,n)]=(0.5*dt/dx)*w[u_idx(j+1,n)]-0.5 # F_rho -rho  ## Ok
                row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(j+1,n); data[cmpt]=(0.5*dt/dx)*w[u_idx(j+1,n)]-0.5
                cmpt +=1
                # J[Fr_idx(j,n),u_idx(j+1,n)]=(0.5*dt/dx)*w[r_idx(j+1,n)] # F_rho - u ## Ok
                row[cmpt]=Fr_idx(j,n); col[cmpt]=u_idx(j+1,n); data[cmpt]=(0.5*dt/dx)*w[r_idx(j+1,n)]
                cmpt +=1
                # J[FV_idx(j,n),V_idx(j+1,n+1)]=eps # F_V - V  ## Ok
                row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(j+1,n+1); data[cmpt]=eps
                cmpt +=1
            if j==Nx:
                # J[Fr_idx(j,n),r_idx(1,n)]=(0.5*dt/dx)*w[u_idx(1,n)]-0.5 # F_rho - rho
                row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(1,n); data[cmpt]=(0.5*dt/dx)*w[u_idx(1,n)]-0.5
                cmpt +=1
                # J[Fr_idx(j,n),u_idx(1,n)]=(0.5*dt/dx)*w[r_idx(1,n)] # F_rho - u
                row[cmpt]=Fr_idx(j,n); col[cmpt]=u_idx(1,n); data[cmpt]=(0.5*dt/dx)*w[r_idx(1,n)]
                cmpt +=1
                # J[FV_idx(j,n),V_idx(1,n+1)]=eps # F_V - V
                row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(1,n+1); data[cmpt]=eps
                cmpt +=1
  
    for j in range(1,Nx+1):
        # J[Frint_idx(j),r_idx(j,0)]=1 # F_rho_int - rho  ## Ok
        row[cmpt]=Frint_idx(j); col[cmpt]=r_idx(j,0); data[cmpt]=1.
        cmpt +=1
        # J[FVter_idx(j),V_idx(j,Nt)]=1 # F_V_ter - V ## Ok
        row[cmpt]=FVter_idx(j); col[cmpt]=V_idx(j,Nt); data[cmpt]=1.
        cmpt +=1
    
    # return J
    return 0


def compute_jacob(a):
    row=np.zeros(14*Nt*Nx+2*Nx)
    col=np.zeros(14*Nt*Nx+2*Nx)
    data=np.zeros(14*Nt*Nx+2*Nx)
    jacobian_mat(a,row, col, data)
    shap=(3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx)
    Jac1 = csc_matrix((data, (row, col)),shape = shap)
    print(Jac1)
    return 0


mu=0.1

Nx=5; Nt=10 # spatial-temporal grid sizes
dx=L/Nx # spatial step size
if mu==0.0:
    dt=min(T/Nt,(CFL*dx)/u_max) # temporal step size
    eps=0.0
else:
    dt=min(T/Nt,CFL*dx/abs(u_max),EPS*(dx**2)/mu) # temporal step size
    eps=mu*dt/(dx**2) # V
x=np.linspace(0,L,Nx+1)
t=np.arange(0,T+dt,dt)
Nt=int((len(t)-1))
print('Nx={Nx}, Nt={Nt}, nu={nu}'.format(Nx=Nx,Nt=Nt,nu=mu))
print('dx={dx}, dt={dt}'.format(dx=round(dx,4),dt=round(dt,4)))

ww=np.linspace(1,3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx)
# print(ww,3*Nt*Nx+2*Nx)
compute_jacob(ww)


#############################numerical jacobian
import numdifftools as nd
jac_mat=nd.Jacobian(Fction)
jac=jac_mat(ww)
jac2 = csc_matrix(jac)
print('*********************************************************************')
print(jac2)
