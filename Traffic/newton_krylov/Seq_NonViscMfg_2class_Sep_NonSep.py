##!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:43:29 2021

@author: amal
"""


import numpy as np
from scipy import integrate
from scipy.optimize.nonlin import newton_krylov
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix
import time

''' population 1 : cars
    population 2 : trucks '''
T=3.0 # horizon length 
l1=1; l2=2; l=l1+l2 # average length of the vehicles in the j-th population [m]
rho1_jam=1; rho2_jam=0.5 # jam density [vehicles/m]
u1_max=1; u2_max=0.5 # free flow speed [*100 km/h]
L=2 # road length [m]
### l1*rho1_jam = l2*rho2_jam = 1
CFL=0.75    # CFL<1
# """ Viscosity"""
EPS=0.45

''' functions '''
''' 2-MFG-Non Separable '''
def f_mfg_2NonSep(u,u_max,rho1,rho2):
    # return 0.5*(u/u_max)**2-u/u_max+((rho1/rho1_jam)+(rho2/rho2_jam))*(u/u_max)
    return 0.5*(u/u_max)**2-u/u_max+((rho1*l1+rho2*l2)/(rho1_jam*l1+rho2_jam*l2))*(u/u_max)

def f_star_p_2NonSep(u_max,p,rho1,rho2):
    # return max(min(u_max*(1-(rho1/rho1_jam)-(rho2/rho2_jam)-u_max*p),u_max),0)
    return max(min(u_max*(1-(rho1*l1+rho2*l2)/(rho1_jam*l1+rho2_jam*l2)-u_max*p),u_max),0)

def f_star_2NonSep(u_max,u,p,rho1,rho2): # p=Vx
    return f_mfg_2NonSep(u,u_max,rho1,rho2)+u*p

''' 2-MFG-Separable '''
def f_mfg_2Sep(u,u_max,rho1,rho2):
    # return 0.5*(u/u_max)**2-u/u_max+((rho1/rho1_jam)+(rho2/rho2_jam))
    return 0.5*(u/u_max)**2-u/u_max+((rho1*l1+rho2*l2)/(rho1_jam*l1+rho2_jam*l2))

def f_star_p_2Sep(u_max,p,rho1,rho2):
    # return max(min(u_max*(1-u_max*p),u_max),0)
    return max(min(u_max*(1-u_max*p),u_max),0)

def f_star_2Sep(u_max,u,p,rho1,rho2): # p=Vx
    return f_mfg_2Sep(u,u_max,rho1,rho2)+u*p

''' Fully segregated TC initial density '''
sigma=0.15
def rho1_int(s): # initial density
    rho_a=0.0; rho_b=1.0 # cars
    return rho_a+(rho_b-rho_a)*np.exp(-0.5*(((s-3*L/4)/sigma)**2))
                
def rho2_int(s): # initial density
    rho_a=0.0; rho_b=0.5 # Trucks
    return rho_a+(rho_b-rho_a)*np.exp(-0.5*(((s-L/4)/sigma)**2))

''' Fully segregated CT initial density '''
# sigma=0.15
# def rho1_int(s): # initial density
#     rho_a=0.0; rho_b=1.0 # cars
#     return rho_a+(rho_b-rho_a)*np.exp(-0.5*(((s-L/4)/sigma)**2))
                
# def rho2_int(s): # initial density
#     rho_a=0.0; rho_b=0.5 # Trucks
#     return rho_a+(rho_b-rho_a)*np.exp(-0.5*(((s-3*L/4)/sigma)**2))


''' Integrated initial TCT density'''
# sigma=0.15
# def rho1_int(s): # initial density
#     rho_a=0.0; rho_b=1.0 # cars
#     ro=rho_a
#     for j in range(L):
#         if j%2!=0:
#             ro=ro+(rho_b-rho_a)*np.exp(-0.5*(((s-(2*j+1)/2)/sigma)**2))
#     return ro         
# def rho2_int(s): # initial density
#     rho_a=0.0; rho_b=0.5 # Trucks
#     ro=rho_a
#     for j in range(L):
#         if j%2==0:
#             ro=ro+(rho_b-rho_a)*np.exp(-0.5*(((s-(2*j+1)/2)/sigma)**2))
#     return ro


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


def r_idx(j,n):
    return (j-1)*(Nt+1)+n

def u_idx(j,n):
    return (Nt+1)*(2*Nx)+(j-1)*Nt+n

def V_idx(j,n):
    return (2*Nt+1)*(2*Nx)+(j-1)*(Nt+1)+n

def Fr_idx(j,n):
    return (j-1)*Nt+n

def Fu_idx(j,n):
    return Nt*(2*Nx)+(j-1)*Nt+n

def FV_idx(j,n):
    return 2*Nt*(2*Nx)+(j-1)*Nt+n

def Frint_idx(j):
    return 3*Nt*(2*Nx)+(j-1)

def FVter_idx(j):
    return (3*Nt+1)*(2*Nx)+(j-1)


def Fct(w,f_star_p,f_star):
    # FF=[F_rho,F_u,F_V,F_rho_int,F_V_ter], F_rho:0->Nt*2*Nx-1, F_u:Nt*2*Nx->2*Nt*2*Nx-1, F_V:2*Nt*2*Nx->3*Nt*2*Nx-1, F_rho_int:3*Nt*2*Nx->3*Nt*2*Nx+2*Nx-1, F_V_ter:3*Nt*2*Nx+2*Nx->3*Nt*2*Nx+2*2*Nx-1
    FF=np.zeros(3*Nt*(2*Nx)+2*(2*Nx))
    for n in range(0,Nt):
        # F_rho1 , F[0]->F[Nt-1] ************** 1  
        FF[Fr_idx(1,n)]=w[r_idx(1,n+1)]-0.5*(w[r_idx(Nx,n)]+w[r_idx(2,n)])\
            +(0.5*dt/dx)*(w[r_idx(2,n)]*w[u_idx(2,n)]-w[r_idx(Nx,n)]*w[u_idx(Nx,n)])
        # F_rho2 , F[Nt*Nx]->F[Nt*Nx+Nt-1] ************** 4  
        FF[Fr_idx(Nx+1,n)]=w[r_idx(Nx+1,n+1)]-0.5*(w[r_idx(Nx+Nx,n)]+w[r_idx(Nx+2,n)])\
            +(0.5*dt/dx)*(w[r_idx(Nx+2,n)]*w[u_idx(Nx+2,n)]-w[r_idx(Nx+Nx,n)]*w[u_idx(Nx+Nx,n)])
        # F_rho1 , F[Nt*Nx-Nt]->F[Nt*Nx-1] ********** 3 
        FF[Fr_idx(Nx,n)]=w[r_idx(Nx,n+1)]-0.5*(w[r_idx(Nx-1,n)]+w[r_idx(1,n)])\
            +(0.5*dt/dx)*(w[r_idx(1,n)]*w[u_idx(1,n)]-w[r_idx(Nx-1,n)]*w[u_idx(Nx-1,n)])
        # F_rho2 , F[2*Nt*Nx-Nt]->F[2*Nt*Nx-1] ********** 6 
        FF[Fr_idx(Nx+Nx,n)]=w[r_idx(Nx+Nx,n+1)]-0.5*(w[r_idx(Nx+Nx-1,n)]+w[r_idx(Nx+1,n)])\
            +(0.5*dt/dx)*(w[r_idx(Nx+1,n)]*w[u_idx(Nx+1,n)]-w[r_idx(Nx+Nx-1,n)]*w[u_idx(Nx+Nx-1,n)])
        # F_u1 , F[2*Nt*Nx]->F[2*Nt*Nx+Nt-1] *********** 7 
        FF[Fu_idx(1,n)]=w[u_idx(1,n)]-f_star_p(u1_max,(w[V_idx(1,n+1)]-w[V_idx(Nx,n+1)])/dx,w[r_idx(1,n)],w[r_idx(Nx+1,n)])
        # F_u2 , F[3*Nt*Nx]->F[3*Nt*Nx+Nt-1] *********** 10 
        FF[Fu_idx(Nx+1,n)]=w[u_idx(Nx+1,n)]-f_star_p(u2_max,(w[V_idx(Nx+1,n+1)]-w[V_idx(Nx+Nx,n+1)])/dx,w[r_idx(1,n)],w[r_idx(Nx+1,n)])
        # F_u1 , F[3*Nt*Nx-Nt]->F[3*Nt*Nx-1] ********* 9 
        FF[Fu_idx(Nx,n)]=w[u_idx(Nx,n)]-f_star_p(u1_max,(w[V_idx(Nx,n+1)]-w[V_idx(Nx-1,n+1)])/dx,w[r_idx(Nx,n)],w[r_idx(Nx+Nx,n)])
        # F_u2 , F[4*Nt*Nx-Nt]->F[4*Nt*Nx-1] ********* 12 
        FF[Fu_idx(Nx+Nx,n)]=w[u_idx(Nx+Nx,n)]-f_star_p(u2_max,(w[V_idx(Nx+Nx,n+1)]-w[V_idx(Nx+Nx-1,n+1)])/dx,w[r_idx(Nx,n)],w[r_idx(Nx+Nx,n)])
        # F_V1 , F[4*Nt*Nx]->F[4*Nt*Nx+Nt-1] *********** 13 
        FF[FV_idx(1,n)]=w[V_idx(1,n+1)]-w[V_idx(1,n)]+dt*f_star(u1_max,w[u_idx(1,n)],(w[V_idx(1,n+1)]-w[V_idx(Nx,n+1)])/dx,w[r_idx(1,n)],w[r_idx(Nx+1,n)])\
            +eps*(w[V_idx(2,n+1)]-2*w[V_idx(1,n+1)]+w[V_idx(Nx,n+1)])
        # F_V2 , F[5*Nt*Nx]->F[5*Nt*Nx+Nt-1] *********** 16
        FF[FV_idx(Nx+1,n)]=w[V_idx(Nx+1,n+1)]-w[V_idx(Nx+1,n)]+dt*f_star(u2_max,w[u_idx(Nx+1,n)],(w[V_idx(Nx+1,n+1)]-w[V_idx(Nx+Nx,n+1)])/dx,w[r_idx(1,n)],w[r_idx(Nx+1,n)])\
            +eps*(w[V_idx(Nx+2,n+1)]-2*w[V_idx(Nx+1,n+1)]+w[V_idx(Nx+Nx,n+1)])
        # F_V1 , F[5*Nt*Nx-Nt]->F[5*Nt*Nx-1] ********** 15 
        FF[FV_idx(Nx,n)]=w[V_idx(Nx,n+1)]-w[V_idx(Nx,n)]+dt*f_star(u1_max,w[u_idx(Nx,n)],(w[V_idx(Nx,n+1)]-w[V_idx(Nx-1,n+1)])/dx,w[r_idx(Nx,n)],w[r_idx(Nx+Nx,n)])\
            +eps*(w[V_idx(1,n+1)]-2*w[V_idx(Nx,n+1)]+w[V_idx(Nx-1,n+1)])
        # F_V2 , F[6*Nt*Nx-Nt]->F[6*Nt*Nx-1] ********** 18 
        FF[FV_idx(Nx+Nx,n)]=w[V_idx(Nx+Nx,n+1)]-w[V_idx(Nx+Nx,n)]+dt*f_star(u2_max,w[u_idx(Nx+Nx,n)],(w[V_idx(Nx+Nx,n+1)]-w[V_idx(Nx+Nx-1,n+1)])/dx,w[r_idx(Nx,n)],w[r_idx(Nx+Nx,n)])\
            +eps*(w[V_idx(Nx+1,n+1)]-2*w[V_idx(Nx+Nx,n+1)]+w[V_idx(Nx+Nx-1,n+1)])
    for j in range(2,Nx):
        for n in range(0,Nt):
            # F_rho1 , F[Nt]->F[Nt*Nx-Nt-1] ************ 2 
            FF[Fr_idx(j,n)]=w[r_idx(j,n+1)]-0.5*(w[r_idx(j-1,n)]+w[r_idx(j+1,n)])\
                +(0.5*dt/dx)*(w[r_idx(j+1,n)]*w[u_idx(j+1,n)]-w[r_idx(j-1,n)]*w[u_idx(j-1,n)])
            # F_rho2 , F[Nt*Nx+Nt]->F[2*Nt*Nx-Nt-1] ************ 5 
            FF[Fr_idx(Nx+j,n)]=w[r_idx(Nx+j,n+1)]-0.5*(w[r_idx(Nx+j-1,n)]+w[r_idx(Nx+j+1,n)])\
                +(0.5*dt/dx)*(w[r_idx(Nx+j+1,n)]*w[u_idx(Nx+j+1,n)]-w[r_idx(Nx+j-1,n)]*w[u_idx(Nx+j-1,n)])
            # F_u1 , F[2*Nt*Nx+Nt]->F[3*Nt*Nx-Nt-1] *********** 8 
            FF[Fu_idx(j,n)]=w[u_idx(j,n)]-f_star_p(u1_max,(w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)],w[r_idx(Nx+j,n)])
            # F_u2 , F[3*Nt*Nx+Nt]->F[4*Nt*Nx-Nt-1] *********** 11 
            FF[Fu_idx(Nx+j,n)]=w[u_idx(Nx+j,n)]-f_star_p(u2_max,(w[V_idx(Nx+j,n+1)]-w[V_idx(Nx+j-1,n+1)])/dx,w[r_idx(j,n)],w[r_idx(Nx+j,n)])
            # F_V1 , F[4*Nt*Nx+Nt]->F[5*Nt*Nx-Nt-1] ********* 14 
            FF[FV_idx(j,n)]=w[V_idx(j,n+1)]-w[V_idx(j,n)]+dt*f_star(u1_max,w[u_idx(j,n)],(w[V_idx(j,n+1)]-w[V_idx(j-1,n+1)])/dx,w[r_idx(j,n)],w[r_idx(Nx+j,n)])\
                +eps*(w[V_idx(j+1,n+1)]-2*w[V_idx(j,n+1)]+w[V_idx(j-1,n+1)])
            # F_V2 , F[5*Nt*Nx+Nt]->F[6*Nt*Nx-Nt-1] ********* 17 
            FF[FV_idx(Nx+j,n)]=w[V_idx(Nx+j,n+1)]-w[V_idx(Nx+j,n)]+dt*f_star(u2_max,w[u_idx(Nx+j,n)],(w[V_idx(Nx+j,n+1)]-w[V_idx(Nx+j-1,n+1)])/dx,w[r_idx(j,n)],w[r_idx(Nx+j,n)])\
                +eps*(w[V_idx(Nx+j+1,n+1)]-2*w[V_idx(Nx+j,n+1)]+w[V_idx(Nx+j-1,n+1)])
        # F_rho_int1 , F[6*Nt*Nx+1]->F[6*Nt*Nx+Nx-2] ********** 20
        FF[Frint_idx(j)]=w[r_idx(j,0)]-(1/dx)*integral(1,x[j-1],x[j])
        # F_rho_int2 , F[6*Nt*Nx+Nx+1]->F[6*Nt*Nx+2*Nx-2] ********** 23
        FF[Frint_idx(Nx+j)]=w[r_idx(Nx+j,0)]-(1/dx)*integral(2,x[j-1],x[j])
        # F_V_ter1 , F[6*Nt*Nx+2*Nx+1]->F[6*Nt*Nx+3*Nx-2] ********* 26
        FF[FVter_idx(j)]=w[V_idx(j,Nt)]-VT(x[j])
        # F_V_ter2 , F[6*Nt*Nx+3*Nx+1]->F[6*Nt*Nx+4*Nx-2] ********* 29
        FF[FVter_idx(Nx+j)]=w[V_idx(Nx+j,Nt)]-VT(x[j])
    # F_rho_int1 , F[6*Nt*Nx] ********* 19
    FF[Frint_idx(1)]=w[r_idx(1,0)]-(1/dx)*integral(1,x[0],x[1])
    # F_rho_int2 , F[6*Nt*Nx+Nx] ********* 22
    FF[Frint_idx(Nx+1)]=w[r_idx(Nx+1,0)]-(1/dx)*integral(2,x[0],x[1])
    # F_rho_int1 , F[6*Nt*Nx+Nx-1] ********* 21
    FF[Frint_idx(Nx)]=w[r_idx(Nx,0)]-(1/dx)*integral(1,x[Nx-1],x[Nx])
    # F_rho_int2 , F[6*Nt*Nx+2*Nx-1] ********* 24
    FF[Frint_idx(Nx+Nx)]=w[r_idx(Nx+Nx,0)]-(1/dx)*integral(2,x[Nx-1],x[Nx])
    # F_V_ter1 , F[6*Nt*Nx+2*Nx] *********** 25 
    FF[FVter_idx(1)]=w[V_idx(1,Nt)]-VT(x[1])
    # F_V_ter2 , F[6*Nt*Nx+3*Nx] *********** 28 
    FF[FVter_idx(Nx+1)]=w[V_idx(Nx+1,Nt)]-VT(x[1])
    # F_V_ter1 , F[6*Nt*Nx+3*Nx-1] ************** 27
    FF[FVter_idx(Nx)]=w[V_idx(Nx,Nt)]-VT(x[Nx])
    # F_V_ter2 , F[6*Nt*Nx+4*Nx-1] ************** 30
    FF[FVter_idx(Nx+Nx)]=w[V_idx(Nx+Nx,Nt)]-VT(x[Nx])

    return FF


def jacobian(w,row,col,data): # Ignoring the forward-backward coupling  parts
    # J=np.zeros((3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx))
    cmpt = 0
    row[:] = 0; col[:] = 0; data[:] = 0.
    for n in range(0,Nt):
        for j in range(1,Nx+1): # 1,Nx-1
            # 1 J[Fr_idx(j,n),r_idx(j,n+1)]=1 # F_rho - rho  ## Ok
            row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(j,n+1); data[cmpt]=1
            cmpt +=1
            # 2 J[Fr_idx(Nx+j,n),r_idx(Nx+j,n+1)]=1 # F_rho - rho  ## Ok
            row[cmpt]=Fr_idx(Nx+j,n); col[cmpt]=r_idx(Nx+j,n+1); data[cmpt]=1
            cmpt +=1
            # 1 J[Fu_idx(j,n),u_idx(j,n)]=1 # F_u - u  ## Ok
            row[cmpt]=Fu_idx(j,n); col[cmpt]=u_idx(j,n); data[cmpt]=1
            cmpt +=1
            # 2 J[Fu_idx(Nx+j,n),u_idx(Nx+j,n)]=1 # F_u - u  ## Ok
            row[cmpt]=Fu_idx(Nx+j,n); col[cmpt]=u_idx(Nx+j,n); data[cmpt]=1
            cmpt +=1
            # 1 J[FV_idx(j,n),V_idx(j,n)]=-1 # F_V - V  ## Ok
            row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(j,n); data[cmpt]=-1
            cmpt +=1
            # 2 J[FV_idx(Nx+j,n),V_idx(Nx+j,n)]=-1 # F_V - V  ## Ok
            row[cmpt]=FV_idx(Nx+j,n); col[cmpt]=V_idx(Nx+j,n); data[cmpt]=-1
            cmpt +=1
            # 1 J[FV_idx(j,n),V_idx(j,n+1)]=1-2*eps # F_V - V  ## Ok
            row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(j,n+1); data[cmpt]=1-2*eps
            cmpt +=1
            # 2 J[FV_idx(Nx+j,n),V_idx(Nx+j,n+1)]=1-2*eps # F_V - V  ## Ok
            row[cmpt]=FV_idx(Nx+j,n); col[cmpt]=V_idx(Nx+j,n+1); data[cmpt]=1-2*eps
            cmpt +=1
            if j!=1:
                # 1 J[Fr_idx(j,n),r_idx(j-1,n)]=-(0.5*dt/dx)*w[u_idx(j-1,n)]-0.5 # F_rho -rho  ## Ok
                row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(j-1,n); data[cmpt]=-(0.5*dt/dx)*w[u_idx(j-1,n)]-0.5
                cmpt +=1
                # 2 J[Fr_idx(Nx+j,n),r_idx(Nx+j-1,n)]=-(0.5*dt/dx)*w[u_idx(Nx+j-1,n)]-0.5 # F_rho -rho  ## Ok
                row[cmpt]=Fr_idx(Nx+j,n); col[cmpt]=r_idx(Nx+j-1,n); data[cmpt]=-(0.5*dt/dx)*w[u_idx(Nx+j-1,n)]-0.5
                cmpt +=1
                # 1 J[Fr_idx(j,n),u_idx(j-1,n)]=-(0.5*dt/dx)*w[r_idx(j-1,n)] # F_rho - u  ## Ok
                row[cmpt]=Fr_idx(j,n); col[cmpt]=u_idx(j-1,n); data[cmpt]=-(0.5*dt/dx)*w[r_idx(j-1,n)]
                cmpt +=1
                # 2 J[Fr_idx(Nx+j,n),u_idx(Nx+j-1,n)]=-(0.5*dt/dx)*w[r_idx(Nx+j-1,n)] # F_rho - u  ## Ok
                row[cmpt]=Fr_idx(Nx+j,n); col[cmpt]=u_idx(Nx+j-1,n); data[cmpt]=-(0.5*dt/dx)*w[r_idx(Nx+j-1,n)]
                cmpt +=1
                # 1 J[FV_idx(j,n),V_idx(j-1,n+1)]=eps # F_V - V   ## Ok
                row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(j-1,n+1); data[cmpt]=eps
                cmpt +=1
                # 2 J[FV_idx(Nx+j,n),V_idx(Nx+j-1,n+1)]=eps # F_V - V   ## Ok
                row[cmpt]=FV_idx(Nx+j,n); col[cmpt]=V_idx(Nx+j-1,n+1); data[cmpt]=eps
                cmpt +=1
            if j==1:
                # 1 J[Fr_idx(j,n),r_idx(Nx,n)]=(0.5*dt/dx)*w[u_idx(Nx,n)]-0.5 # F_rho - rho
                row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(Nx,n); data[cmpt]=(0.5*dt/dx)*w[u_idx(Nx,n)]-0.5
                cmpt +=1
                # 2 J[Fr_idx(Nx+j,n),r_idx(Nx+Nx,n)]=(0.5*dt/dx)*w[u_idx(Nx+Nx,n)]-0.5 # F_rho - rho
                row[cmpt]=Fr_idx(Nx+j,n); col[cmpt]=r_idx(Nx+Nx,n); data[cmpt]=(0.5*dt/dx)*w[u_idx(Nx+Nx,n)]-0.5
                cmpt +=1
                # 1 J[Fr_idx(j,n),u_idx(Nx,n)]=-(0.5*dt/dx)*w[r_idx(Nx,n)] # F_rho - u
                row[cmpt]=Fr_idx(j,n); col[cmpt]=u_idx(Nx,n); data[cmpt]=-(0.5*dt/dx)*w[r_idx(Nx,n)]
                cmpt +=1
                # 2 J[Fr_idx(Nx+j,n),u_idx(Nx+Nx,n)]=-(0.5*dt/dx)*w[r_idx(Nx+Nx,n)] # F_rho - u
                row[cmpt]=Fr_idx(Nx+j,n); col[cmpt]=u_idx(Nx+Nx,n); data[cmpt]=-(0.5*dt/dx)*w[r_idx(Nx+Nx,n)]
                cmpt +=1
                # 1 J[FV_idx(j,n),V_idx(Nx,n+1)]=eps # F_V - V
                row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(Nx,n+1); data[cmpt]=eps
                cmpt +=1
                # 2 J[FV_idx(Nx+j,n),V_idx(Nx+Nx,n+1)]=eps # F_V - V
                row[cmpt]=FV_idx(Nx+j,n); col[cmpt]=V_idx(Nx+Nx,n+1); data[cmpt]=eps
                cmpt +=1
            if j!=Nx:
                # 1 J[Fr_idx(j,n),r_idx(j+1,n)]=(0.5*dt/dx)*w[u_idx(j+1,n)]-0.5 # F_rho -rho  ## Ok
                row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(j+1,n); data[cmpt]=(0.5*dt/dx)*w[u_idx(j+1,n)]-0.5
                cmpt +=1
                # 2 J[Fr_idx(Nx+j,n),r_idx(Nx+j+1,n)]=(0.5*dt/dx)*w[u_idx(Nx+j+1,n)]-0.5 # F_rho -rho  ## Ok
                row[cmpt]=Fr_idx(Nx+j,n); col[cmpt]=r_idx(Nx+j+1,n); data[cmpt]=(0.5*dt/dx)*w[u_idx(Nx+j+1,n)]-0.5
                cmpt +=1
                # 1 J[Fr_idx(j,n),u_idx(j+1,n)]=(0.5*dt/dx)*w[r_idx(j+1,n)] # F_rho - u ## Ok
                row[cmpt]=Fr_idx(j,n); col[cmpt]=u_idx(j+1,n); data[cmpt]=(0.5*dt/dx)*w[r_idx(j+1,n)]
                cmpt +=1
                # 2 J[Fr_idx(Nx+j,n),u_idx(Nx+j+1,n)]=(0.5*dt/dx)*w[r_idx(Nx+j+1,n)] # F_rho - u ## Ok
                row[cmpt]=Fr_idx(Nx+j,n); col[cmpt]=u_idx(Nx+j+1,n); data[cmpt]=(0.5*dt/dx)*w[r_idx(Nx+j+1,n)]
                cmpt +=1
                # 1 J[FV_idx(j,n),V_idx(j+1,n+1)]=eps # F_V - V  ## Ok
                row[cmpt]=FV_idx(j,n); col[cmpt]=V_idx(j+1,n+1); data[cmpt]=eps
                cmpt +=1
                # 2 J[FV_idx(Nx+j,n),V_idx(Nx+j+1,n+1)]=eps # F_V - V  ## Ok
                row[cmpt]=FV_idx(Nx+j,n); col[cmpt]=V_idx(Nx+j+1,n+1); data[cmpt]=eps
                cmpt +=1
            if j==Nx:
                # 1 J[Fr_idx(j,n),r_idx(1,n)]=(0.5*dt/dx)*w[u_idx(1,n)]-0.5 # F_rho - rho
                row[cmpt]=Fr_idx(j,n); col[cmpt]=r_idx(1,n); data[cmpt]=(0.5*dt/dx)*w[u_idx(1,n)]-0.5
                cmpt +=1
                # 2 J[Fr_idx(Nx+j,n),r_idx(Nx+1,n)]=(0.5*dt/dx)*w[u_idx(Nx+1,n)]-0.5 # F_rho - rho
                row[cmpt]=Fr_idx(Nx+j,n); col[cmpt]=r_idx(Nx+1,n); data[cmpt]=(0.5*dt/dx)*w[u_idx(Nx+1,n)]-0.5
                cmpt +=1
                # 1 J[Fr_idx(j,n),u_idx(1,n)]=(0.5*dt/dx)*w[r_idx(1,n)] # F_rho - u
                row[cmpt]=Fr_idx(j,n); col[cmpt]=u_idx(1,n); data[cmpt]=(0.5*dt/dx)*w[r_idx(1,n)]
                cmpt +=1
                # 2 J[Fr_idx(Nx+j,n),u_idx(Nx+1,n)]=(0.5*dt/dx)*w[r_idx(Nx+1,n)] # F_rho - u
                row[cmpt]=Fr_idx(Nx+j,n); col[cmpt]=u_idx(Nx+1,n); data[cmpt]=(0.5*dt/dx)*w[r_idx(Nx+1,n)]
                cmpt +=1
                # 1 J[FV_idx(j,n),V_idx(1,n+1)]=eps # F_V - V
                row[cmpt]=FV_idx(Nx+j,n); col[cmpt]=V_idx(Nx+1,n+1); data[cmpt]=eps
                cmpt +=1
                # 2 J[FV_idx(j,n),V_idx(1,n+1)]=eps # F_V - V
                row[cmpt]=FV_idx(Nx+j,n); col[cmpt]=V_idx(Nx+1,n+1); data[cmpt]=eps
                cmpt +=1
  
    for j in range(1,Nx+1):
        # 1 J[Frint_idx(j),r_idx(j,0)]=1 # F_rho_int - rho  ## Ok
        row[cmpt]=Frint_idx(j); col[cmpt]=r_idx(j,0); data[cmpt]=1
        cmpt +=1
        # 2 J[Frint_idx(Nx+j),r_idx(Nx+j,0)]=1 # F_rho_int - rho  ## Ok
        row[cmpt]=Frint_idx(Nx+j); col[cmpt]=r_idx(Nx+j,0); data[cmpt]=1
        cmpt +=1
        # 1 J[FVter_idx(j),V_idx(j,Nt)]=1 # F_V_ter - V ## Ok
        row[cmpt]=FVter_idx(j); col[cmpt]=V_idx(j,Nt); data[cmpt]=1
        cmpt +=1
        # 2 J[FVter_idx(Nx+j),V_idx(Nx+j,Nt)]=1 # F_V_ter - V ## Ok
        row[cmpt]=FVter_idx(Nx+j); col[cmpt]=V_idx(Nx+j,Nt); data[cmpt]=1
        cmpt +=1
    
    # return J
    return 0


xglo = None

def get_preconditioner(a):
    # Jac=jacobian(a)
    # Jac1 = csc_matrix(Jac)
    row=np.zeros(10*Nt*(2*Nx)+2*(2*Nx))
    col=np.zeros(10*Nt*(2*Nx)+2*(2*Nx))
    data=np.zeros(10*Nt*(2*Nx)+2*(2*Nx))
    jacobian(a,row, col, data)
    shap=(3*Nt*(2*Nx)+2*(2*Nx),3*Nt*(2*Nx)+2*(2*Nx))
    Jac1 = csc_matrix((data, (row, col)),shape = shap)
    # the *incomplete LU* decomposition
    J_ilu = spla.spilu(Jac1)
    M_x = lambda r: J_ilu.solve(r)
    M = spla.LinearOperator(shap, M_x)
    # M=np.linalg.inv(Jac1)
    
    #.......................................................................For updating
    def xglobal( x, F ):
        """ newton_krylov calls this at each iteration: xglo = x """
        global xglo
        xglo = x.copy()
        # print( "update: x %s  F %s " % (nu.asum(x), nu.asum(F)) )  # test
            # nu.asum: array summary, size min av max

    M.update = xglobal

    return M

    
def sol_to(old_Nt,old_Nx,sol,rho1,u1,V1,rho2,u2,V2):
    def r_id(j,n):
        return (j-1)*(old_Nt+1)+n
    def u_id(j,n):
        return (old_Nt+1)*(2*old_Nx)+(j-1)*old_Nt+n
    def V_id(j,n):
        return (2*old_Nt+1)*(2*old_Nx)+(j-1)*(old_Nt+1)+n
    for j in range(1,old_Nx+1):
        for n in range(0,old_Nt):
            rho1[j,n]=sol[r_id(j,n)]
            u1[j,n]=sol[u_id(j,n)]
            V1[j,n]=sol[V_id(j,n)]
        rho1[j,old_Nt]=sol[r_id(j,old_Nt)]
        V1[j,old_Nt]=sol[V_id(j,old_Nt)]
    for j in range(old_Nx+1,2*old_Nx+1):
        for n in range(0,old_Nt):
            rho2[j-old_Nx,n]=sol[r_id(j,n)]
            u2[j-old_Nx,n]=sol[u_id(j,n)]
            V2[j-old_Nx,n]=sol[V_id(j,n)]
        rho2[j-old_Nx,old_Nt]=sol[r_id(j,old_Nt)]
        V2[j-old_Nx,old_Nt]=sol[V_id(j,old_Nt)]
    for n in range(0,old_Nt):
        rho1[0,n]=rho1[old_Nx,n]
        V1[0,n]=V1[old_Nx,n]
        u1[0,n]=u1[old_Nx,n]
        rho2[0,n]=rho2[old_Nx,n]
        V2[0,n]=V2[old_Nx,n]
        u2[0,n]=u2[old_Nx,n]
    rho1[0,old_Nt]=rho1[old_Nx,old_Nt]
    V1[0,old_Nt]=V1[old_Nx,old_Nt]
    rho2[0,old_Nt]=rho2[old_Nx,old_Nt]
    V2[0,old_Nt]=V2[old_Nx,old_Nt]
    return 0

    
def to_sol(new_Nt,old_Nx,sol,rho1,u1,V1,rho2,u2,V2): 
    def r_id(j,n):
        return (j-1)*(2*new_Nt+1)+n
    def u_id(j,n):
        return (2*new_Nt+1)*2*(2*old_Nx)+(j-1)*2*new_Nt+n
    def V_id(j,n):
        return (2*2*new_Nt+1)*2*(2*old_Nx)+(j-1)*(2*new_Nt+1)+n
    
    for j in range(1,2*old_Nx+1):
        for n in range(0,2*new_Nt):
            sol[r_id(j,n)]=rho1[j,n]
            sol[u_id(j,n)]=u1[j,n]
            sol[V_id(j,n)]=V1[j,n]
        sol[r_id(j,2*new_Nt)]=rho1[j,2*new_Nt]
        sol[V_id(j,2*new_Nt)]=V1[j,2*new_Nt]
    for j in range(2*old_Nx+1,2*(2*old_Nx)+1):
        for n in range(0,2*new_Nt):
            sol[r_id(j,n)]=rho2[j-2*old_Nx,n]
            sol[u_id(j,n)]=u2[j-2*old_Nx,n]
            sol[V_id(j,n)]=V2[j-2*old_Nx,n]
        sol[r_id(j,2*new_Nt)]=rho2[j-2*old_Nx,2*new_Nt]
        sol[V_id(j,2*new_Nt)]=V2[j-2*old_Nx,2*new_Nt]
    return 0

    


import scipy.interpolate as interpolate

def interpol(n,new_n,data): # 1D interpolation
    
    """" Go from a coarse grid Nt*Nx to a finer grid spacing (2*Nt)*(2*Nx) """""
    i = np.indices(data.shape)[0]/(n-1)  # [0, ..., 1]
    new_i = np.linspace(0, 1, new_n)
    # Create a linear interpolation function based on the original data
    linear_interpolation_func = interpolate.interp1d(i, data, kind='cubic') 
    # ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
    new_data = linear_interpolation_func(new_i)

    return new_data


def multigrid(new_Nt,old_Nt,old_Nx,w):
    rho1=np.zeros((old_Nx+1,old_Nt+1))
    u1=np.zeros((old_Nx+1,old_Nt))
    V1=np.zeros((old_Nx+1,old_Nt+1))
    rho2=np.zeros((old_Nx+1,old_Nt+1))
    u2=np.zeros((old_Nx+1,old_Nt))
    V2=np.zeros((old_Nx+1,old_Nt+1))
    sol_to(old_Nt,old_Nx,w,rho1,u1,V1,rho2,u2,V2)
    new1_rho1=np.zeros((2*old_Nx+1,old_Nt+1))
    new1_u1=np.zeros((2*old_Nx+1,old_Nt))
    new1_V1=np.zeros((2*old_Nx+1,old_Nt+1))
    new1_rho2=np.zeros((2*old_Nx+1,old_Nt+1))
    new1_u2=np.zeros((2*old_Nx+1,old_Nt))
    new1_V2=np.zeros((2*old_Nx+1,old_Nt+1))
    for n in range(old_Nt):
        new1_rho1[1:,n]=interpol(old_Nx,2*old_Nx,rho1[1:,n])
        new1_u1[1:,n]=interpol(old_Nx,2*old_Nx,u1[1:,n])
        new1_V1[1:,n]=interpol(old_Nx,2*old_Nx,V1[1:,n])
        new1_rho2[1:,n]=interpol(old_Nx,2*old_Nx,rho2[1:,n])
        new1_u2[1:,n]=interpol(old_Nx,2*old_Nx,u2[1:,n])
        new1_V2[1:,n]=interpol(old_Nx,2*old_Nx,V2[1:,n])
    new1_rho1[1:,old_Nt]=interpol(old_Nx,2*old_Nx,rho1[1:,old_Nt])
    new1_V1[1:,old_Nt]=interpol(old_Nx,2*old_Nx,V1[1:,old_Nt])
    new1_rho2[1:,old_Nt]=interpol(old_Nx,2*old_Nx,rho2[1:,old_Nt])
    new1_V2[1:,old_Nt]=interpol(old_Nx,2*old_Nx,V2[1:,old_Nt])
    new_rho1=np.zeros((2*old_Nx+1,2*new_Nt+1))
    new_u1=np.zeros((2*old_Nx+1,2*new_Nt))
    new_V1=np.zeros((2*old_Nx+1,2*new_Nt+1))
    new_rho2=np.zeros((2*old_Nx+1,2*new_Nt+1))
    new_u2=np.zeros((2*old_Nx+1,2*new_Nt))
    new_V2=np.zeros((2*old_Nx+1,2*new_Nt+1))
    for j in range(0,2*old_Nx+1):
        new_rho1[j,:]=interpol(old_Nt+1,2*new_Nt+1,new1_rho1[j,:])
        new_u1[j,:]=interpol(old_Nt,2*new_Nt,new1_u1[j,:])
        new_V1[j,:]=interpol(old_Nt+1,2*new_Nt+1,new1_V1[j,:]) 
        new_rho2[j,:]=interpol(old_Nt+1,2*new_Nt+1,new1_rho2[j,:])
        new_u2[j,:]=interpol(old_Nt,2*new_Nt,new1_u2[j,:])
        new_V2[j,:]=interpol(old_Nt+1,2*new_Nt+1,new1_V2[j,:]) 
    new_w = np.zeros(3*(2*new_Nt)*(2*(2*old_Nx))+2*(2*(2*old_Nx)))
    to_sol(new_Nt,old_Nx,new_w,new_rho1,new_u1,new_V1,new_rho2,new_u2,new_V2)
    
    return new_w

#---------Functions     
F  = lambda x : Fct(x,f_star_p_2NonSep,f_star_2NonSep)
# F  = lambda x : Fct(x,f_star_p_2Sep,f_star_2Sep)

#---------viscosity coefficient
mu=[0.05, 0.04, 0.03, 0.02, 0.01] # viscosity coefficient

#---------For save_text
guess_text='guess.dat'
sol_text='sol.dat'

#---------Coarse grid
Nx=30; Nt=120 # spatial-temporal grid sizes 
multip=2

use_precond=1 
use_multigrid=1 

import os
import sys 
stdoutOrigin=sys.stdout 
sys.stdout = open("outputs.dat", "w")

for i in range(0,5):
    if i == 0:
        use_interp = 0
    else:
        use_interp = 1
        sol=np.loadtxt(sol_text)
        old_Nx=int(sol[0]); old_Nt=int(sol[1])
        sol=sol[2:]
        Nx=old_Nx*multip; Nt=old_Nt*multip

    dx=L/Nx # spatial step size
    if mu[i]==0.0:
        dt=min(T/Nt,(CFL*dx)/max(u1_max,u2_max)) # temporal step size
        eps=0.0
    else:
        dt=min(T/Nt,CFL*dx/max(u1_max,u2_max),EPS*(dx**2)/mu[i]) # temporal step size
        eps=mu[i]*dt/(dx**2) # V
    x=np.linspace(0,L,Nx+1)
    t=np.arange(0,T+dt,dt)
    Nt=multip*int((len(t)-1)/multip)
    print('Nx={Nx}, Nt={Nt}, nu={nu}'.format(Nx=Nx,Nt=Nt,nu=mu[i]))
    print('dx={dx}, dt={dt}'.format(dx=round(dx,4),dt=round(dt,4)))
    
    #---------------------MultiGrid
    if use_interp==1 and use_multigrid==1:
            t0 = time.process_time()   ###
            guess=multigrid(int(Nt/multip),old_Nt,old_Nx,sol)
            t1 = time.process_time()   ###
            print("Time spent (multigrid) :",t1-t0)
            if os.path.exists(guess_text):
                os.remove(guess_text)
            np.savetxt(guess_text, guess)
    elif use_interp==0: guess = np.zeros(3*Nt*(2*Nx)+2*(2*Nx))
    #---------------------preconditionning
    if use_precond==1:
        t0 = time.process_time()   ###
        prec=get_preconditioner(guess)
        t1 = time.process_time()   ###
        time1=t1-t0
        print("Time spent (anal_precond) :",time1)
    elif use_precond==0: prec=None
    #---------------------Newton-GMRES
    t0 = time.process_time()   ###
    sol = newton_krylov(F, guess, method='lgmres', verbose=1, inner_M=prec) # f_tol=2e-08 (default 6e-06), maxiter=500 (default 50000)
    t1 = time.process_time()   ###
    time2=t1-t0
    print("Time spent (gmres) :",time2)
    cpu_time=time1+time2
    print("CPU time :",cpu_time)
    #------Save : Nx,Nt,sol
    # import os
    if os.path.exists(sol_text):
        os.remove(sol_text)
    with open(sol_text, "a") as text_file:
            text_file.write(str(Nx))
            text_file.write("\n")
            text_file.write(str(Nt))
            text_file.write("\n")
            np.savetxt(text_file, sol)
    
sys.stdout.close()
sys.stdout=stdoutOrigin
print('End')


        


