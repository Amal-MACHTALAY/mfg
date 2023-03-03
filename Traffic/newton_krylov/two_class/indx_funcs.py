#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:51:23 2022

@author: amal
"""

# j = 1 : 2*Nx ; n = 0 : Nt

def r_idx(j,n,Nt): 
    return (j-1)*(Nt+1)+n

def u_idx(j,n,Nt,Nx):
    return (Nt+1)*(2*Nx)+(j-1)*Nt+n

def V_idx(j,n,Nt,Nx):
    return (2*Nt+1)*(2*Nx)+(j-1)*(Nt+1)+n

def Fr_idx(j,n,Nt):
    return (j-1)*Nt+n

def Fu_idx(j,n,Nt,Nx):
    return Nt*(2*Nx)+(j-1)*Nt+n

def FV_idx(j,n,Nt,Nx):
    return 2*Nt*(2*Nx)+(j-1)*Nt+n

def Frint_idx(j,Nt,Nx):
    return 3*Nt*(2*Nx)+(j-1)

def FVter_idx(j,Nt,Nx):
    return (3*Nt+1)*(2*Nx)+(j-1)

import numpy as np
def solutions(sol,Nt,Nx): # solution 1D to 2D
    rho1=np.zeros((Nx,Nt+1))
    u1=np.zeros((Nx,Nt))
    V1=np.zeros((Nx,Nt+1))
    Q1=np.zeros((Nx,Nt))
    rho2=np.zeros((Nx,Nt+1))
    u2=np.zeros((Nx,Nt))
    V2=np.zeros((Nx,Nt+1))
    Q2=np.zeros((Nx,Nt))
    for j in range(1,Nx+1):
        i = j+Nx
        for n in range(0,Nt+1):
            rho1[j-1,n]=sol[r_idx(j,n,Nt)]
            rho2[j-1,n]=sol[r_idx(i,n,Nt)]
            if n<Nt: 
                u1[j-1,n]=sol[u_idx(j,n,Nt,Nx)]
                u2[j-1,n]=sol[u_idx(i,n,Nt,Nx)]
                Q1[j-1,n]=rho1[j-1,n]*u1[j-1,n]
                Q2[j-1,n]=rho2[j-1,n]*u2[j-1,n]
            V1[j-1,n]=sol[V_idx(j,n,Nt,Nx)]
            V2[j-1,n]=sol[V_idx(i,n,Nt,Nx)]
    
    return rho1,u1,V1,Q1,rho2,u2,V2,Q2

