#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:51:23 2022

@author: amal
"""

# j = 1 : Nx ; n = 0 : Nt

def r_idx(j,n,Nt): 
    return (j-1)*(Nt+1)+n

def u_idx(j,n,Nt,Nx):
    return (Nt+1)*Nx+(j-1)*Nt+n

def V_idx(j,n,Nt,Nx):
    return (2*Nt+1)*Nx+(j-1)*(Nt+1)+n

def Fr_idx(j,n,Nt):
    return (j-1)*Nt+n

def Fu_idx(j,n,Nt,Nx):
    return Nt*Nx+(j-1)*Nt+n

def FV_idx(j,n,Nt,Nx):
    return 2*Nt*Nx+(j-1)*Nt+n

def Frint_idx(j,Nt,Nx):
    return 3*Nt*Nx+(j-1)

def FVter_idx(j,Nt,Nx):
    return 3*Nt*Nx+Nx+(j-1)

import numpy as np
def solutions(sol,Nt,Nx): # solution 1D to 2D
    rho=np.zeros((Nx,Nt+1))
    u=np.zeros((Nx,Nt))
    V=np.zeros((Nx,Nt+1))
    Q=np.zeros((Nx,Nt))
    for j in range(1,Nx+1):
        for n in range(0,Nt+1):
            rho[j-1,n]=sol[(j-1)*(Nt+1)+n]
            if n<Nt: 
                u[j-1,n]=sol[(Nt+1)*Nx+(j-1)*Nt+n]
                Q[j-1,n]=rho[j-1,n]*u[j-1,n]
            V[j-1,n]=sol[(2*Nt+1)*Nx+(j-1)*(Nt+1)+n]
    
    return rho,u,V,Q

