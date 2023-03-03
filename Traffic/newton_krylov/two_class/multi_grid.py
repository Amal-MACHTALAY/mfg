#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:54:05 2022

@author: amal
"""

import numpy as np
from indx_funcs import r_idx, u_idx, V_idx
        
def from_1d_to_2d(old_Nt,old_Nx,sol,rho1,u1,V1,rho2,u2,V2): # solution 1D to 2D
    for j in range(1,old_Nx+1):
        i = j+old_Nx
        for n in range(0,old_Nt+1):
            rho1[j-1,n]=sol[r_idx(j,n,old_Nt)]
            rho2[j-1,n]=sol[r_idx(i,n,old_Nt)]
            if n<old_Nt:
                u1[j-1,n]=sol[u_idx(j,n,old_Nt,old_Nx)]
                u2[j-1,n]=sol[u_idx(i,n,old_Nt,old_Nx)]
            V1[j-1,n]=sol[V_idx(j,n,old_Nt,old_Nx)]
            V2[j-1,n]=sol[V_idx(i,n,old_Nt,old_Nx)]

        
def from_2d_to_1d(new_Nt,old_Nx,sol,rho1,u1,V1,rho2,u2,V2,multip):# solution 2D to 1D
    mNt = multip*new_Nt
    mNx = multip*old_Nx
    for j in range(1,mNx+1):
        i = j+mNx
        for n in range(0,mNt+1):
            sol[r_idx(j,n,mNt)]=rho1[j-1,n]
            sol[r_idx(i,n,mNt)]=rho2[j-1,n]
            if n<multip*new_Nt:
                sol[u_idx(j,n,mNt,mNx)]=u1[j-1,n]
                sol[u_idx(i,n,mNt,mNx)]=u2[j-1,n]
            sol[V_idx(j,n,mNt,mNx)]=V1[j-1,n]
            sol[V_idx(i,n,mNt,mNx)]=V2[j-1,n]


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


def multigrid(new_Nt,old_Nt,old_Nx,w,multip):
    
    rho1=np.zeros((old_Nx,old_Nt+1))
    u1=np.zeros((old_Nx,old_Nt))
    V1=np.zeros((old_Nx,old_Nt+1))
    rho2=np.zeros((old_Nx,old_Nt+1))
    u2=np.zeros((old_Nx,old_Nt))
    V2=np.zeros((old_Nx,old_Nt+1))
    from_1d_to_2d(old_Nt,old_Nx,w,rho1,u1,V1,rho2,u2,V2)
    
    new1_rho1=np.zeros((multip*old_Nx,old_Nt+1))
    new1_u1=np.zeros((multip*old_Nx,old_Nt))
    new1_V1=np.zeros((multip*old_Nx,old_Nt+1))
    new1_rho2=np.zeros((multip*old_Nx,old_Nt+1))
    new1_u2=np.zeros((multip*old_Nx,old_Nt))
    new1_V2=np.zeros((multip*old_Nx,old_Nt+1))
    for n in range(old_Nt+1):
        new1_rho1[:,n]=interpol(old_Nx,multip*old_Nx,rho1[:,n])
        new1_rho2[:,n]=interpol(old_Nx,multip*old_Nx,rho2[:,n])
        if n<old_Nt:
            new1_u1[:,n]=interpol(old_Nx,multip*old_Nx,u1[:,n])
            new1_u2[:,n]=interpol(old_Nx,multip*old_Nx,u2[:,n])
        new1_V1[:,n]=interpol(old_Nx,multip*old_Nx,V1[:,n])
        new1_V2[:,n]=interpol(old_Nx,multip*old_Nx,V2[:,n])
        
    new_rho1=np.zeros((multip*old_Nx,multip*new_Nt+1))
    new_u1=np.zeros((multip*old_Nx,multip*new_Nt))
    new_V1=np.zeros((multip*old_Nx,multip*new_Nt+1))
    new_rho2=np.zeros((multip*old_Nx,multip*new_Nt+1))
    new_u2=np.zeros((multip*old_Nx,multip*new_Nt))
    new_V2=np.zeros((multip*old_Nx,multip*new_Nt+1))
    for j in range(multip*old_Nx):
        new_rho1[j,:]=interpol(old_Nt+1,multip*new_Nt+1,new1_rho1[j,:])
        new_rho2[j,:]=interpol(old_Nt+1,multip*new_Nt+1,new1_rho2[j,:])
        new_u1[j,:]=interpol(old_Nt,multip*new_Nt,new1_u1[j,:])
        new_u2[j,:]=interpol(old_Nt,multip*new_Nt,new1_u2[j,:])
        new_V1[j,:]=interpol(old_Nt+1,multip*new_Nt+1,new1_V1[j,:])
        new_V2[j,:]=interpol(old_Nt+1,multip*new_Nt+1,new1_V2[j,:])
        
    new_w = np.zeros(3*(multip*new_Nt)*(multip*(2*old_Nx))+2*(multip*(2*old_Nx)))
    from_2d_to_1d(new_Nt,old_Nx,new_w,new_rho1,new_u1,new_V1,new_rho2,new_u2,new_V2,multip)
    
    return new_w