#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:54:05 2022

@author: amal
"""

import numpy as np

def from_1d_to_2d(old_Nt,old_Nx,sol,rho,u,V): # solution 1D to 2D
    for j in range(0,old_Nx):
        for n in range(0,old_Nt):
            rho[j,n]=sol[j*(old_Nt+1)+n]
            u[j,n]=sol[(old_Nt+1)*old_Nx+j*old_Nt+n]
            V[j,n]=sol[(2*old_Nt+1)*old_Nx+j*(old_Nt+1)+n]
        rho[j,old_Nt]=sol[j*(old_Nt+1)+old_Nt]
        V[j,old_Nt]=sol[(2*old_Nt+1)*old_Nx+j*(old_Nt+1)+old_Nt]


def from_2d_to_1d(new_Nt,old_Nx,sol,rho,u,V,multip):# solution 2D to 1D
    for j in range(0,multip*old_Nx):
        for n in range(0,multip*new_Nt):
            sol[j*(multip*new_Nt+1)+n]=rho[j,n]
            sol[(multip*new_Nt+1)*multip*old_Nx+j*multip*new_Nt+n]=u[j,n]
            sol[(2*multip*new_Nt+1)*multip*old_Nx+j*(multip*new_Nt+1)+n]=V[j,n]
        sol[j*(multip*new_Nt+1)+multip*new_Nt]=rho[j,multip*new_Nt]
        sol[(2*multip*new_Nt+1)*multip*old_Nx+j*(multip*new_Nt+1)+multip*new_Nt]=V[j,multip*new_Nt]


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
    
    rho=np.zeros((old_Nx,old_Nt+1))
    u=np.zeros((old_Nx,old_Nt))
    V=np.zeros((old_Nx,old_Nt+1))
    from_1d_to_2d(old_Nt,old_Nx, w,rho,u,V)
    
    new1_rho=np.zeros((multip*old_Nx,old_Nt+1))
    new1_u=np.zeros((multip*old_Nx,old_Nt))
    new1_V=np.zeros((multip*old_Nx,old_Nt+1))
    for n in range(old_Nt):
        new1_rho[:,n]=interpol(old_Nx,multip*old_Nx,rho[:,n])
        new1_u[:,n]=interpol(old_Nx,multip*old_Nx,u[:,n])
        new1_V[:,n]=interpol(old_Nx,multip*old_Nx,V[:,n])
    new1_rho[:,old_Nt]=interpol(old_Nx,multip*old_Nx,rho[:,old_Nt])
    new1_V[:,old_Nt]=interpol(old_Nx,multip*old_Nx,V[:,old_Nt])
    new_rho=np.zeros((multip*old_Nx,multip*new_Nt+1))
    new_u=np.zeros((multip*old_Nx,multip*new_Nt))
    new_V=np.zeros((multip*old_Nx,multip*new_Nt+1))
    for j in range(multip*old_Nx):
        new_rho[j,:]=interpol(old_Nt+1,multip*new_Nt+1,new1_rho[j,:])
        new_u[j,:]=interpol(old_Nt,multip*new_Nt,new1_u[j,:])
        new_V[j,:]=interpol(old_Nt+1,multip*new_Nt+1,new1_V[j,:])
        
    new_w = np.zeros(3*(multip*new_Nt)*(multip*old_Nx)+2*(multip*old_Nx))
    from_2d_to_1d(new_Nt, old_Nx, new_w,new_rho,new_u,new_V, multip)
    
    return new_w