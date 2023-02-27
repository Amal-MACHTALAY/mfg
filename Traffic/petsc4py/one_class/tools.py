#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 22:36:29 2022
@author: kissami
"""
from modules import from_1d_to_2d, from_2d_to_1d
import numpy as np

'''************************ functions **********************************'''
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

# def initialguess(new_Nt,old_Nt,old_Nx,w,multip):
def initialguess(Nt, Nx, multip, text):
    
    new_Nt = int(Nt/multip)

    npzfile = np.load(text)
    w = npzfile['solution']
    old_Nx = int(npzfile['Nx'])
    old_Nt = int(npzfile['Nt'])
    
    # old_Nx = int(w[0])
    # old_Nt = int(w[1])
    # w = w[2:]

    rho=np.zeros((old_Nx,old_Nt+1))
    u=np.zeros((old_Nx,old_Nt))
    V=np.zeros((old_Nx,old_Nt+1))
    from_1d_to_2d(old_Nt,old_Nx, w,rho,u,V)
    
    new1_rho=np.zeros((multip*old_Nx,old_Nt+1))
    new1_u=np.zeros((multip*old_Nx,old_Nt))
    new1_V=np.zeros((multip*old_Nx,old_Nt+1))
    for n in range(old_Nt+1):
        new1_rho[:,n]=interpol(old_Nx,multip*old_Nx,rho[:,n])
        if n<old_Nt:
            new1_u[:,n]=interpol(old_Nx,multip*old_Nx,u[:,n])
        new1_V[:,n]=interpol(old_Nx,multip*old_Nx,V[:,n])
    new_rho=np.zeros((multip*old_Nx,multip*new_Nt+1))
    new_u=np.zeros((multip*old_Nx,multip*new_Nt))
    new_V=np.zeros((multip*old_Nx,multip*new_Nt+1))
    for j in range(multip*old_Nx):
        new_rho[j,:]=interpol(old_Nt+1,multip*new_Nt+1,new1_rho[j,:])
        new_u[j,:]=interpol(old_Nt,multip*new_Nt,new1_u[j,:])
        new_V[j,:]=interpol(old_Nt+1,multip*new_Nt+1,new1_V[j,:])
        
    new_w = np.zeros(3*(multip*new_Nt)*(multip*old_Nx)+2*(multip*old_Nx))
    from_2d_to_1d(new_Nt, old_Nx, new_w,new_rho,new_u,new_V, multip)
  
    X = new_w
    
    return X

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