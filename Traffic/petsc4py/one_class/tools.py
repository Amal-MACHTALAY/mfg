#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 22:36:29 2022

@author: kissami
"""
from modules import sol_to, to_sol
import numpy as np

'''************************ functions **********************************'''
import scipy.interpolate as interpolate
def interpol(n, new_n, data): # 1D interpolation
    
    """" Go from a coarse grid Nt*Nx to a finer grid spacing (2*Nt)*(2*Nx) """""
    i = np.indices(data.shape)[0]/(n-1)  # [0, ..., 1]
    new_i = np.linspace(0, 1, new_n)
    linear_interpolation_func = interpolate.interp1d(i, data, kind='cubic') 
    # ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
    new_data = linear_interpolation_func(new_i)
    return new_data


def initialguess(Nt, Nx, multip, filename):
    
    new_Nt = int(Nt/multip)
    
    npzfile=np.load(filename)
    w=npzfile['sol']
    old_Nt=int(npzfile['Nt'])
    old_Nx=int(npzfile['Nx'])
    # w = np.loadtxt(filename)
    
    # old_Nx = int(w[0])
    # old_Nt = int(w[1])
    # w = w[2:]

    rho=np.zeros((old_Nx,old_Nt+1))
    u=np.zeros((old_Nx,old_Nt))
    V=np.zeros((old_Nx,old_Nt+1))
    sol_to(old_Nt,old_Nx, w,rho,u,V)
    
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
    to_sol(new_Nt, old_Nx, new_w,new_rho,new_u,new_V, multip)
  
    X = new_w
    
    return X
