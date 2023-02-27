#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:39:31 2022

@author: amal
"""

import numpy as np

''' inputs '''
def inputs():
    return T,L,u_max,rho_jam, rho_a, rho_b, gama, CFL, EPS, nb_grid, mu, Nx0, Nt0, multip, use_precond, use_multigrid

T=1.0 # horizon length 
N=61 # number of cars 
u_max=N # free flow speed
rho_jam=1.0 # jam density
L=N # road length
CFL=0.75    # CFL<1
# rho_a=0.05; rho_b=0.95; gama=0.1
rho_a=0.2; rho_b=0.8; gama=0.15*L
EPS=0.45 # viscosity coeff
nb_grid = 5 # number of grids
#---------viscosity coefficient
mu=[0.0]*(nb_grid) # LWR  
# mu=[0.05, 0.045, 0.04, 0.035, 0.03, 0.02]  # Sep  
# mu=[0.04, 0.03, 0.02, 0.01, 0.0] # NonSep 
#---------Coarse grid
Nx0=15; Nt0=60 # spatial-temporal grid sizes 
multip=2

use_precond= True 
use_multigrid= True 


''' LWR functions '''
# def U(rho): # Greenshields desired speed
#     return u_max*(1-rho/rho_jam)

# def f_mfg(u,r):
#     return 0.5*((U(r)-u)**2) # MFG-LWR

# def f_starp(p,r): # 0<=u<=u_max
#     return U(r)-p # MFG-LWR
    
# def f_star(p,r,u): # p=Vx
#     return -0.5*(p**2)+U(r)*p # MFG-LWR  

# ############# For exact jacobian
# def f_starp_p(p,r):
#     return -1.0
    
# def f_starp_r(p,r):   
#     return -u_max/rho_jam

# def f_star_p(p,r,u):
#     return -p+U(r)

# def f_star_r(p,r,u):
#     return -(u_max/rho_jam)*p

# def f_star_u(p,r,u):
#     return 0.0
##############

''' Separable functions '''
def f_mfg(u,r):
    return 0.5*((u/u_max)**2)-(u/u_max)+(r/rho_jam) # MFG-Separable

def f_starp(p,r): # 0<=u<=u_max
    return max(min(u_max*(1-p*u_max),u_max),0.0) # MFG-Separable
    
def f_star(p,r,u): # p=Vx
    return u*p+f_mfg(u,r) # MFG-Separable   

############ For exact jacobian
def f_starp_p(p,r):
    return -u_max**2
    
def f_starp_r(p,r):   
    return 0.0

def f_star_p(p,r,u):
    return u

def f_star_r(p,r,u): # ok
    return 1/rho_jam

def f_star_u(p,r,u): # ok
    return p+u/(u_max**2)-1/u_max
##############

''' Non-separable functions '''
# def f_mfg(u,r):
#     return 0.5*((u/u_max)**2)-(u/u_max)+((u*r)/(u_max*rho_jam)) # MFG-NonSeparable

# def f_starp(p,r): # 0<=u<=u_max
#     return max(min(u_max*(1-r/rho_jam-u_max*p),u_max),0.0) # MFG-NonSeparable

# def f_star(p,r,u): # p=Vx
#     return u*p+f_mfg(u,r) # MFG-NonSeparable  

# ############ For exact jacobian
# def f_starp_p(p,r):
#     return -u_max**2
    
# def f_starp_r(p,r):   
#     return -u_max/rho_jam

# def f_star_p(p,r,u):
#     return u

# def f_star_r(p,r,u): # ok
#     return u/(u_max*rho_jam)

# def f_star_u(p,r,u): # ok
#     return p+u/(u_max**2)-1/u_max+r/(u_max*rho_jam)
# ##############


def rho_int(s): # initial density
    return rho_a+(rho_b-rho_a)*np.exp(-0.5*((s-0.5*L)/gama)**2) # 0<=rho<=rho_jam

def VT(a): # Terminal cost
    return 0.0



