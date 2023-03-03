#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:39:31 2022

@author: amal
"""

import numpy as np

''' inputs '''
def inputs():
    return T,L,l1,l2,u1_max,u2_max,rho1_jam,rho2_jam, CFL, EPS, nb_grid, mu, Nx0, Nt0, multip, use_precond, use_multigrid

T=3.0 # horizon length 
''' population 1 : cars
    population 2 : trucks '''
l1=1; l2=2; l=l1+l2 # average length of the vehicles in the j-th population [m]
rho1_jam=1; rho2_jam=0.5 # jam density [vehicles/m]
u1_max=1; u2_max=0.5 # free flow speed [*100 km/h]
L1=1; L2=1
L=L1+L2 # road length [m]
### l1*rho1_jam = l2*rho2_jam = 1
CFL=0.75    # CFL<1
# """ Viscosity"""
EPS=0.45
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


''' functions '''
''' 2-MFG-LWR '''
# # Greenshields desired speed
# def U(u_max,rho1,rho2): # u_max= u1_max or u2_max
#     return u_max*(1-(rho1*l1+rho2*l2))

# # Cost functional
# def f_mfg(u,u_max,rho1,rho2): # u= u1 or u2
#     return 0.5*((U(u_max,rho1,rho2)-u)**2) # 2-MFG-LWR

# def f_starp(u_max,p,rho1,rho2): # # u_max= u1_max or u2_max
#     return U(u_max,rho1,rho2)-p 
    
# def f_star(u_max,p,rho1,rho2): # p=Vx
#     return -0.5*(p**2)+U(u_max,rho1,rho2)*p  

# ############# For exact jacobian 
# def f_starp_p(u_max,p,r1,r2):
#     return -1.0
    
# def f_starp_r(u_max,p,r1,r2): 
#     if u_max==u1_max: l_=l1
#     if u_max==u2_max: l_=l2
#     return -u_max*l_

# def f_star_p(u_max,p,r1,r2,u):
#     return -p+U(u_max,r1,r2)

# def f_star_r(u_max,p,r1,r2,u):
#     if u_max==u1_max: l_=l1
#     if u_max==u2_max: l_=l2
#     return -u_max*l_*p

# def f_star_u(u_max,p,r1,r2,u):
#     return 0.0
# ##############

''' 2-MFG-Separable '''
def f_mfg(u,u_max,rho1,rho2):
    return 0.5*(u/u_max)**2-u/u_max+((rho1*l1+rho2*l2)/(rho1_jam*l1+rho2_jam*l2))

def f_starp(u_max,p,rho1,rho2):
    return max(min(u_max*(1-u_max*p),u_max),0)

def f_star(u_max,u,p,rho1,rho2): # p=Vx
    return f_mfg(u,u_max,rho1,rho2)+u*p 

########### For exact jacobian 
def f_starp_p(u_max,p,r1,r2):
    return -u_max**2
    
def f_starp_r(u_max,p,r1,r2):   
    return 0.0

def f_star_p(u_max,p,r1,r2,u):
    return u

def f_star_r(u_max,p,r1,r2,u): 
    if u_max==u1_max: l_=l1
    if u_max==u2_max: l_=l2
    return l_/(rho1_jam*l1+rho2_jam*l2)

def f_star_u(u_max,p,r1,r2,u): 
    return p+u/(u_max**2)-1/u_max
# #############

''' 2-MFG-Non Separable '''
# def f_mfg(u,u_max,rho1,rho2):
#     return 0.5*(u/u_max)**2-u/u_max+((rho1*l1+rho2*l2)/(rho1_jam*l1+rho2_jam*l2))*(u/u_max)

# def f_starp(u_max,p,rho1,rho2):
#     return max(min(u_max*(1-(rho1*l1+rho2*l2)/(rho1_jam*l1+rho2_jam*l2)-u_max*p),u_max),0)

# def f_star(u_max,u,p,rho1,rho2): # p=Vx
#     return f_mfg(u,u_max,rho1,rho2)+u*p

# ############ For exact jacobian 
# def f_starp_p(u_max,p,r1,r2):
#     return -u_max**2
    
# def f_starp_r(u_max,p,r1,r2): 
#     if u_max==u1_max: l_=l1
#     if u_max==u2_max: l_=l2
#     return -(u_max*l_)/(rho1_jam*l1+rho2_jam*l2)

# def f_star_p(u_max,p,r1,r2,u):
#     return u

# def f_star_r(u_max,p,r1,r2,u): 
#     if u_max==u1_max: l_=l1
#     if u_max==u2_max: l_=l2
#     return (l_/(rho1_jam*l1+rho2_jam*l2))*(u/u_max)

# def f_star_u(u_max,p,r1,r2,u): 
#     return p+u/(u_max**2)-1/u_max+((r1*l1+r2*l2)/(rho1_jam*l1+rho2_jam*l2))*(1/u_max)
# ##############


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

def VT(a): # Terminal cost
    return 0.0



