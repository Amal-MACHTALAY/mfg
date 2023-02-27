#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:50:16 2022

@author: amal
"""

'''
    Approximation of jacobian matrix by ignoring the forward-backward coupling  parts

'''

from indx_funcs import r_idx, u_idx, V_idx, Fr_idx, Fu_idx, FV_idx, Frint_idx, FVter_idx

''' CE : Explicit Lax-Friedrichs scheme 
    HJB : Implicit upwind difference scheme
'''

def jacobian(w,row,col,data,Nt,Nx,dt,dx,eps): 
    cmpt = 0
    row[:] = 0; col[:] = 0; data[:] = 0.
    
    for j in range(1,Nx+1):
        ''' l (left) <- j-1
            r (right) <- j+1
        '''
        if j>1: l=j-1
        else: l=Nx
        if j<Nx: r=j+1
        else: r=1
        
        row[cmpt]=Frint_idx(j,Nt,Nx); col[cmpt]=r_idx(j,0,Nt); data[cmpt]=1
        cmpt +=1
        row[cmpt]=FVter_idx(j,Nt,Nx); col[cmpt]=V_idx(j,Nt,Nt,Nx); data[cmpt]=1
        cmpt +=1
        
        for n in range(0,Nt):
            # F_rho / rho
            row[cmpt]=Fr_idx(j,n,Nt); col[cmpt]=r_idx(j,n+1,Nt); data[cmpt]=1
            cmpt +=1
            row[cmpt]=Fr_idx(j,n,Nt); col[cmpt]=r_idx(l,n,Nt); data[cmpt]=-(0.5*dt/dx)*w[u_idx(l,n,Nt,Nx)]-0.5
            cmpt +=1
            row[cmpt]=Fr_idx(j,n,Nt); col[cmpt]=r_idx(r,n,Nt); data[cmpt]=(0.5*dt/dx)*w[u_idx(r,n,Nt,Nx)]-0.5
            cmpt +=1
            # F_rho / u
            row[cmpt]=Fr_idx(j,n,Nt); col[cmpt]=u_idx(l,n,Nt,Nx); data[cmpt]=-(0.5*dt/dx)*w[r_idx(l,n,Nt)]
            cmpt +=1
            row[cmpt]=Fr_idx(j,n,Nt); col[cmpt]=u_idx(r,n,Nt,Nx); data[cmpt]=(0.5*dt/dx)*w[r_idx(r,n,Nt)]
            cmpt +=1
            # F_u / u
            row[cmpt]=Fu_idx(j,n,Nt,Nx); col[cmpt]=u_idx(j,n,Nt,Nx); data[cmpt]=1
            cmpt +=1
            # F_V / V
            row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=V_idx(j,n,Nt,Nx); data[cmpt]=-1/dt
            cmpt +=1
            row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=V_idx(l,n+1,Nt,Nx); data[cmpt]=eps
            cmpt +=1
            row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=V_idx(j,n+1,Nt,Nx); data[cmpt]=1/dt-2*eps
            cmpt +=1
            row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=V_idx(r,n+1,Nt,Nx); data[cmpt]=eps
            cmpt +=1      
            
    return 0


''' CE : Explicit Lax-Friedrichs scheme 
    HJB : Implicit central difference scheme
'''

# def jacobian(w,row,col,data,Nt,Nx,dt,dx,eps): 
#     cmpt = 0
#     row[:] = 0; col[:] = 0; data[:] = 0.
    
#     for j in range(1,Nx+1):
#         ''' l (left) <- j-1
#             r (right) <- j+1
#         '''
#         if j>1: l=j-1
#         else: l=Nx
#         if j<Nx: r=j+1
#         else: r=1
        
#         row[cmpt]=Frint_idx(j,Nt,Nx); col[cmpt]=r_idx(j,0,Nt); data[cmpt]=1
#         cmpt +=1
#         row[cmpt]=FVter_idx(j,Nt,Nx); col[cmpt]=V_idx(j,Nt,Nt,Nx); data[cmpt]=1
#         cmpt +=1
        
#         for n in range(0,Nt):
#             # F_rho / rho
#             row[cmpt]=Fr_idx(j,n,Nt); col[cmpt]=r_idx(j,n+1,Nt); data[cmpt]=1
#             cmpt +=1
#             row[cmpt]=Fr_idx(j,n,Nt); col[cmpt]=r_idx(l,n,Nt); data[cmpt]=-(0.5*dt/dx)*w[u_idx(l,n,Nt,Nx)]-0.5
#             cmpt +=1
#             row[cmpt]=Fr_idx(j,n,Nt); col[cmpt]=r_idx(r,n,Nt); data[cmpt]=(0.5*dt/dx)*w[u_idx(r,n,Nt,Nx)]-0.5
#             cmpt +=1
#             # F_rho / u
#             row[cmpt]=Fr_idx(j,n,Nt); col[cmpt]=u_idx(l,n,Nt,Nx); data[cmpt]=-(0.5*dt/dx)*w[r_idx(l,n,Nt)]
#             cmpt +=1
#             row[cmpt]=Fr_idx(j,n,Nt); col[cmpt]=u_idx(r,n,Nt,Nx); data[cmpt]=(0.5*dt/dx)*w[r_idx(r,n,Nt)]
#             cmpt +=1
#             # F_u / u
#             row[cmpt]=Fu_idx(j,n,Nt,Nx); col[cmpt]=u_idx(j,n,Nt,Nx); data[cmpt]=1
#             cmpt +=1
#             # F_V / V
#             row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=V_idx(j,n,Nt,Nx); data[cmpt]=-1/dt
#             cmpt +=1
#             row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=V_idx(l,n+1,Nt,Nx); data[cmpt]=eps
#             cmpt +=1
#             row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=V_idx(j,n+1,Nt,Nx); data[cmpt]=1/dt-2*eps
#             cmpt +=1
#             row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=V_idx(r,n+1,Nt,Nx); data[cmpt]=eps
#             cmpt +=1      
            
#     return 0