#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:45:03 2022

@author: amal
"""

import numpy as np
from scipy import integrate
from indx_funcs import r_idx, u_idx, V_idx, Fr_idx, Fu_idx, FV_idx, Frint_idx, FVter_idx

def integral(a,b,rho_int): 
    x2 = lambda x: rho_int(x)
    I=integrate.quad(x2, a, b)
    return I[0]

''' CE : Explicit forward Lax-Friedrichs scheme 
    HJB : Exlicit backward upwind difference scheme
'''
def newton_func(w,f_starp,f_star,rho_int,Nt,Nx,dt,dx,eps,x,VT):
    FF=np.zeros(3*Nt*Nx+2*Nx)
    for j in range(1,Nx+1):
        
        ''' l (left) <- j-1
            r (right) <- j+1
        '''
        if j>1: l=j-1
        else: l=Nx
        if j<Nx: r=j+1
        else: r=1
        
        # F_rho_int , F[3*Nt*Nx]->F[3*Nt*Nx+Nx-1] ********** 4
        FF[Frint_idx(j,Nt,Nx)]=w[r_idx(j,0,Nt)]-(1/dx)*integral(x[j-1],x[j],rho_int)
        # F_V_ter , F[3*Nt*Nx+Nx]->F[3*Nt*Nx+2*Nx-1] ********* 5
        FF[FVter_idx(j,Nt,Nx)]=w[V_idx(j,Nt,Nt,Nx)]-VT(x[j])
        for n in range(0,Nt):
            # F_rho , F[0]->F[Nt*Nx-1] ************ 1
            FF[Fr_idx(j,n,Nt)]=w[r_idx(j,n+1,Nt)]-0.5*(w[r_idx(l,n,Nt)]+w[r_idx(r,n,Nt)])\
                +(0.5*dt/dx)*(w[r_idx(r,n,Nt)]*w[u_idx(r,n,Nt,Nx)]-w[r_idx(l,n,Nt)]*w[u_idx(l,n,Nt,Nx)])
            # F_u , F[Nt*Nx]->F[2*Nt*Nx-1] *********** 2 
            FF[Fu_idx(j,n,Nt,Nx)]=w[u_idx(j,n,Nt,Nx)]-f_starp((w[V_idx(r,n+1,Nt,Nx)]-w[V_idx(j,n+1,Nt,Nx)])/dx,w[r_idx(j,n,Nt)])
            # F_V , F[2*Nt*Nx]->F[3*Nt*Nx-1] ********* 3 
            FF[FV_idx(j,n,Nt,Nx)]=(w[V_idx(j,n+1,Nt,Nx)]-w[V_idx(j,n,Nt,Nx)])/dt\
                +f_star((w[V_idx(r,n+1,Nt,Nx)]-w[V_idx(j,n+1,Nt,Nx)])/dx,w[r_idx(j,n,Nt)],w[u_idx(j,n,Nt,Nx)])\
                +eps*(w[V_idx(r,n+1,Nt,Nx)]-2*w[V_idx(j,n+1,Nt,Nx)]+w[V_idx(l,n+1,Nt,Nx)])
            
    return FF  

