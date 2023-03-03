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

''' CE : Explicit Lax-Friedrichs scheme 
    HJB : Implicit upwind difference scheme
'''
def newton_func(w,f_starp,f_star,rho_int1,rho_int2,Nt,Nx,dt,dx,eps,x,VT,u1_max,u2_max):
    FF=np.zeros(3*Nt*(2*Nx)+2*(2*Nx))
    for j in range(1,2*Nx+1):
        ''' l (left) <- j-1
            r (right) <- j+1
        '''
        
        if j!=1 and j!=Nx+1: l=j-1
        elif j==1: l=Nx
        else: l=2*Nx
        if j!=Nx and j!=2*Nx: r=j+1
        elif j==Nx: r=1
        else: r=Nx+1
        
        # F_rho_int , F[3*Nt*(2*Nx)]->F[3*Nt*(2*Nx)+(2*Nx)-1] ********** 4
        if j<Nx+1:
            u_max=u1_max
            s1=j; s2=j+Nx
            FF[Frint_idx(j,Nt,Nx)]=w[r_idx(j,0,Nt)]-(1/dx)*integral(x[s1-1],x[s1],rho_int1)
        else:
            u_max=u2_max
            s1=j-Nx; s2=j
            FF[Frint_idx(j,Nt,Nx)]=w[r_idx(j,0,Nt)]-(1/dx)*integral(x[s1-1],x[s1],rho_int2)
        # F_V_ter , F[3*Nt*(2*Nx)+(2*Nx)]->F[3*Nt*(2*Nx)+2*(2*Nx)-1] ********* 5
        FF[FVter_idx(j,Nt,Nx)]=w[V_idx(j,Nt,Nt,Nx)]-VT(x[s1])
        for n in range(0,Nt):
            # F_rho , F[0]->F[Nt*(2*Nx)-1] ************ 1
            FF[Fr_idx(j,n,Nt)]=w[r_idx(j,n+1,Nt)]-0.5*(w[r_idx(l,n,Nt)]+w[r_idx(r,n,Nt)])\
                +(0.5*dt/dx)*(w[r_idx(r,n,Nt)]*w[u_idx(r,n,Nt,Nx)]-w[r_idx(l,n,Nt)]*w[u_idx(l,n,Nt,Nx)])
            # F_u , F[Nt*(2*Nx)]->F[2*Nt*(2*Nx)-1] *********** 2 
            FF[Fu_idx(j,n,Nt,Nx)]=w[u_idx(j,n,Nt,Nx)]\
                -f_starp(u_max,(w[V_idx(r,n+1,Nt,Nx)]-w[V_idx(j,n+1,Nt,Nx)])/dx,w[r_idx(s1,n,Nt)],w[r_idx(s2,n,Nt)])
            # F_V , F[2*Nt*(2*Nx)]->F[3*Nt*(2*Nx)-1] ********* 3 
            FF[FV_idx(j,n,Nt,Nx)]=(w[V_idx(j,n+1,Nt,Nx)]-w[V_idx(j,n,Nt,Nx)])/dt\
                +f_star(u_max,w[u_idx(j,n,Nt,Nx)],(w[V_idx(r,n+1,Nt,Nx)]-w[V_idx(j,n+1,Nt,Nx)])/dx,w[r_idx(s1,n,Nt)],w[r_idx(s2,n,Nt)])\
                +eps*(w[V_idx(r,n+1,Nt,Nx)]-2*w[V_idx(j,n+1,Nt,Nx)]+w[V_idx(l,n+1,Nt,Nx)])
            
    return FF  






  
    
    