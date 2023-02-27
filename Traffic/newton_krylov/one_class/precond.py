#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 18:57:04 2022

@author: amal
"""

import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import csc_matrix

    
def get_preconditioner(a,Nt,Nx,dt,dx,eps,jacobian):
    row=np.zeros(14*Nt*Nx+2*Nx) # exact jacobian : 14*Nt*Nx+2*Nx
    col=np.zeros(14*Nt*Nx+2*Nx) # approx jacobain : 10*Nt*Nx+2*Nx
    data=np.zeros(14*Nt*Nx+2*Nx)
    jacobian(a,row,col,data,Nt,Nx,dt,dx,eps)
    shap=(3*Nt*Nx+2*Nx,3*Nt*Nx+2*Nx)
    Jac1 = csc_matrix((data, (row, col)),shape = shap)
    # the *incomplete LU* decomposition
    J_ilu = spla.splu(Jac1)
    # J_ilu = spla.spilu(Jac1)
    M_x = lambda r: J_ilu.solve(r)
    M = spla.LinearOperator(shap, M_x)
    # M=spla.inv(Jac1)
    return M
