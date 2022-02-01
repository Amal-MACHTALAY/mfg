from pyccel.decorators import pure

'''************************ functions **********************************'''
@pure
def U(rho:'float', u_max:'float', rho_jam:'float'): # Greenshields desired speed
    return u_max*(1-rho/rho_jam)
@pure
def f_mfg_LWR(uu:'float',r:'float', u_max:'float', rho_jam:'float' ):
    return 0.5*((U(r, u_max, rho_jam)-uu)**2) # MFG-LWR
@pure
def f_star_p_LWR(p:'float', r:'float', u_max:'float', rho_jam:'float'): # 0<=u<=u_max
    return U(r, u_max, rho_jam)-p # MFG-LWR
@pure
def f_star_LWR(p:'float', r:'float', u_max:'float', rho_jam:'float'): # p=Vx
    return -0.5*(p**2)+U(r, u_max, rho_jam)*p # MFG-LWR

@pure
def rho_int(rho_a: float, rho_b: float, L: float, gamma: float, s: float):
    from numpy import exp
    return rho_a+(rho_b-rho_a)*exp(-0.5*((s-0.5*L)/gamma)**2) # 0<=rho<=rho_jam

@pure
def VT(a:'float'): # Terminal cost
    return 0.0
@pure
def r_idx(j:'int', n:'int', Nt:'int'):
    return (j-1)*(Nt+1)+n
@pure
def u_idx(j:'int', n:'int', Nt:'int', Nx:'int'):
    return (Nt+1)*Nx+(j-1)*Nt+n
@pure
def V_idx(j:'int',n:'int', Nt:'int', Nx:'int'):
    return (2*Nt+1)*Nx+(j-1)*(Nt+1)+n
@pure
def Fr_idx(j:'int', n:'int', Nt:'int'):
    return (j-1)*Nt+n
@pure
def Fu_idx(j:'int',n:'int', Nt:'int', Nx:'int'):
    return Nt*Nx+(j-1)*Nt+n
@pure
def FV_idx(j:'int',n:'int', Nt:'int', Nx:'int'):
    return 2*Nt*Nx+(j-1)*Nt+n
@pure
def Frint_idx(j:'int', Nt:'int', Nx:'int'):
    return 3*Nt*Nx+(j-1)
@pure
def FVter_idx(j:'int', Nt:'int', Nx:'int'):
    return 3*Nt*Nx+Nx+(j-1)

@pure
def f_star_p(p:'float', r:'float', u_max:'float', rho_jam:'float'): # 0<=u<=u_max
    return U(r, u_max, rho_jam)-p # MFG-LWR
@pure
def f_star(p:'float', r:'float', u_max:'float', rho_jam:'float'): # p=Vx
    return -0.5*(p**2)+U(r, u_max, rho_jam)*p # MFG-LWR


##########################Quadrature###########################################
def rho_int_1(s: float):
    """
    We don't really need this function.
    """
    rho_a = 0.05
    rho_b = 0.95
    L     = 1.
    gamma = 0.1

    return rho_int(rho_a, rho_b, L, gamma, s)

def integrate_rho_int_v2(a: float, b: float):
    """
    We use here the quadrature formula with order=7 and n_subdivision=10.
    in order to compute the points & weights, you can use gauss_legendre with
    the desired order then print the points and weights using :

    >>> print([x for x in xs])
    >>> print([w for w in ws])
    """

    # ...
    n_subdivision = 10
    order = 7

    quad_xs = [0.9602898564975362,
               0.7966664774136268,
               0.525532409916329,
               0.18343464249564978,
               -0.18343464249564978,
               -0.525532409916329,
               -0.7966664774136268,
               -0.9602898564975362]
    quad_ws = [0.10122853629036972,
               0.22238103445337445,
               0.31370664587788744,
               0.36268378337836193,
               0.36268378337836193,
               0.31370664587788744,
               0.22238103445337445,
               0.10122853629036972]
    # ...

    m = order + 1
    integral = 0.
    dx = (b-a) / n_subdivision
    left = a
    for ie in range(n_subdivision):
        right = left + dx
        # these formulae can be simplified
        c0 = 0.5*(left+right)
        c1 = 0.5*(right-left)
        for k in range(m):
            xk = c1*quad_xs[k] + c0
            wk = c1*quad_ws[k]

            integral += wk * rho_int_1(xk)

        left += dx

    return integral

####################################for interpolation##########################
@pure
def sol_to(old_Nt:'int', old_Nx:'int', sol:'float[:]', rho:'float[:,:]', u:'float[:,:]',V:'float[:,:]'): # solution 1D to 2D
    for j in range(0,old_Nx):
        for n in range(0,old_Nt):
            rho[j,n]=sol[j*(old_Nt+1)+n]
            u[j,n]=sol[(old_Nt+1)*old_Nx+j*old_Nt+n]
            V[j,n]=sol[(2*old_Nt+1)*old_Nx+j*(old_Nt+1)+n]
        rho[j,old_Nt]=sol[j*(old_Nt+1)+old_Nt]
        V[j,old_Nt]=sol[(2*old_Nt+1)*old_Nx+j*(old_Nt+1)+old_Nt]

@pure
def to_sol(new_Nt:'int', old_Nx:'int', sol:'float[:]', rho:'float[:,:]', u:'float[:,:]', V:'float[:,:]', multip:'int'):# solution 2D to 1D
    for j in range(0,multip*old_Nx):
        for n in range(0,multip*new_Nt):
            sol[j*(multip*new_Nt+1)+n]=rho[j,n]
            sol[(multip*new_Nt+1)*multip*old_Nx+j*multip*new_Nt+n]=u[j,n]
            sol[(2*multip*new_Nt+1)*multip*old_Nx+j*(multip*new_Nt+1)+n]=V[j,n]
        sol[j*(multip*new_Nt+1)+multip*new_Nt]=rho[j,multip*new_Nt]
        sol[(2*multip*new_Nt+1)*multip*old_Nx+j*(multip*new_Nt+1)+multip*new_Nt]=V[j,multip*new_Nt]

################################Jacobian#######################################

import numpy as np
def loc_to_glob(i,j,Nx,Nt,cord0,cord1):
        I=i+cord0*Nx    # Nx=(ex-sx+1)
        J=j+cord1*Nt    # Nt=(ey-sy+1)
        return I,J 
    
def block_vector_idx(idx,i,j,Nx,Nt): 
    # w_idx=[r_idx(i,j),u_idx(i,j),V_idx(i,j)]
        cmpt = 0
        idx[cmpt]=r_idx(i,j,Nt) 
        cmpt +=1
        if j!=Nt:
            idx[cmpt]=u_idx(i,j,Nt,Nx)
        cmpt +=1
        idx[cmpt]=V_idx(i,j,Nt,Nx)
        
def indx1(i,j,Nt): 
    return (i-1)*(Nt+1)+j

def block_F_eq_idx(idx,i,j,Nx,Nt): # F_eq : contain equations
        cmpt = 0
        idx[cmpt]=Fr_idx(i,j,Nt)  
        cmpt +=1
        idx[cmpt]=Fu_idx(i,j,Nt,Nx) 
        cmpt +=1
        idx[cmpt]=FV_idx(i,j,Nt,Nx)

def block_F_cond_idx(idx,j,Nx,Nt): # F_cond : contain intitial and terminal conditions
        cmpt = 0
        idx[cmpt]=Frint_idx(j,Nt,Nx) 
        cmpt +=1
        idx[cmpt]=FVter_idx(j,Nt,Nx)
        
def indx2(i,j,Nt): 
    return (i-1)*Nt+j
    
# @pure
def compute_jacobian(w:'float[:]', row:'int[:]', col:'int[:]', data:'float[:]', 
                     Nt:'int', Nx:'int', dt:'float', dx:'float', eps:'float', cord0:'int', cord1:'int', Nxg:'int', Ntg:'int'):
    ###### cord0=coord2d[0], cord1=coord2d[0], Nxg=npoints[0], Ntg=npoints[1], Nx=(ex-sx+1), Nt=(ey-sy+1)
    
    w_idx=np.zeros((Nt+1)*Nx) # local (if global : change Nx, Nt with Nxg, Ntg)
    for j in range(1,Nx+1):
        for n in range(0,Nt+1):
            block_vector_idx(w_idx[indx1(j,n,Nt)],j,n,Nx,Nt)               
    ## Then use : r_indx(j,n,Nt)=w_idx[indx1(j,n,Nt)][0], u_indx(j,n,Nt)=w_idx[indx1(j,n,Nt)][1], V_indx(j,n,Nt)=w_idx[indx1(j,n,Nt)][2]
        
    
    FF_eq_idx=np.zeros(Ntg*Nxg) # global (if local : change Nxg, Ntg with Nx, Nt)
    FF_cond_idx=np.zeros(Nxg)
    for j in range(1,Nxg+1):
        block_F_cond_idx(FF_cond_idx[j],j,Nxg,Ntg)
        for n in range(0,Ntg):
            block_F_eq_idx(FF_eq_idx[indx2(j,n,Ntg)],j,n,Nxg,Ntg)  
             
    ## Then use : Fr_indx(j,n)=FF_eq_idx[indx2(J,N,Ntg)][0], Fu_indx(j,n)=FF_eq_idx[indx2(J,N,Ntg)][1], FV_indx(j,n)=FF_eq_idx[indx2(J,N,Ntg)][2]
                # Fint_indx(j)=FF_cond_idx[J][0], Fter_indx(j)=FF_cond_idx[J][1]

    
    # if local use (j,n), if global use (J,N)
    
    # row[:] = 0; col[:] = 0.; data[:] = 0.
    cmpt = 0
    for n in range(0,Nt):
        for j in range(1,Nx+1): # 1,Nx-1
            J,N=loc_to_glob(j,n,Nx,Nt,cord0,cord1)
            
            # row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = w_idx[indx1(j,n+1,Nt)][0]; data[cmpt] = 1
            row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][0]; col[cmpt] = w_idx[indx1(j,n+1,Nt)][0]; data[cmpt] = 1
            cmpt +=1
            # row[cmpt] = Fu_idx(j,n,Nt,Nx); col[cmpt] = w_idx[indx1(j,n,Nt)][1]; data[cmpt] = 1
            row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][1]; col[cmpt] = w_idx[indx1(j,n,Nt)][1]; data[cmpt] = 1
            cmpt +=1
            # row[cmpt] = FV_idx(j,n,Nt,Nx); col[cmpt] = w_idx[indx1(j,n,Nt)][2]; data[cmpt] = -1
            row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][2]; col[cmpt] = w_idx[indx1(j,n,Nt)][2]; data[cmpt] = -1
            cmpt +=1
            # row[cmpt] = FV_idx(j,n,Nt,Nx); col[cmpt] = w_idx[indx1(j,n+1,Nt)][2]; data[cmpt] = 1-2*eps
            row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][2]; col[cmpt] = w_idx[indx1(j,n+1,Nt)][2]; data[cmpt] = 1-2*eps
            cmpt +=1
            
            if j!=1:
                # row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = w_idx[indx1(j-1,n,Nt)][0]; data[cmpt] = -(0.5*dt/dx)*w[u_idx(j-1,n,Nt,Nx)]-0.5
                row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][0]; col[cmpt] = w_idx[indx1(j-1,n,Nt)][0]; data[cmpt] = -(0.5*dt/dx)*w[w_idx[indx1(j-1,n,Nt)][1]]-0.5
                cmpt +=1
                # row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = w_idx[indx1(j-1,n,Nt)][0]; data[cmpt] = -(0.5*dt/dx)*w[r_idx(j-1,n,Nt)]
                row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][0]; col[cmpt] = w_idx[indx1(j-1,n,Nt)][0]; data[cmpt] = -(0.5*dt/dx)*w[w_idx[indx1(j-1,n,Nt)][0]]
                cmpt +=1
                # row[cmpt] = FV_idx(j,n,Nt,Nx); col[cmpt] = V_idx(j-1,n+1,Nt,Nx); data[cmpt] = eps
                row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][2]; col[cmpt] = w_idx[indx1(j-1,n+1,Nt)][2]; data[cmpt] = eps
                cmpt +=1
                
            if j==1:
                # row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = r_idx(Nx,n,Nt); data[cmpt] = (0.5*dt/dx)*w[u_idx(Nx,n,Nt,Nx)]-0.5
                row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][0]; col[cmpt] = w_idx[indx1(Nx,n,Nt)][0]; data[cmpt] = (0.5*dt/dx)*w[w_idx[indx1(Nx,n,Nt)][1]]-0.5
                cmpt +=1
                # row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = u_idx(Nx,n,Nt,Nx); data[cmpt] = -(0.5*dt/dx)*w[r_idx(Nx,n,Nt)]
                row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][0]; col[cmpt] = w_idx[indx1(Nx,n,Nt)][1]; data[cmpt] = -(0.5*dt/dx)*w[w_idx[indx1(Nx,n,Nt)][0]]
                cmpt +=1
                # row[cmpt] = FV_idx(j,n,Nt,Nx); col[cmpt] = V_idx(Nx,n+1,Nt,Nx); data[cmpt] = eps
                row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][2]; col[cmpt] = w_idx[indx1(Nx,n+1,Nt)][2]; data[cmpt] = eps
                cmpt +=1
           
            if j!=Nx:
                # row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = r_idx(j+1,n,Nt); data[cmpt] = (0.5*dt/dx)*w[u_idx(j+1,n,Nt,Nx)]-0.5
                row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][0]; col[cmpt] = w_idx[indx1(j+1,n,Nt)][0]; data[cmpt] = (0.5*dt/dx)*w[w_idx[indx1(j+1,n,Nt)][1]]-0.5
                cmpt +=1
                # row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = u_idx(j+1,n,Nt,Nx); data[cmpt] = (0.5*dt/dx)*w[r_idx(j+1,n,Nt)]
                row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][0]; col[cmpt] = w_idx[indx1(j+1,n,Nt)][1]; data[cmpt] = (0.5*dt/dx)*w[w_idx[indx1(j+1,n,Nt)][0]]
                cmpt +=1
                # row[cmpt] = FV_idx(j,n,Nt,Nx); col[cmpt] = V_idx(j+1,n+1,Nt,Nx); data[cmpt] = eps
                row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][2]; col[cmpt] = w_idx[indx1(j+1,n+1,Nt)][2]; data[cmpt] = eps
                cmpt +=1

            if j==Nx:
                # row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = r_idx(1,n,Nt); data[cmpt] = (0.5*dt/dx)*w[u_idx(1,n,Nt,Nx)]-0.5
                row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][0]; col[cmpt] = w_idx[indx1(1,n,Nt)][0]; data[cmpt] = (0.5*dt/dx)*w[w_idx[indx1(1,n,Nt)][1]]-0.5
                cmpt +=1
                # row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = u_idx(1,n,Nt,Nx); data[cmpt] = (0.5*dt/dx)*w[r_idx(1,n,Nt)]
                row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][0]; col[cmpt] = w_idx[indx1(1,n,Nt)][1]; data[cmpt] = (0.5*dt/dx)*w[w_idx[indx1(1,n,Nt)][0]]
                cmpt +=1
                # row[cmpt] = FV_idx(j,n,Nt,Nx); col[cmpt] = V_idx(1,n+1,Nt,Nx); data[cmpt] = eps
                row[cmpt] = FF_eq_idx[indx2(J,N,Ntg)][2]; col[cmpt] = w_idx[indx1(1,n+1,Nt)][2]; data[cmpt] = eps
                cmpt +=1
                
    
    for j in range(1,Nx+1):
        n=0; J,N=loc_to_glob(j,n,Nx,Nt,cord0,cord1)
        # row[cmpt] = Frint_idx(j,Nt,Nx); col[cmpt] = r_idx(j,0,Nt); data[cmpt] = 1
        row[cmpt] = FF_cond_idx[J][0]; col[cmpt] = w_idx[indx1(j,0,Nt)][0]; data[cmpt] = 1
        cmpt +=1
        n=Nt; J,N=loc_to_glob(j,n,Nx,Nt,cord0,cord1)
        # row[cmpt] = FVter_idx(j,Nt,Nx); col[cmpt] = V_idx(j,Nt,Nt,Nx); data[cmpt] = 1
        row[cmpt] = FF_cond_idx[J][1]; col[cmpt] = w_idx[indx1(j,Nt,Nt)][2]; data[cmpt] = 1
        cmpt +=1
        

    
def compute_FF(w:'float[:]', FF:'float[:]', Nt:'int', Nx:'int', dt:'float', dx:'float', eps:'float',
               u_max:'float', rho_jam:'float', x:'float[:]', cord0:'int', cord1:'int', Nxg:'int', Ntg:'int'):
    ###### cord0=coord2d[0], cord1=coord2d[0], Nxg=npoints[0], Ntg=npoints[1], Nx=(ex-sx+1), Nt=(ey-sy+1)
    
    w_idx=np.zeros((Nt+1)*Nx) # local (if global : change Nx, Nt with Nxg, Ntg)
    for j in range(1,Nx+1):
        for n in range(0,Nt+1):
            block_vector_idx(w_idx[indx1(j,n,Nt)],j,n,Nx,Nt)               
    ## Then use : r_indx(j,n,Nt)=w_idx[indx1(j,n,Nt)][0], u_indx(j,n,Nt)=w_idx[indx1(j,n,Nt)][1], V_indx(j,n,Nt)=w_idx[indx1(j,n,Nt)][2]
        
    
    FF_eq_idx=np.zeros(Ntg*Nxg) # global (if local : change Nxg, Ntg with Nx, Nt)
    FF_cond_idx=np.zeros(Nxg)
    for j in range(1,Nxg+1):
        block_F_cond_idx(FF_cond_idx[j],j,Nxg,Ntg)
        for n in range(0,Ntg):
            block_F_eq_idx(FF_eq_idx[indx2(j,n,Ntg)],j,n,Nxg,Ntg)  
             
    ## Then use : Fr_indx(j,n)=FF_eq_idx[indx2(J,N,Ntg)][0], Fu_indx(j,n)=FF_eq_idx[indx2(J,N,Ntg)][1], FV_indx(j,n)=FF_eq_idx[indx2(J,N,Ntg)][2]
                # Fint_indx(j)=FF_cond_idx[J][0], Fter_indx(j)=FF_cond_idx[J][1]

    
    # # if local use (j,n), if global use (J,N)

    for n in range(0,Nt):
        j=1; J,N=loc_to_glob(j,n,Nx,Nt,cord0,cord1)
        # F_rho , F[0]->F[Nt-1] ************** 1  
        # FF[Fr_idx(1,n,Nt)]=w[r_idx(1,n+1,Nt)]-0.5*(w[r_idx(Nx,n,Nt)]+w[r_idx(2,n,Nt)])\
        #     +(0.5*dt/dx)*(w[r_idx(2,n,Nt)]*w[u_idx(2,n,Nt,Nx)]-w[r_idx(Nx,n,Nt)]*w[u_idx(Nx,n,Nt,Nx)])
        FF[FF_eq_idx[indx2(J,N,Ntg)][0]]=w[w_idx[indx1(j,n+1,Nt)][0]]-0.5*(w[w_idx[indx1(Nx,n,Nt)][0]]+w[w_idx[indx1(j+1,n,Nt)][0]])\
            +(0.5*dt/dx)*(w[w_idx[indx1(j+1,n,Nt)][0]]*w[w_idx[indx1(j+1,n,Nt)][1]]-w[w_idx[indx1(Nx,n,Nt)][0]]*w[w_idx[indx1(Nx,n,Nt)][1]])   
        # F_u , F[Nt*Nx]->F[Nt*Nx+Nt-1] *********** 4 
        # FF[Fu_idx(1,n,Nt,Nx)]=w[u_idx(1,n,Nt,Nx)]-f_star_p((w[V_idx(1,n+1,Nt,Nx)]-w[V_idx(Nx,n+1,Nt,Nx)])/dx,w[r_idx(1,n,Nt)], u_max, rho_jam )
        FF[FF_eq_idx[indx2(J,N,Ntg)][1]]=w[w_idx[indx1(j,n,Nt)][1]]-f_star_p((w[w_idx[indx1(j,n+1,Nt)][2]]-w[w_idx[indx1(Nx,n+1,Nt)][2]])/dx,w[w_idx[indx1(j,n,Nt)][0]], u_max, rho_jam )
        
        # F_V , F[2*Nt*Nx]->F[2*Nt*Nx+Nt-1] *********** 7 
        # FF[FV_idx(1,n,Nt,Nx)]=w[V_idx(1,n+1,Nt,Nx)]-w[V_idx(1,n,Nt,Nx)]\
        #     +dt*f_star((w[V_idx(1,n+1,Nt,Nx)]-w[V_idx(Nx,n+1,Nt,Nx)])/dx, w[r_idx(1,n,Nt)], u_max, rho_jam)\
        #     +eps*(w[V_idx(2,n+1,Nt,Nx)]-2*w[V_idx(1,n+1,Nt,Nx)]+w[V_idx(Nx,n+1,Nt,Nx)])
        FF[FF_eq_idx[indx2(J,N,Ntg)][2]]=w[w_idx[indx1(j,n+1,Nt)][2]]-w[w_idx[indx1(j,n,Nt)][2]]\
            +dt*f_star((w[w_idx[indx1(j,n+1,Nt)][2]]-w[w_idx[indx1(Nx,n+1,Nt)][2]])/dx, w[w_idx[indx1(j,n,Nt)][0]], u_max, rho_jam)\
            +eps*(w[w_idx[indx1(j+1,n+1,Nt)][2]]-2*w[w_idx[indx1(j,n+1,Nt)][2]]+w[w_idx[indx1(Nx,n+1,Nt)][2]])
            
        j=Nx; J,N=loc_to_glob(j,n,Nx,Nt,cord0,cord1)
        # F_rho , F[Nt*Nx-Nt]->F[Nt*Nx-1] ********** 3 
        # FF[Fr_idx(Nx,n,Nt)]=w[r_idx(Nx,n+1,Nt)]-0.5*(w[r_idx(Nx-1,n,Nt)]+w[r_idx(1,n,Nt)])\
        #     +(0.5*dt/dx)*(w[r_idx(1,n,Nt)]*w[u_idx(1,n,Nt,Nx)]-w[r_idx(Nx-1,n,Nt)]*w[u_idx(Nx-1,n,Nt,Nx)])
        FF[FF_eq_idx[indx2(J,N,Ntg)][0]]=w[w_idx[indx1(j,n,Nt)][0]]-0.5*(w[w_idx[indx1(j-1,n,Nt)][0]]+w[w_idx[indx1(1,n,Nt)][0]])\
            +(0.5*dt/dx)*(w[w_idx[indx1(1,n,Nt)][0]]*w[w_idx[indx1(1,n,Nt)][1]]-w[w_idx[indx1(j-1,n,Nt)][0]]*w[w_idx[indx1(j-1,n,Nt)][1]])    
        # F_u , F[2*Nt*Nx-Nt]->F[2*Nt*Nx-1] ********* 6 
        # FF[Fu_idx(Nx,n,Nt,Nx)]=w[u_idx(Nx,n,Nt,Nx)]-f_star_p((w[V_idx(Nx,n+1,Nt,Nx)]-w[V_idx(Nx-1,n+1,Nt,Nx)])/dx,w[r_idx(Nx,n,Nt)], u_max, rho_jam)
        FF[FF_eq_idx[indx2(J,N,Ntg)][1]]=w[w_idx[indx1(j,n,Nt)][1]]-f_star_p((w[w_idx[indx1(j,n+1,Nt)][2]]-w[w_idx[indx1(j-1,n+1,Nt)][2]])/dx,w[w_idx[indx1(j,n,Nt)][0]], u_max, rho_jam)
        # F_V , F[3*Nt*Nx-Nt]->F[3*Nt*Nx-1] ********** 9 
        # FF[FV_idx(Nx,n,Nt,Nx)]=w[V_idx(Nx,n+1,Nt,Nx)]-w[V_idx(Nx,n,Nt,Nx)]\
        #     +dt*f_star((w[V_idx(Nx,n+1,Nt,Nx)]-w[V_idx(Nx-1,n+1,Nt,Nx)])/dx, w[r_idx(Nx,n,Nt)], u_max, rho_jam)\
        #     +eps*(w[V_idx(1,n+1,Nt,Nx)]-2*w[V_idx(Nx,n+1,Nt,Nx)]+w[V_idx(Nx-1,n+1,Nt,Nx)])
        FF[FF_eq_idx[indx2(J,N,Ntg)][2]]=w[w_idx[indx1(j,n+1,Nt)][2]]-w[w_idx[indx1(j,n,Nt)][2]]\
            +dt*f_star((w[w_idx[indx1(j,n+1,Nt)][2]]-w[w_idx[indx1(j-1,n+1,Nt)][2]])/dx, w[w_idx[indx1(j,n,Nt)][0]], u_max, rho_jam)\
            +eps*(w[w_idx[indx1(1,n+1,Nt)][2]]-2*w[w_idx[indx1(j,n+1,Nt)][2]]+w[w_idx[indx1(j-1,n+1,Nt)][2]])
            
    for j in range(2,Nx):
        for n in range(0,Nt):
            J,N=loc_to_glob(j,n,Nx,Nt,cord0,cord1)
            # F_rho , F[Nt]->F[Nt*Nx-Nt-1] ************ 2 
            # FF[Fr_idx(j,n,Nt)]=w[r_idx(j,n+1,Nt)]-0.5*(w[r_idx(j-1,n,Nt)]+w[r_idx(j+1,n,Nt)])\
            #     +(0.5*dt/dx)*(w[r_idx(j+1,n,Nt)]*w[u_idx(j+1,n,Nt,Nx)]-w[r_idx(j-1,n,Nt)]*w[u_idx(j-1,n,Nt,Nx)])
            FF[FF_eq_idx[indx2(J,N,Ntg)][0]]=w[w_idx[indx1(j,n+1,Nt)][0]]-0.5*(w[w_idx[indx1(j-1,n,Nt)][0]]+w[w_idx[indx1(j+1,n,Nt)][0]])\
                +(0.5*dt/dx)*(w[w_idx[indx1(j+1,n,Nt)][0]]*w[w_idx[indx1(j+1,n,Nt)][1]]-w[w_idx[indx1(j-1,n,Nt)][0]]*w[w_idx[indx1(j-1,n,Nt)][1]])   
            # F_u , F[Nt*Nx+Nt]->F[2*Nt*Nx-Nt-1] *********** 5 
            # FF[Fu_idx(j,n,Nt,Nx)]=w[u_idx(j,n,Nt,Nx)]\
            #     -f_star_p((w[V_idx(j,n+1,Nt,Nx)]-w[V_idx(j-1,n+1,Nt,Nx)])/dx,w[r_idx(j,n,Nt)], u_max, rho_jam)
            FF[FF_eq_idx[indx2(J,N,Ntg)][1]]=w[w_idx[indx1(j,n,Nt)][1]]\
                -f_star_p((w[w_idx[indx1(j,n+1,Nt)][2]]-w[w_idx[indx1(j-1,n+1,Nt)][2]])/dx,w[w_idx[indx1(j,n,Nt)][0]], u_max, rho_jam)    
            # F_V , F[2*Nt*Nx+Nt]->F[3*Nt*Nx-Nt-1] ********* 8 
            # FF[FV_idx(j,n,Nt,Nx)]=w[V_idx(j,n+1,Nt,Nx)]-w[V_idx(j,n,Nt,Nx)]\
            #     +dt*f_star((w[V_idx(j,n+1,Nt,Nx)]-w[V_idx(j-1,n+1,Nt,Nx)])/dx,w[r_idx(j,n,Nt)], u_max, rho_jam)\
            #     +eps*(w[V_idx(j+1,n+1,Nt,Nx)]-2*w[V_idx(j,n+1,Nt,Nx)]+w[V_idx(j-1,n+1,Nt,Nx)])
            FF[FF_eq_idx[indx2(J,N,Ntg)][2]]=w[w_idx[indx1(j,n+1,Nt)][2]]-w[w_idx[indx1(j,n,Nt)][2]]\
                +dt*f_star((w[w_idx[indx1(j,n+1,Nt)][2]]-w[w_idx[indx1(j-1,n+1,Nt)][2]])/dx,w[w_idx[indx1(j,n,Nt)][0]], u_max, rho_jam)\
                +eps*(w[w_idx[indx1(j+1,n+1,Nt)][2]]-2*w[w_idx[indx1(j,n+1,Nt)][2]]+w[w_idx[indx1(j-1,n+1,Nt)][2]])
                
        # F_rho_int , F[3*Nt*Nx+1]->F[3*Nt*Nx+Nx-2] ********** 11
        # FF[Frint_idx(j,Nt,Nx)]=w[r_idx(j,0,Nt)]-(1/dx)*integrate_rho_int_v2(x[j-1],x[j])
        FF[FF_cond_idx[J][0]]=w[w_idx[indx1(j,0,Nt)][0]]-(1/dx)*integrate_rho_int_v2(x[j-1],x[j])
        # F_V_ter , F[3*Nt*Nx+Nx+1]->F[3*Nt*Nx+2*Nx-2] ********* 14
        # FF[FVter_idx(j,Nt,Nx)]=w[V_idx(j,Nt,Nt,Nx)]-VT(x[j])
        FF[FF_cond_idx[J][1]]=w[w_idx[indx1(j,Nt,Nt)][2]]-VT(x[j]) 
    j=1; n=Nt; J,N=loc_to_glob(j,n,Nx,Nt,cord0,cord1)
    # F_rho_int , F[3*Nt*Nx] ********* 10
    # FF[Frint_idx(1,Nt,Nx)]=w[r_idx(1,0,Nt)]-(1/dx)*integrate_rho_int_v2(x[0],x[1])
    FF[FF_cond_idx[J][0]]=w[w_idx[indx1(j,0,Nt)][0]]-(1/dx)*integrate_rho_int_v2(x[0],x[1])
    # F_V_ter , F[3*Nt*Nx+Nx] *********** 13 
    # FF[FVter_idx(1,Nt,Nx)]=w[V_idx(1,Nt,Nt,Nx)]-VT(x[1])
    FF[FF_cond_idx[J][1]]=w[w_idx[indx1(j,Nt,Nt)][2]]-VT(x[1])
    
    j=Nx; n=Nt; J,N=loc_to_glob(j,n,Nx,Nt,cord0,cord1)
    # F_rho_int , F[3*Nt*Nx+Nx-1] ********* 12
    # FF[Frint_idx(Nx,Nt,Nx)]=w[r_idx(Nx,0,Nt)]-(1/dx)*integrate_rho_int_v2(x[Nx-1],x[Nx])
    FF[FF_cond_idx[J][0]]=w[w_idx[indx1(j,0,Nt)][0]]-(1/dx)*integrate_rho_int_v2(x[Nx-1],x[Nx])
    # F_V_ter , F[3*Nt*Nx+2*Nx-1] ************** 15
    # FF[FVter_idx(Nx,Nt,Nx)]=w[V_idx(Nx,Nt,Nt,Nx)]-VT(x[Nx])
    FF[FF_cond_idx[J][1]]=w[w_idx[indx1(j,Nt,Nt)][2]]-VT(x[Nx])