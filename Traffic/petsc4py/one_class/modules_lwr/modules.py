from pyccel.decorators import pure

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


'''************************ functions LWR **********************************'''
@pure
def UU(rho:'float', u_max:'float', rho_jam:'float'): # Greenshields desired speed
    return u_max*(1-rho/rho_jam)
@pure
def f_starp(p:'float', r:'float', u_max:'float', rho_jam:'float'): # 0<=u<=u_max # f_star_p
    return UU(r, u_max, rho_jam)-p # MFG-LWR
@pure
def f_star(p:'float', r:'float', u:'float', u_max:'float', rho_jam:'float'): # p=Vx
    return -0.5*(p**2)+UU(r, u_max, rho_jam)*p # MFG-LWR
@pure
def f_starp_p(p:'float', r:'float', u_max:'float', rho_jam:'float'):  # f_star_p_der_arg1
    return -1.0
@pure   
def f_starp_r(p:'float', r:'float', u_max:'float', rho_jam:'float'): # f_star_p_der_arg2
    return -(u_max/rho_jam)
@pure
def f_star_p(p:'float', r:'float', u:'float', u_max:'float', rho_jam:'float'):  # f_star_der_arg1
    return -p+UU(r, u_max, rho_jam)
@pure
def f_star_r(p:'float', r:'float', u:'float', u_max:'float', rho_jam:'float'):  # f_star_der_arg2
    return -(u_max/rho_jam)*p
@pure
def f_star_u(p:'float', r:'float', u:'float', u_max:'float', rho_jam:'float'):
    return 0.0
    
    
"""************************************************************************************"""

@pure
def rho_int(rho_a: float, rho_b: float, L: float, gamma: float, s: float):
    from numpy import exp
    return rho_a+(rho_b-rho_a)*exp(-0.5*((s-0.5*L)/gamma)**2) # 0<=rho<=rho_jam

@pure
def VT(a:'float'): # Terminal cost
    return 0.0
@pure
def r_idx(j:'int', n:'int', Nt:'int'):  # j : 1 -> Nx  AND n : 0 -> Nt
    return (j-1)*(Nt+1)+n              # 0 -> Nx*Nt+Nx-1
@pure
def u_idx(j:'int', n:'int', Nt:'int', Nx:'int'): # j : 1 -> Nx  AND n : 0 -> Nt-1
    return (Nt+1)*Nx+(j-1)*Nt+n        # Nx*Nt+Nx -> 2*Nx*Nt+Nx-1
@pure
def V_idx(j:'int',n:'int', Nt:'int', Nx:'int'): # j : 1 -> Nx  AND n : 0 -> Nt
    return (2*Nt+1)*Nx+(j-1)*(Nt+1)+n     # 2*Nx*Nt+Nx -> 3*Nx*Nt+2*Nx-1
@pure
def Fr_idx(j:'int', n:'int', Nt:'int'):  # j : 1 -> Nx  AND n : 0 -> Nt-1
    return (j-1)*Nt+n                   # 0 -> Nx*Nt-1
@pure
def Fu_idx(j:'int',n:'int', Nt:'int', Nx:'int'):  # j : 1 -> Nx  AND n : 0 -> Nt-1
    return Nt*Nx+(j-1)*Nt+n                      # Nx*Nt -> 2*Nx*Nt-1
@pure
def FV_idx(j:'int',n:'int', Nt:'int', Nx:'int'):  # j : 1 -> Nx  AND n : 0 -> Nt-1
    return 2*Nt*Nx+(j-1)*Nt+n                    # 2*Nx*Nt -> 3*Nx*Nt-1
@pure
def Frint_idx(j:'int', Nt:'int', Nx:'int'):  # j : 1 -> Nx  
    return 3*Nt*Nx+(j-1)                        # 3*Nx*Nt -> 3*Nx*Nt+Nx-1
@pure
def FVter_idx(j:'int', Nt:'int', Nx:'int'):  # j : 1 -> Nx  
    return 3*Nt*Nx+Nx+(j-1)                  # 3*Nx*Nt+Nx -> 3*Nx*Nt+2*Nx-1


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
def from_1d_to_2d(old_Nt:'int', old_Nx:'int', sol:'float[:]', rho:'float[:,:]', u:'float[:,:]',V:'float[:,:]'): # solution 1D to 2D
    for j in range(0,old_Nx):
        for n in range(0,old_Nt+1):
            rho[j,n]=sol[j*(old_Nt+1)+n]
            if n<old_Nt:
                u[j,n]=sol[(old_Nt+1)*old_Nx+j*old_Nt+n]
            V[j,n]=sol[(2*old_Nt+1)*old_Nx+j*(old_Nt+1)+n]

@pure
def from_2d_to_1d(new_Nt:'int', old_Nx:'int', sol:'float[:]', rho:'float[:,:]', u:'float[:,:]', V:'float[:,:]', multip:'int'):# solution 2D to 1D
    for j in range(0,multip*old_Nx):
        for n in range(0,multip*new_Nt+1):
            sol[j*(multip*new_Nt+1)+n]=rho[j,n]
            if n<multip*new_Nt:
                sol[(multip*new_Nt+1)*multip*old_Nx+j*multip*new_Nt+n]=u[j,n]
            sol[(2*multip*new_Nt+1)*multip*old_Nx+j*(multip*new_Nt+1)+n]=V[j,n]
        
################################Jacobian#######################################
# @pure
def compute_jacobian(w:'float[:]', row:'int[:]', col:'int[:]', data:'float[:]',u_max:'float', rho_jam:'float', 
                     Nt:'int', Nx:'int', dt:'float', dx:'float', eps:'float', ranges:'int[:,:]'):
    
    cmpt = 0
    row[:] = 0; col[:] = 0; data[:] = 0.
    
    for j in range(ranges[1][0], ranges[1][1]+1): # 0,Nx+1
        if j != 0:
            ''' l (left) <- j-1
                r (right) <- j+1
            '''
            if j>1: l=j-1
            else: l=Nx
            if j<Nx: r=j+1
            else: r=1
            
            row[cmpt] = Frint_idx(j,Nt,Nx); col[cmpt] = r_idx(j,0,Nt); data[cmpt] = 1
            cmpt +=1
            row[cmpt] = FVter_idx(j,Nt,Nx); col[cmpt] = V_idx(j,Nt,Nt,Nx); data[cmpt] = 1
            cmpt +=1
            
            for n in range(ranges[0][0], ranges[0][1]): #0, Nt
                # F_rho / rho
                row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = r_idx(j,n+1,Nt); data[cmpt] = 1.
                cmpt +=1
                row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = r_idx(l,n,Nt); data[cmpt] = -(0.5*dt/dx)*w[u_idx(l,n,Nt,Nx)]-0.5
                cmpt +=1
                row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = r_idx(r,n,Nt); data[cmpt] = (0.5*dt/dx)*w[u_idx(r,n,Nt,Nx)]-0.5
                cmpt +=1
                # F_rho / u
                row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = u_idx(r,n,Nt,Nx); data[cmpt] = (0.5*dt/dx)*w[r_idx(r,n,Nt)]
                cmpt +=1
                row[cmpt] = Fr_idx(j,n,Nt); col[cmpt] = u_idx(l,n,Nt,Nx); data[cmpt] = -(0.5*dt/dx)*w[r_idx(l,n,Nt)]
                cmpt +=1
                # F_u /rho
                row[cmpt]=Fu_idx(j,n,Nt,Nx); col[cmpt]=r_idx(j,n,Nt); 
                data[cmpt]=-f_starp_r((w[V_idx(r,n+1,Nt,Nx)]-w[V_idx(j,n+1,Nt,Nx)])/dx,w[r_idx(j,n,Nt)],u_max, rho_jam)
                cmpt +=1
                # F_u / u
                row[cmpt] = Fu_idx(j,n,Nt,Nx); col[cmpt] = u_idx(j,n,Nt,Nx); data[cmpt] = 1.
                cmpt +=1
                # F_u / V
                valu = (1/dx)*f_starp_p((w[V_idx(r,n+1,Nt,Nx)]-w[V_idx(j,n+1,Nt,Nx)])/dx,w[r_idx(j,n,Nt)], u_max, rho_jam)
                row[cmpt]=Fu_idx(j,n,Nt,Nx); col[cmpt]=V_idx(r,n+1,Nt,Nx); 
                data[cmpt]=-valu
                cmpt +=1
                row[cmpt]=Fu_idx(j,n,Nt,Nx); col[cmpt]=V_idx(j,n+1,Nt,Nx); 
                data[cmpt]=valu
                cmpt +=1
                # F_V / rho
                row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=r_idx(j,n,Nt); 
                data[cmpt]=f_star_r((w[V_idx(r,n+1,Nt,Nx)]-w[V_idx(j,n+1,Nt,Nx)])/dx,w[r_idx(j,n,Nt)],w[u_idx(j,n,Nt,Nx)], u_max, rho_jam)
                cmpt +=1
                # F_V / u
                row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=u_idx(j,n,Nt,Nx); 
                data[cmpt]=f_star_u((w[V_idx(r,n+1,Nt,Nx)]-w[V_idx(j,n+1,Nt,Nx)])/dx,w[r_idx(j,n,Nt)],w[u_idx(j,n,Nt,Nx)], u_max, rho_jam)
                cmpt +=1
                # F_V / V
                row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=V_idx(j,n,Nt,Nx); data[cmpt]=-1./dt
                cmpt +=1
                row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=V_idx(j,n+1,Nt,Nx); 
                data[cmpt]=1/dt-(1/dx)*f_star_p((w[V_idx(r,n+1,Nt,Nx)]-w[V_idx(j,n+1,Nt,Nx)])/dx,w[r_idx(j,n,Nt)],w[u_idx(j,n,Nt,Nx)], u_max, rho_jam)#-2*eps
                cmpt +=1
                row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=V_idx(r,n+1,Nt,Nx); 
                data[cmpt]=(1/dx)*f_star_p((w[V_idx(r,n+1,Nt,Nx)]-w[V_idx(j,n+1,Nt,Nx)])/dx,w[r_idx(j,n,Nt)],w[u_idx(j,n,Nt,Nx)], u_max, rho_jam)#+eps
                cmpt +=1      
                # row[cmpt]=FV_idx(j,n,Nt,Nx); col[cmpt]=V_idx(l,n+1,Nt,Nx); data[cmpt]=eps
                # cmpt +=1  

            
     
def compute_FF(w:'float[:]', FF:'float[:]', Nt:'int', Nx:'int', dt:'float', dx:'float', eps:'float',
               u_max:'float', rho_jam:'float', x:'float[:]', ranges:'int[:,:]', RANK:'int'):
    # print('****************************\n',RANK, ranges[0][0], ranges[0][1], ranges[1][0], ranges[1][1])
    
    FF[:] = 0.
    
    for j in range(ranges[1][0], ranges[1][1]+1): # 0,Nx+1
        if j != 0:
            ''' l (left) <- j-1
                r (right) <- j+1
            '''
            if j>1: l=j-1
            else: l=Nx
            if j<Nx: r=j+1
            else: r=1
            
            # F_rho_int , F[3*Nt*Nx]->F[3*Nt*Nx+Nx-1] ********** 4
            FF[Frint_idx(j,Nt,Nx)]=w[r_idx(j,0,Nt)]-(1/dx)*integrate_rho_int_v2(x[j-1],x[j])
            # F_V_ter , F[3*Nt*Nx+Nx]->F[3*Nt*Nx+2*Nx-1] ********* 5
            FF[FVter_idx(j,Nt,Nx)]=w[V_idx(j,Nt,Nt,Nx)]-VT(x[j])
            
            for n in range(ranges[0][0], ranges[0][1]): # 0,Nt
                # F_rho , F[0]->F[Nt*Nx-1] ************ 1
                FF[Fr_idx(j,n,Nt)]=w[r_idx(j,n+1,Nt)]-0.5*(w[r_idx(l,n,Nt)]+w[r_idx(r,n,Nt)])\
                    +(0.5*dt/dx)*(w[r_idx(r,n,Nt)]*w[u_idx(r,n,Nt,Nx)]-w[r_idx(l,n,Nt)]*w[u_idx(l,n,Nt,Nx)])
                # F_u , F[Nt*Nx]->F[2*Nt*Nx-1] *********** 2 
                FF[Fu_idx(j,n,Nt,Nx)]=w[u_idx(j,n,Nt,Nx)]-f_starp((w[V_idx(r,n+1,Nt,Nx)]-w[V_idx(j,n+1,Nt,Nx)])/dx,w[r_idx(j,n,Nt)], u_max, rho_jam)
                # F_V , F[2*Nt*Nx]->F[3*Nt*Nx-1] ********* 3 
                FF[FV_idx(j,n,Nt,Nx)]=(w[V_idx(j,n+1,Nt,Nx)]-w[V_idx(j,n,Nt,Nx)])/dt\
                    +f_star((w[V_idx(r,n+1,Nt,Nx)]-w[V_idx(j,n+1,Nt,Nx)])/dx,w[r_idx(j,n,Nt)],w[u_idx(j,n,Nt,Nx)], u_max, rho_jam)#\
                    # +eps*(w[V_idx(r,n+1,Nt,Nx)]-2*w[V_idx(j,n+1,Nt,Nx)]+w[V_idx(l,n+1,Nt,Nx)])
