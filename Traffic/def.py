# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 23:41:14 2021

@author: MACHTALAY AMAL
"""

apt=42  # change apt to update random samples

def location_int(pdf, x_mfg):
    """ generate initial locations from a given initial density pdf """
    random_loc=sample(pdf,N)
    loc=np.zeros(len(random_loc))
    for j in range(len(random_loc)):
        loc[j]=closest(x_mfg,random_loc[j])
    return loc

def closest(lst, val):
    """ Find Closest number in a list """
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-val))]

def sample(pdf, nbr_samples, n_max=10**6):
    np.random.seed(apt)
    """ genearte a list of random samples from a given pdf
    suggests random samples between 0 and L 
    and accepts-rejects the suggestion with probability pdf(x) 
    """
    samples=[]
    n=0
    while len(samples)<nbr_samples and n<n_max:
        x=np.random.uniform(low=0,high=L)
        new_sample=pdf(x)
        assert new_sample>=0 and new_sample<=1
        if np.random.uniform(low=0,high=1) <=new_sample:
            samples += [x]
        n+=1
    return sorted(samples)

def find_index(lst, val):
    """ find the index of val in lst"""
    ind='nan'
    for i in range(len(lst)):
        if lst[i]==val:
            ind=i
            break
    return ind

def control(u_mfg, x_mfg, t_mfg):
    X=np.zeros((N,len(t_mfg)))
    v=np.zeros((N,len(t_mfg)))
    X[:,0]=location_int(rho_int, x_mfg)
    for n in t_mfg:
        for i in range(N):
            ind=find_index(x_mfg,X[i,n])
            v[i,n]=u_mfg[ind,n]
            new_x=X[i,n]+dt*v[i,n]
            if n<N-1:
                X[i,n+1]=closest(x_mfg,new_x)
    return X,v
        
    
