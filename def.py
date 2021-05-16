# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 23:41:14 2021

@author: MACHTALAY AMAL
"""

def closest(lst, val):
    """ Find Closest number in a list """
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-val))]

def inverse_sample_function(dist, pnts, x_min=-100, x_max=100, n=1e5, 
                            **kwargs):
        
    x = np.linspace(x_min, x_max, int(n))
    cumulative = np.cumsum(dist(x, **kwargs))
    cumulative -= cumulative.min()
    f = interp1d(cumulative/cumulative.max(), x)
        
    return f(np.random.random(pnts))

def rejection_sampling_1D( f_x, g_var, g_mean, M, num_samples ):
    ''' does rejection sampling using a Gaussian distribution as  the
    underlying density, given a target density function f_x and a number of
    samples we want generated.'''
    g_x = lambda x: 1/np.sqrt(2*np.pi*g_var)*np.exp(-(x-g_mean)**2/(2*g_var))
    good_samples = 0;
    # do shit until we have enough samples
    keepers = np.zeros(num_samples)
    while good_samples < num_samples:
        # generate a new sample using our underlying distribution
        new_samp = np.random.randn(1)*np.sqrt(g_var)+g_mean
        # generate a new uniform
        new_check = np.random.rand(1)
        # first, let's make sure M is big enough
        while f_x(new_samp)/g_x(new_samp)/M > 1:
            M = 2*M
            return rejection_sampling_1D( f_x, g_var, M, num_samples )
        # now let's check if our stuff actually works.
        if new_check >= f_x(new_samp)/g_x(new_samp)/M:
            keepers[good_samples] = new_samp
            good_samples = good_samples + 1
        else:
            continue
    return good_samples
