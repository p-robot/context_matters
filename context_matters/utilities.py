#!/usr/bin/env python3
"""
List of utility functions for the context_matters module
"""

import numpy as np

def epsilon_soft(Qs, actions, epsilon):
    """
    
    See figure 5.6 of Sutton and Barto.  
    
    """
    
    # Find the best action(s)
    astar = (Qs == np.min(Qs))
    
    # Probability of choosing the best action and non-best action
    p_astar = 1 - epsilon + epsilon/len(actions)
    p_a = epsilon/len(actions)
    
    probabilities = p_a * np.ones(len(actions))
    probabilities[astar] = p_astar
    
    # Normalise (in case there are ties for the current 'best' action).  
    probabilities = probabilities/np.sum(probabilities)
    
    return probabilities



def NeighbourhoodDetail(D, outer, inner = 0):
    """
    Extension of the 'Neighbourhood' function above but giving the ID of the 
    IP farm that's closest to each suscept farm that's earmarked for control.  
    
    Return boolean array of whether distances (in D) are within a particular 
    combination of radii (inner, outer), and also the row number where such a 
    condition is satisfied.  
    
    {True : (D_{i,j} > inner) & (D_{i,j} < outer) }
    {i : (D_{i,j} > inner) & (D_{i,j} < outer) }
    
    INPUT
    
    D : distance matrix of dimension 
        (sum(detected_mask) x sum(not_detected_mask))
    
    outer: 
        outer radius of vaccination ring
    
    inner: 
        inner radius of vaccination ring
    
    OUTPUT
    
    candidates: boolean np.array of length sum(not_detected_mask) having True
         values for which feedlots are candidates for vaccination.  More
         specifically, True for which feedlots 1) don't have fmd infected
         status, 2) are within the vaccination ring for SOME infected premises,
         and 3) are within the inner vaccination ring for NONE of the infected
         premises.  
    
    mindist2ip
    which_ip_min (how are ties dealt with in this case?  )
    
    
    Radii cut-off are inclusive - on the border means the feedlot is included
    in the ring.  
    
    Note: 
    The notation used in np.empty_like is new in Numpy version 1.6.0 so 
    therefore any computer or HPCU must set python to 2.7.3 or later.
    
    EXAMPLE
    
    import scipy.spatial as sp
    import numpy as np
    import utilities as utils
    
    # Generate points on a grid
    x = np.linspace(-2, 2, 11)
    XX, YY = np.meshgrid(x, x)
    XX = XX.ravel(order = 'C'); YY = YY.ravel(order = 'C')
    coords = zip(XX, YY)
    
    # Generate a vector of the infection status of farms (anything > 0 is inf)
    status = np.zeros(len(coords))
    cntr = [60, 64]
    status[cntr] = 1
    
    # Calculate a distance matrix between points
    D = sp.distance.cdist(coords, coords)
    
    # Subset the distance matrix to be infected (rows) - susceptible (cols)
    D_inf_to_sus = D[cntr,:]
    D_inf_to_sus = np.delete(D_inf_to_sus, cntr, 1)
    
    # If using iPython, time the implementation
    %timeit utils.NeighbourhoodDetail(D_inf_to_sus,outer=1.9,inner=1.1)
    
    # Calculate the 'neighbours' within each infected premises
    ca, mi, wh = utils.NeighbourhoodDetail(D_inf_to_sus,outer=1.9, inner=1.1)
    
    # The output is in terms of susceptible/infected farms, so need these:
    sus = np.where(status == 0)[0]
    inf = np.where(status > 0)[0]
    
    # Plot farms, those in the 'neighbourhood' are in orange, the numbers show
    # index of the which of the infected farm that designated them as a cull
    # candidate, numbers in blue show the minimum distance to any IP
    
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    
    # Green for each farm
    ax.scatter(XX, YY, c = 'green')
    
    # Red for infected farms
    ax.scatter(XX[cntr], YY[cntr], c = 'red')
    
    # Orange for those in the 'neighbourhood'
    ax.scatter(XX[sus[ca]], YY[sus[ca]], c = 'orange', \
        s = 200, lw = 0, marker = 'o')
    
    for a, b, c in zip(XX[sus[ca]], YY[sus[ca]], [str(x) for x in wh]):
        ax.text(a, b, c, va = 'center', ha = 'center')
    
    strings = ['\n\n'+str(np.round(x,2)) for x in mi]
    for a, b, c in zip(XX[sus[ca]], YY[sus[ca]], strings):
        ax.text(a, b, c, va = 'center', ha = 'center', \
        color = 'blue', fontsize = 8)
    
    plt.show()
    
    """
    
    nbhd = (D <= outer)&(D >= inner)
    candidates = nbhd.any(axis = 0)
    
    # Make sure no feedlot is within 'inner' radius to any infected premises
    too_close = ((D < inner) & (D != 0)).any(axis = 0)
    
    # Combine these two conditions
    candidates[too_close] = False
    
    which_ip_min = D[:, candidates].argmin(axis = 0)
    mindist2ip = D[:, candidates].min(axis = 0)
    
    return(candidates, mindist2ip, which_ip_min)

