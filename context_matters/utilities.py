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


def Coords2Grid(x, y, grid_n_x, grid_n_y):
    """
    Determine grid locations of farms from x, y coordinates
    Returns the grid of farms in row-major order.  
    
    Parameters
    ----------
        x : np.array; float
            x coordinates of farms
        
        y : np.array; float
            y coordinates of farms
        
        grid_n_x: int
            number of desired grid squares in the horizontal direction
            
        grid_n_y: int
            number of desired grid squares in the vertical direction
        
    Returns
    -------
        grid : np.array
            row-major grid number of farm
    
    Example
    -------
    from matplotlib import pyplot as plt
    import numpy as np
    
    # Generate a 4 x 4 grid over a landscape of 10 farms
    N = 100; dim = 20.
    xx = np.random.rand(N)*dim; yy = np.random.rand(N)*dim
    grid = Coords2Grid(xx, yy, 4, 4)
    
    plt.scatter(xx, yy, color = 'grey')
    for x, y, g in zip(xx, yy, grid):
        plt.text(x, y, g)
    
    for x, y, g in zip(xx, yy, gx):
        plt.text(x - 0.5, y, g, color = 'blue')
    
    for x, y, g in zip(xx, yy, gy):
        plt.text(x + 0.5, y, g, color = 'red')
    
    plt.show()
    
    """
    
    # Create boundaries for the grid squares
    xbins = np.linspace(
        start = np.min(x), 
        stop = x.max(), 
        num = grid_n_x + 1)
    
    ybins = np.linspace(
        start = np.max(y), 
        stop = np.min(y), 
        num = grid_n_y + 1)
    
    # Find the x- and y-specific grid squares
    grid_x = np.digitize(x, xbins, right = True)
    
    grid_y = np.digitize(y, ybins, right = False)
    
    # Adjust top and bottom x/y grid values due to numerical rounding
    grid_x[grid_x == 0] = 1
    grid_y[grid_y == 0] = 1
    
    # Adjust for python index numbering
    grid_x = grid_x - 1
    grid_y = grid_y - 1
    
    # Convert from 2D coords to a 1D index (row-major order)
    grid = np.ravel_multi_index((grid_x, grid_y), \
        dims = (grid_n_x, grid_n_y), order = 'F')
    
    return grid


def calc_grid_probs(grid, grid_n_x, grid_n_y, Suscept, kernel_function, \
    d_perimeter_x, d_perimeter_y):
    """
    Calculate the probability of infection from one grid square to the next.
    (quicker than version above)
    
    Notes
    -----
    Grid and Suscept MUST be ordered according to grid number!  
    
    Could also use pairwise chebyshev distances (scipy.spatial.distance.cdist)
    instead of using ogrid etc (below) if the x/y lengths of the grid cells
    are the same.  
    
    
    Parameters
    ----------
        grid: np.array
    
        grid_n_x: np.array
    
        grid_n_y: np.array
    
        Suscept: np.array
    
        kernel_function: function
    
        d_perimeter_x : float
        
        d_perimeter_y : float
        
        
    Returns
    -------
        
        MaxRate
        
        Num
        
        first_
        
        last_
        
        max_sus
        
        Dist2all
        
        KDist2all
    
    
    Example
    -------
    # Calculate grid-level probabilities of infection for the UK data
    
    import epi_speed_test.py.fmd as fmd
    kernel = fmd.UKKernel
    out = calc_grid_probs(grid, 100, 100, np.random.rand(len(grid)), kernel, \
        np.ptp(demo.easting.values), np.ptp(demo.northing.values))
    
    
    Updates
    -------
    
    # 6th Oct 2014 Changed use of grid.max()+1 to grid_n_x*grid_n_y, WP
    # Feb 2017 Changed to use np.ogrid etc, WP
    """
    
    NG = grid_n_x*grid_n_y
    
    # Find the farm indices per grid square
    fpg = [np.where(grid == i)[0] for i in range(NG)]
    # Perhaps be made quicker by only looking at the grids with farms in them.
    
    # Maximum susceptibility of a feedlot per grid square (0 for empty grid)
    max_sus = np.asarray([np.max(Suscept[x]) if len(x) > 0 else 0 for x in fpg])
    
    # Find the number of farms per grid square
    Num = np.array([len(i) for i in fpg])
    
    # First farm in grid, -1 if the list is empty
    first_ = np.asarray([x[0] if len(x) > 0 else -1 for x in fpg])
    
    # Last farm in grid, -2 if the list is empty
    last_ = np.asarray([x[-1] if len(x) > 0 else -2 for x in fpg])
    
    NCOL = float(grid_n_x); NROW = float(grid_n_y)
    HEIGHT = d_perimeter_y/NROW; WIDTH = d_perimeter_x/NCOL
    
    # Calculate the minimum sq-distance between each grid square
    # X and Y position of each grid square
    Y, X = np.ogrid[0:NROW*NCOL, 0:NROW*NCOL]
    HDIST = np.abs(Y%NCOL - X%NCOL) - 1
    VDIST = np.abs(Y/NCOL - X/NCOL) - 1
    HDIST[HDIST < 0] = 0; VDIST[VDIST < 0] = 0
    Dist2all = np.add((HDIST*WIDTH)**2, (VDIST*HEIGHT)**2)
    
    KDist2all = kernel_function(Dist2all) # very slow
    
    # Could perhaps use Numpy broadcasting for this ... 
    MaxRate = np.multiply(KDist2all, np.tile(max_sus, (grid_n_x*grid_n_y, 1)))
    
    # All grids with no farms should have infinite rate of input infection
    # All grids with no farms should have infinite rate of susceptibility
    MaxRate[Num == 0] = np.inf
    MaxRate[:, Num == 0] = np.inf
    
    # Set rate from one grid to itself to infinity.  
    np.fill_diagonal(MaxRate, np.inf)
    
    return(MaxRate, Num, first_, last_, max_sus, Dist2all, KDist2all)
