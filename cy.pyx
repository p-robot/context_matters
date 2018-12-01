"""
Cython version of some functions related to FMD individual-based model.  

Run the following from the home directory after changes are made:
python setup.py build_ext --inplace
"""

#from __future__ import division
import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int64_t DTYPE_t
ctypedef np.float64_t DTYPE_d

# Define expontential function from C math lib.
cdef extern from "math.h":
    double exp(double arg)

def exp_func(arg):
    return exp(arg)

def exp_func_array(np.ndarray[DTYPE_d, ndim = 1] arg):
    return exp(arg)


def dist_sqeuclidean1(
    np.ndarray[DTYPE_d, ndim = 1] x1, 
    np.ndarray[DTYPE_d, ndim = 1] x2
    ):
    return (x1[0] - x2[0])*(x1[0] - x2[0]) + (x1[1] - x2[1])*(x1[1] - x2[1])


def dist_sqeuclidean(
    np.ndarray[DTYPE_d, ndim = 1] x1, 
    np.ndarray[DTYPE_d, ndim = 1] x2
    ):
    return (x1[0] - x2[0])**2 + (x1[1] - x2[1])**2


def Kernel(
    np.ndarray[DTYPE_d, ndim = 1] x1, 
    np.ndarray[DTYPE_d, ndim = 1] x2
    ):
    x = dist_sqeuclidean1(x1, x2)
    return(0.3)

def KernelExponential(
    np.ndarray[DTYPE_d, ndim = 1] x1, 
    np.ndarray[DTYPE_d, ndim = 1] x2
    ):
    
    x = dist_sqeuclidean1(x1, x2)
    K = 4.8* exp_func(-2.4*np.sqrt(x))
    return(K)


def UKKernel(
    np.ndarray[DTYPE_d, ndim = 1] x1, 
    np.ndarray[DTYPE_d, ndim = 1] x2
    ):
    
    cdef float K, x
    cdef float *P = [-9.2123e-5, 9.5628e-4, 3.3966e-3, -3.3687e-2, \
        -1.30519e-1, -0.609262, -3.231772]
    
    x = dist_sqeuclidean1(x1, x2)
    
    K = P[0]*x**6 + P[1]*x**5 + P[2]*x**4 + P[3]*x**3 + P[4]*x**2+P[5]*x+P[6]
    K = exp_func(K)
    
    if (x < 0.0138):
        K = 0.3093
    elif (x > 60*60):
        K = 0.0
    
    return(K)


def UKKernelDist(np.ndarray[DTYPE_d, ndim = 2] x):
    """
    With squared distance (km) as an input
    Using Horner's method.  
    """
    cdef float *P = [-9.2123e-5, 9.5628e-4, 3.3966e-3, -3.3687e-2, \
        -1.30519e-1, -0.609262, -3.231772]
    cdef np.ndarray[DTYPE_d, ndim = 2] K
    
    K = P[6] + x*(P[5] + x*(P[4] + x*(P[3] + x*(P[2] + x*(P[1] + x*P[0])))))
    K = np.exp(K)
    
    K[x < 0.0138] = 0.3093
    K[x > 60*60] = 0.0
    
    return(K)


def cyIterateWho(
    np.ndarray[DTYPE_t, ndim = 1] status, 
    np.ndarray[DTYPE_d, ndim = 2] KDIST, 
    np.ndarray[DTYPE_d, ndim = 1] Suscept, 
    np.ndarray[DTYPE_d, ndim = 1] Transmiss, 
    np.ndarray[DTYPE_t, ndim = 1] delay_latency, 
    float spark):
    
    """
    Function to evolve the infection through time, recording whom infected who
    
    Parameters
    ----------
      - status : np.array (int)
          infection status of each premises (days since infection event)
      - KDIST : np.array (float; 2D)
          distance matrix evaluated according to the spatial kernel
      - Suscept : np.array (float)
          array of susceptibility of each premises
      - Transmiss : np.array (float)
          array of transmissibility of each premises
      - delay_latency : np.array(int)
          latency of the disease in days (no. of days that premises are latently
          infected after infection event)
      - spark : float
          baseline level of infection
    
    Returns
    -------
      - status: np.array
          updated state vector; infection status of each premises
        
      - NEW_INFECTIONS: np.array
          indices of newly infected premises
        
      - INFECTORS: np.array
          indices of premises that infected the newly infected premises
    """
    
    # Define the type of all variables used
    cdef int slen = status.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 1] Event = np.zeros(slen, dtype = DTYPE)
    cdef np.ndarray[DTYPE_t, ndim = 1] INF = np.where((status >= delay_latency))[0]
    cdef np.ndarray[DTYPE_t, ndim = 1] SUS = np.where(status == 0)[0]
    
    cdef DTYPE_d TRANSSUSCEPTK, P
    cdef np.ndarray[DTYPE_t, ndim = 2] NN
    cdef np.ndarray[DTYPE_t, ndim = 2] INFECTION_EVENT
    cdef np.ndarray[DTYPE_t, ndim = 1] INFECTORS, NEW_INFECTIONS
    cdef int nINF = INF.shape[0], nSUS = SUS.shape[0], i, j, k
    
    # Create an empty array of who infected who
    INFECTORS = NEW_INFECTIONS = np.array([], dtype = DTYPE)
    
    if((nINF > 0) & (nSUS > 0)):
        
        INFECTION_EVENT = np.zeros((nINF, nSUS), dtype = DTYPE)
        # Loop through all infectious farms.  
        for i in range(nINF):
            
            for j in range(nSUS):
                
                TRANSSUSCEPTK = Transmiss[INF[i]] * Suscept[SUS[j]] * KDIST[INF[i], SUS[j]]
                # Find the probability of each susceptible farm being infected from
                # each infected farm
                
                P = 1 - exp_func(-TRANSSUSCEPTK + spark)
                
                # Generate random numbers for the infection events
                INFECTION_EVENT[i, j] = 1*(P > np.random.rand())
                
        #########################
        # FINDING THE INFECTORS #
        #########################
        
        # Find which susceptibles became infected
        NEW_INFECTIONS = np.sum(INFECTION_EVENT, axis = 0)
        INFECTORS = np.ones(len(NEW_INFECTIONS), dtype = DTYPE)
        
        if NEW_INFECTIONS.any():
        
            # Find which premises actually infected the newly infected premises
            
            # Subset matrix of all possible infections to show just new infections
            NN = INFECTION_EVENT[:, NEW_INFECTIONS > 0]
            
            # Loop through all the infection events (finding the infector)
            for k, n in enumerate(NN.T):
                I = np.where(n)[0]
                INFECTORS[k] = INF[np.random.choice(I)]
            
            #INFECTORS = np.array([np.where(n)[0] for n in NN.T])
            
            # Choose one infector randomly (in cases where multiple IPs
            # could have infected a susceptible premises)
            #INFECTORS = np.array([INF[np.random.choice(n)] for n in INFECTORS])
            
            # Expand back to the size of the array
            # Find the index of susceptible farms which become infected
            NEW_INFECTIONS = SUS[(np.sum(INFECTION_EVENT, axis = 0) > 0)]
            
            # Update the event vector
            Event[NEW_INFECTIONS] = 1
    
    status[status > 0] += 1
    
    # Update the status vector
    status = status + Event
    
    # Output 
    return(status, NEW_INFECTIONS, INFECTORS)


def hull_area(np.ndarray[DTYPE_d, ndim=1] x, np.ndarray[DTYPE_d, ndim=1] y):
    """
    Calculate area of a simple 2D polygon.  

    Cython function to calculate the area of a 2D convex hull given input 
    numpy arrays of the x, y coordinates of the vertices enclosing the convex
    hull.  The coordinates must be given counterclockwise and there should be 
    no holes in the hull.  Input coordinates should be for a simple polygon.  
    This uses the shoelace formula to calculate the area of the polygon.  

    Args:
        x: x coordinates of the vertices of the polygon
        y: y coordinates of the vertices of the polygon

    Working variables:
        N: number of vertices given
        pt: index of the points in the array of vertices
        ptn: index of the next point in the array of vertices (loops back to
            the start once the final vertex is read)
        A: area of the 2D hull

    Returns: 
        Area of the 2D hull defined by the input coordinates of the vertices.  

    Example:
        import scipy.spatial as sp
        from am4fmd.data.load_data import circ3km
        from am4fmd.cy import hull_area

        hull = sp.ConvexHull(zip(circ3km.x, circ3km.y))
        x = hull.points[hull.vertices][:,0]
        y = hull.points[hull.vertices][:,1]
        area = hull_area(x, y)
    """

    cdef int N = len(x) # else try x.shape[0]
    cdef int pt, ptn
    cdef DTYPE_d A = 0.0

    for pt in range(N):
        ptn = (pt + 1) % N
        A += x[pt] * y[ptn]
        A -= x[ptn] * y[pt]
    return A / 2.0
