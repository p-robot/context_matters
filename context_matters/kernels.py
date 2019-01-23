#!/usr/bin/env python3
"""
Spatial transmission kernels, evaluating the squared Euclidean distance between two points.  

The following kernels are included: 
    PowerKernel: 
        A power-law kernel object
    ExponentialKernel: 
        An exponential kernel object
    UKKernel: 
         Polynomial kernel from the 2001 UK outbreak (Keeling and Rohani, 2008)
    KeelingKernel: 
         Empirical kernel from data from Keeling et al. (2001).  
    JewellKernel: 
        Cauchy kernel from Jewell et al. (2009).  
    Constant kernel: 
        Kernel with constant return value for any distance (default 0.3).  
    Zero kernel: 
        Kernel with zero return value for any distance.  
    DiggleKernel: 
        Kernel from Diggle (2006)
    ChowellKernel: 
        Exponential kernel from Chowell et al. (2006)
    DeardonKernel:
        Power-law kernel (truncated) from Deardon et al. (2010)


References
----------
* Keeling and Rohani (2008) Modeing Infectious Diseases.  Princeton Press.  
* Jewell et al. (2009) Predicting undetected infections during the 2007 foot-and-mouth disease outbreak.  J. R. Soc. Interface (2009) 6, 1145 –1151.  
* Diggle (2006) Spatio-temporal point processes, partial likelihood, foot and mouth disease.  Statistical Methods in Medical Research 2006; 15: 325–336
* Chowell et al. (2006) The role of spatial mixing in the spread of foot-and-mouth disease.  Preventive Veterinary Medicine 73 (2006) 297–314
* Deardon et al. (2010) Inference for individual-level models of infectious diseases in large populations. Stat. Sin. 20 (1) 239-261
"""

import numpy as np
from . import core
from context_matters.data.load_data import kkern

class PowerKernel(core.Kernel):
    """
    Class representing a power-law kernel
    """
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self._k = kwargs.pop('k', 0.41)
        self._k = float(self._k)
        
    def __call__(self, dist_squared):
        return self.k/dist_squared
    
    @property
    def k(self):
        "Coefficient for power law kernel"
        return self._k

class ExponentialKernel(core.Kernel):
    """
    Class representing an exponential kernel
    """
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self._g = kwargs.pop('g', 0.32386809538389649)#4.8)
        self._h = kwargs.pop('h', 0.0023630128749078491)#2.4)
        
    def __call__(self, dist_squared):
        
        # Check if scalar, adjust as needed
        dist_squared = np.asarray(dist_squared)
        is_scalar = False if dist_squared.ndim > 0 else True
        dist_squared.shape = (1,)*(1-dist_squared.ndim) + dist_squared.shape
        
        K = self.g*np.exp(-self.h*np.sqrt(dist_squared))
        
        return K
    
    @property
    def g(self):
        "Multiplicative coefficient for exponential kernel"
        return self._g
    @property
    def h(self):
        "Exp coefficient for exponential kernel"
        return self._h


class ExponentialKernelTruncated(core.Kernel):
    """
    Class representing an exponential kernel
    """
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self._g = kwargs.pop('g', 0.32386809538389649)#4.8)
        self._h = kwargs.pop('h', 0.0023630128749078491)#2.4)
        
        self._k0 = kwargs.pop('k0', 0.3093)
        self._delta0 = kwargs.pop('delta0', 0.0138)
        self._delta_max = kwargs.pop('delta_max', 60*60.)
        
    def __call__(self, dist_squared):
        
        # Check if scalar, adjust as needed
        dist_squared = np.asarray(dist_squared)
        is_scalar = False if dist_squared.ndim > 0 else True
        dist_squared.shape = (1,)*(1-dist_squared.ndim) + dist_squared.shape
        
        K = self.g*np.exp(-self.h*np.sqrt(dist_squared))
        
        K[(dist_squared < self.delta0)] = self.k0
        K[(dist_squared >= self.delta_max)] = 0
        return K
    
    @property
    def g(self):
        "Multiplicative coefficient for exponential kernel"
        return self._g
    @property
    def h(self):
        "Exp coefficient for exponential kernel"
        return self._h
    @property
    def k0(self):
        return self._k0
    @property
    def delta0(self):
        return self._delta0
    @property
    def delta_max(self):
        return self._delta_max


class UKKernel(core.Kernel):
    """Class representing the UK kernel
    
    Kernel takes input as a distance-squared and returns the kernel evaluated
    at that point.  The UKKernel object represents the kernel from the UK 2001
    outbreak.  Taken from Keeling and Rohani (2008), program 7.6.  
    
    Args:
        dist_squared: a distance squared
    
    Returns:
        The UK kernel function evaluated at the input squared distance.  
    """
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self._k0 = kwargs.pop('k0', 0.3093)
        self._scaling_factor = kwargs.pop('scaling_factor', 1.0)
        self._delta0 = kwargs.pop('delta0', 0.0138)
        self._delta_max = kwargs.pop('delta_max', 60*60.)
    
    def __call__(self, dist_squared):
        
        # Check if scalar, adjust as needed
        dist_squared = np.asarray(dist_squared)
        is_scalar = False if dist_squared.ndim > 0 else True
        dist_squared.shape = (1,)*(1-dist_squared.ndim) + dist_squared.shape
        
        P = np.array([-9.2123*10**(-5), 9.5628*10**(-4), \
            3.3966*10**(-3), -3.3687*10**(-2), \
            -1.30519*10**(-1), -0.609262, -3.231772])
        
        K = np.exp(np.polyval(P,dist_squared))
        
        K[(dist_squared < self.delta0)] = self.k0
        K[(dist_squared >= self.delta_max)] = 0
        
        K = self.scaling_factor * K
        
        return(K if not is_scalar else K[0])
        
    @property
    def k0(self):
        return self._k0
    @property
    def scaling_factor(self):
        return self._scaling_factor
    @property
    def delta0(self):
        return self._delta0
    @property
    def delta_max(self):
        return self._delta_max


class ChowellKernel(core.Kernel):
    """
    Class representing the kernel from Chowell et al. (2006)
    An exponential kernel
    """
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self._q = kwargs.pop('q', 1.07)
        
    def __call__(self, dist_squared):
        return np.exp(-self.q*np.sqrt(dist_squared))
    
    @property
    def q(self):
        "Coefficient for Chowell kernel"
        return self._q


class DiggleKernel(core.Kernel):
    """
    Class representing the kernel described in Diggle (2006).  
    
    Includes a 'spark' term, rho.  
    """
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self._nu = kwargs.pop('nu', 1.0)
        self._kappa = kwargs.pop('kappa', 0.5)
        self._phi = kwargs.pop('phi', 0.41)
        self._rho = kwargs.pop('rho', 1.3E-4)
        
    def __call__(self, dist_squared):
        K = - np.power(np.sqrt(dist_squared)/self.phi, self.kappa)
        return self.nu * np.exp(K) + self.rho
    
    @property
    def nu(self):
        """Baseline probability of infection at zero distance"""
        return self._nu
    @property
    def kappa(self):
        return self._kappa
    @property
    def phi(self):
        return self._phi
    @property
    def rho(self):
        return self._rho


class JewellKernel(core.Kernel):
    """
    Cauchy distance kernel as used in Jewell et al. (2009).  
    """
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self._beta5 = kwargs.pop('beta5', 0.75)
        self._k0 = kwargs.pop('k0', 0.3093)
        self._delta0 = kwargs.pop('delta0', 0.0138)
        self._delta_max = kwargs.pop('delta_max', 60*60.)
        self._scaling_factor = kwargs.pop('scaling_factor', 1.0)
    
    def __call__(self, dist_squared):
        """
        Value of 0.75 is the approximate value of the mode of the posterior
        of beta_5 (estimated from figure 1f in the supplementary information).
        """
        
        dist_squared = np.asarray(dist_squared)
        is_scalar = False if dist_squared.ndim > 0 else True
        dist_squared.shape = (1,)*(1-dist_squared.ndim) + dist_squared.shape
        
        K = self.beta5**2/(dist_squared + self.beta5**2)
        
        K[(dist_squared < self.delta0)] = self.k0
        K[(dist_squared >= self.delta_max)] = 0
        
        K = self.scaling_factor * K
        
        return(K if not is_scalar else K[0])
        
    @property
    def beta5(self):
        return self._beta5
    @property
    def k0(self):
        return self._k0
    @property
    def delta0(self):
        return self._delta0
    @property
    def delta_max(self):
        return self._delta_max
    @property
    def scaling_factor(self):
        return self._scaling_factor


class ConstantKernel(core.Kernel):
    """
    Constant kernel, return constant value for any distance (default 0.3)
    """
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self._C = kwargs.pop('C', 0.3)
    
    def __call__(self, x):
        
        x = np.asarray(x)
        is_scalar = False if x.ndim > 0 else True
        x.shape = (1,)*(1 - x.ndim) + x.shape
        
        K = np.ones(x.shape) * self.C
        K[(x >= 60.*60.)] = 0
        
        return(K if not is_scalar else K[0])
    
    @property
    def C(self):
        return self._C


class ZeroKernel(ConstantKernel):
    """
    Zero kernel, return zero for any distance; for testing purposes
    """
    def __init__(self):
        self._C = 0.0


class DeardonKernel(core.Kernel):
    """
    Power-law kernel (truncated) from Deardon et al. (2010)
    """
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self._k0 = kwargs.pop('k0', 1.85E-5)
        self._b = kwargs.pop('b', -1.66)
        self._delta0 = kwargs.pop('delta0', 719.)
        self._delta_max = kwargs.pop('delta_max', 30000.)
    
    def __call__(self, dist_squared):
        
        dist_squared = np.asarray(dist_squared)
        is_scalar = False if dist_squared.ndim > 0 else True
        dist_squared.shape = (1,)*(1-dist_squared.ndim) + dist_squared.shape
        
        K = dist_squared**self.b
        
        K[(dist_squared < self.delta0)] = self.k0
        K[(dist_squared > self.delta_max*self.delta_max)] = 0
        
        return(K if not is_scalar else K[0])
        
    @property
    def k0(self):
        return self._k0
    @property
    def b(self):
        return self._b
    @property
    def delta0(self):
        return self._delta0
    @property
    def delta_max(self):
        return self._delta_max


class DiggleKernelTruncated(core.Kernel):
    """
    Class representing the kernel described in Diggle (2006).  
    
    Includes a 'spark' term, rho.  
    """
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self._nu = kwargs.pop('nu', 1.0)
        self._kappa = kwargs.pop('kappa', 0.5)
        self._phi = kwargs.pop('phi', 0.41)
        self._rho = kwargs.pop('rho', 1.3E-4)
        
        self._k0 = kwargs.pop('k0', 0.3093)
        self._delta0 = kwargs.pop('delta0', 0.0138)
        self._delta_max = kwargs.pop('delta_max', 60*60.)
        
    def __call__(self, dist_squared):
        K = - np.power(np.sqrt(dist_squared)/self.phi, self.kappa)
        
        K = self.nu * np.exp(K) + self.rho
        
        K[(dist_squared < self.delta0)] = self.k0
        K[(dist_squared >= self.delta_max)] = 0
        
        return K
    
    @property
    def nu(self):
        """Baseline probability of infection at zero distance"""
        return self._nu
    @property
    def kappa(self):
        return self._kappa
    @property
    def phi(self):
        return self._phi
    @property
    def rho(self):
        return self._rho
    @property
    def k0(self):
        return self._k0
    @property
    def delta0(self):
        return self._delta0
    @property
    def delta_max(self):
        return self._delta_max


class KeelingKernel(core.Kernel):
    """
    Empirical kernel generated from data from the 2001 outbreak of FMD in the
    UK.  See Keeling et al. (2001) for details.
    """
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self._k0 = kwargs.pop('k0', 0.313254)
        self._scaling_factor = kwargs.pop('scaling_factor', 1.0)
    
    def __call__(self, dist_squared):
        """
        Interpolate from empirical Keeling kernel.  
        
        Columns in file saved in ./data/Kernel are 1) distance in meters, 
        and 2) kernel value.  
        
        Values are left truncated to 0.313254, and right truncated to 0
        
        Args:
            dist_squared: squared distance in km
        Returns:
            kernel function evaluated as the dist_squared distance
        """
        # Assumes xp values are increasing.  
        
        dist_squared = np.asarray(dist_squared)
        is_scalar = False if dist_squared.ndim > 0 else True
        dist_squared.shape = (1,)*(1-dist_squared.ndim) + dist_squared.shape
        
        D = np.sqrt(dist_squared)
        
        K = np.interp(D*1000, xp = kkern[0], fp = kkern[1], \
            left = self.k0, right = 0)
        
        K = self.scaling_factor * K
        
        # Set distances of zero to have a kernel value of be zero.  
        #masked_K = np.ma.masked_array(K, D == 0.0)
        #K = masked_K.filled(fill_value = 0.0)
        
        return(K if not is_scalar else K[0])
    
    @property
    def k0(self):
        return self._k0
    @property
    def scaling_factor(self):
        return self._scaling_factor
