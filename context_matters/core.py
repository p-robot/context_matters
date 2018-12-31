#!/usr/bin/env python3
"""
Core classes for objects in a Foot-and-mouth (FMD) outbreak.

Reinforcement learning problems require the definition of an environment and
agent.  This module defines the core classes for use within an FMD 
outbreak when looking at control measures that act uniformly on infected 
premises in the whole landscape of an FMD outbreak.  For an outbreak to be
simulated it is necessary to pass an agent and environment to a simulation 
object that is defined as a subclass from the simulation class in the rli 
module.  It is expected that users will define custom simulation objects 
(subclasses of rli.Simulation) that will include custom full2obs and 
collect_data functions.  This module uses a spatial kernel for simulating the 
spread of disease between farms.  There is no within-farm simulation of spread.

Explain how to get data in...
Actions are expected to take the form of ring vaccination or culling.  
The environment requires the definition of a landscape.  

This module is intended 
to be used with the rli module (see documentation within rli.py for 
descriptions of Sensation, Action, Agent, State and Simulation).  utils.py 
defines additional functions that are used in examples.  The infection process
code, used within the step method of the Outbreak object is written in 
Cython and stored within the module cy.  If farm-specific
actions are intended then the user will have to redefine the agent and 
environment in this module.  

Available classes:
-------------
- Simulation

- StateFarms: A state of farms in a Landscape; rli.Sensation
- State: A state of farms in a Landscape; rli.Sensation
- Cull: A ring culling action; rli.Action
- Vacc: A ring vaccination action; rli.Action
- StaticAgent: An agent that only performs one action; rli.Agent
- Outbreak: Class representing an foot-and-mouth outbreak; rli.Environment
- Kernel: A spatial kernel object
- Landscape: A spatial allocation of feedlots, used in an Outbreak object

"""

import numpy as np, copy, pandas as pd
import scipy.spatial as sp
from am4fmd import utils, rli, cy, iterate

class State(rli.Sensation):
    """
    Class representing the state of an outbreak.
    
    This class represents the state of all farms within an FMD outbreak, 
    defined on a landscape.  This is a subclass of the Sensation class within
    the rli module.  State objects are passed between an agent and an
    environment in a reinforcement learning simulation.  It is expected that 
    this state will be input into an agent's step method and that a State's
    object will be output from an outbreak's step function.  The location of 
    farms is stored within the Landscape object.  
    This is just one way to represent the state of an outbreak.  Other user 
    defined states may be used to represent an outbreak (agents and 
    environments will have to be redefined also).  
    
    Attributes
    ----------
        status: numpy.ndarray
            the number of days since infection for each farm.  
            Values of zero indicate a farm is susceptible, values of greater 
            than zero indicated a farm is infected, values of -1 mean a farm 
            is completely culled.  When infectivity of farms and notification 
            begins depends on the latency and detection periods defined within 
            the Outbreak class.  
            
        ci: int
            Number of days from vaccination to conferment of immunity.  This is
            required so as to build the array for storing the vaccination 
            status of each farm.  
        
        nfarms: int
            the number of farms present.  Calculated from the length of
            the passed 'status' array.  
            
        n_cattle: numpy.ndarray
            the number of head of cattle within each farm (np.array).  
            Either a numpy array or integer can be passed at instantiation.  If
            an integer is passed, then it will be assumed that this represents
            the number of cattle on each farm.  
            
        n_sheep: numpy.ndarray
            the number of head of sheep within each farm (np.array).  
            Either a numpy array or integer can be passed at instantiation.  If
            an integer is passed, then it will be assumed that this represents
            the number of sheep on each farm.
        
        being_culled: numpy.ndarray
            boolean array indicating if each farm currently 
            has culling activities occuring on it.  Defaults to False for all
            farms if not provided at instantiation.  
            
        being_vacc: numpy.ndarray
            boolean array indicating if each farm currently has vaccination
            activities occuring on it.  Defaults to False for all farms.  
        
        being_disposed: numpy.ndarray
            boolean array indicating if each farm currently has vaccination
            activities occuring on it.  Defaults to False for all farms.
        
        when_reported: numpy.ndarray
            integer array indicating the day number at which this farm was
            reported as infected, otherwise default is numpy.inf.  
        
        once_an_ip: numpy.ndarray
            boolean array indictating which farms were once IPs.  
        
        dist2IP: numpy.ndarray
            float array denoting distance to the IP which instigated control 
            measures on the farm in question (e.g. ring culling around this 
            farm caused the farm in question to become a cull candidate).  
        
        whichIP: numpy.ndarray
            int array denoting the ID (index) of the farm which caused the farm
            in question to have control activities take place on it.  
        
        terminal: boolean
            is this the terminal state?
    
    """
    def __init__(self, status, ci, n_cattle = None, n_sheep = None, x = None, y = None, **kwargs):
        
        self._x = x
        self._y = y
        
        self._status = np.asarray(status)
        n = len(status)
        
        self._nfarms = n
        self._n_cattle = np.asarray(n_cattle)
        self._n_sheep = np.asarray(n_sheep)
        self.n_cattle_original = kwargs.pop('n_cattle_original', np.asarray(n_cattle))
        self.n_carcass = np.zeros(n)
        self.n_disposed = np.zeros(n)
        self.terminal = False
        
        if "being_culled" in kwargs:
            self.being_culled = np.asarray(kwargs["being_culled"])
        else:
            self.being_culled = np.array([False for ii in range(n)])
        
        if "being_vacc" in kwargs:
            self.being_vacc = np.asarray(kwargs["being_vacc"])
        else:
            self.being_vacc = np.array([False for ii in range(n)])
        
        if "being_disposed" in kwargs:
            self.being_disposed = np.asarray(kwargs["being_disposed"])
        else:
            self.being_disposed = np.array([False for ii in range(n)])
        
        if 'when_reported' in kwargs:
            self.when_reported = kwargs['when_reported']
        else:
            self.when_reported = np.array([np.inf for ii in range(n)])
        
        if "once_an_ip" in kwargs:
            self.once_an_ip = np.asarray(kwargs["once_an_ip"])
        else:
            self.once_an_ip = np.array([False for ii in range(n)])
        
        if "dist2IP" in kwargs:
            self.dist2IP = np.asarray(kwargs["dist2IP"])
        else:
            self.dist2IP = np.array([np.inf for ii in range(n)])
        
        if "whichIP" in kwargs:
            self.whichIP = np.asarray(kwargs["whichIP"])
        else:
            self.whichIP = np.array([-1 for ii in range(n)])
        
        self.notified = None
        self._n_notified = None
        
        self.n_inf = np.sum(status > 9)
        
        self.hull = None
        self.area = None
        
        self.vacc_used = 0
        self.ci = ci
        self.n_vacc = np.zeros( (ci, n) )
        self.n_i = np.array([0 for ll in range(n)])
        self.n_i_pseudo = np.array([0 for ll in range(n)])
        
    @property
    def status(self):
        """
        Infectious status of each feedlot (tuple)
        """
        return self._status
    
    @status.setter
    def status(self, value):
        "Allow the status to be set, to avoid an AttributeError"
        self._status = value
    
    @property
    def nfarms(self):
        "Number of premises on landscape"
        return self._nfarms
    
    @property
    def n_sheep(self):
        "Number of sheep in each farm"
        return self._n_sheep
    
    @property
    def n_cattle(self):
        "Number of cattle in each farm"
        return self._n_cattle
    
    @property
    def x(self):
        "X position of farms"
        return self._x
    
    @property
    def y(self):
        "X position of farms"
        return self._y


class Cull(rli.Action):
    """
    Class representing a culling action
    
    This class defines a culling action to be performed around infected 
    premises.  It is expected that the user will define several culling actions
    of different radii and pass these to the agent to choose from.  These cull
    actions are applied to the outbreak environment throughout the simulation.
    
    Attributes
    ----------
        inner_r: float
            Inner radius of the culling ring around an infected premises
            
        outer_r: float
            Outer radius of the culling ring around an infected premises
    """
    def __init__(self, inner_r, outer_r, dccull = True):
        self._inner_r = inner_r
        self._outer_r = outer_r
        self._dccull = dccull
    
    def __repr__(self):
        return "Cull: "+str(self.inner_r)+" & "+str(self.outer_r)+ " km"
    
    def __str__(self):
        return "Cull: "+str(self.inner_r)+" & "+str(self.outer_r)+ " km"
    
    @property
    def inner_r(self):
        "Inner radius"
        return self._inner_r
        
    @property
    def outer_r(self):
        "Outer radius"
        return self._outer_r
        
    @property
    def dccull(self):
        "Should DC culling be implemented?"
        return self._dccull


class Vacc(rli.Action):
    """
    Class representing a vaccination action
    
    This class defines a vaccination action to be performed around infected 
    premises.  It is expected that the user will define several such actions
    of different radii and pass these to the agent to choose from.  These vacc
    actions are applied to the outbreak environment throughout the simulation.
    
    Attributes
    ----------
        inner_r : float
            Inner radius of the vaccination ring around infected premises
            
        outer_r : float
            Outer radius of the vaccination ring around infected premises
        
    """
    def __init__(self, inner_r, outer_r, dccull = True):
        self._inner_r = inner_r
        self._outer_r = outer_r
        self._dccull = dccull
    
    def __repr__(self):
        return "Vacc: "+str(self.inner_r) + \
            " & "+str(self.outer_r)+ " km"
    
    def __str__(self):
        return "Vacc: "+str(self.inner_r) + \
            " & "+str(self.outer_r)+ " km"
    
    @property
    def inner_r(self):
        "Inner radius"
        return self._inner_r
        
    @property
    def outer_r(self):
        "Outer radius"
        return self._outer_r
    
    @property
    def dccull(self):
        "Should DC culling be implemented?"
        return self._dccull


class MCExploringStarts(rli.Agent):
    """
    Class representing Monte Carlo exploring starts control algorithm 
    assuming exploring starts and that episodes always terminate for all
    policies.  
    
    Figure 5.4 of Sutton and Barto (2008) Reinforcement Learning
    
    Choose starting state.  
    Choose a starting action
    """
    
    def __init__(self):
        # Append the list of available actions
        pass
    
    def start_trial(self, state):
        """
        Choose 
        """
        pass
    
    def step(self, s, action, reward, next_s, t):
        pass


class SarsaAgentTabular(rli.Agent):
    """
    Class representing a SARSA Agent with value function stored in a dictionary.
    
    This class represents an agent that learns using the SARSA RL algorithm.  
    See Sutton and Barto (1998) for the section on temporal difference 
    learning and the defintion of the SARSA algorithm.  This implementation 
    does not include eligibility traces and uses a tabular function 
    approximation (the Q-table is stored in a dictionary that has keys of 
    states and a vector of values, one for each action, within those).  
    
    Attributes
    ----------
        epsilon: float
            probability of choosing a random action at each decision point
            
        alpha: float
            the step-size parameter (see Sutton and Barto (1998))
            
        gamma: float
            the discount rate for rewards
        
        Q: dict
            the value function, here a dictionary
        
        visits: dict
            a dictionary recording the number of visits to each state
            
        actions: list; rli.Action
            a list of actions that can be taken at any decision point, 
            these should be of class Cull.  Instantiating a SarsaAgentTabular
            adds an index attribute to each action (Action.ind) so that actions
            are consistently referred (in lists/arrays) using a correct index.
            
        control_switch_times: list
            a list of times at which the agent may change control action
        
        starting_action: core.Action
            a starting action as an index of the list 'actions'
    
    Methods:
        
        start_trial: 
            Ready agent for the start of a trial by returning a first action
            
        step: 
            Step the agent through a timestep of the simulation
    """
    def __init__(self, actions, epsilon, alpha, gamma, \
            starting_action = None, **kwargs):
        
        self._epsilon = epsilon
        self._alpha = alpha
        self._gamma = gamma
        self.Q = dict()
        self.visits = dict()
        self._starting_action = starting_action
        
        # List all valid actions
        self._actions = actions
        
        for i, a in enumerate(self._actions):
            a.ind = i
        
        self._control_switch_times = range(2000)
    
    def start_trial(self, state):
        """
        Ready the agent for the start of a trial.  
        
        This method uses an epsilon-greedy algorithm, the current value 
        function, and the current state to choose the first action in a 
        simulation.  If the state has not been seen before then an array of 
        large negative values are used to initialize the value of each action 
        in that state.  This is so that the algorithm is forced to sample all
        the actions, for this same state, through simulation time.  If a 
        starting action has been set then this is returned instead.  
        
        Args:
            state: the starting state of the environment
        
        Returns: 
            The action to take given being in state 'state'.  
        """
        
        if self.starting_action is not None:
            action = self.actions[self.starting_action]
        else:
            # If Q[state] has not been seen before, create an array for it.  
            print(state)
            print(state.obs)
            if not(state.obs in self.Q):
                
                self.Q[state.obs] = np.ones(len(self.actions))*-10.
                self.visits[state.obs] = 0
            
            action = utils.egreedy(self.Q[state.obs], \
                                    self.actions, self.epsilon)
        
        return action
        
    def step(self, s, action, reward, next_s, t):
        """
        Run the agent through one timestep of the simulation.  
        
        This is where learning occurs.  See the step method of the Agent class
        in the rli.py module for a more general description of this method.  
        At each time step of the simulation the agent checks if the terminal 
        state has been reched.  If so, a dummy action is returned.  Otherwise
        the agent checks that this time point represents a decision point, 
        which are listed within Agent.control_switch_times.  If this is a 
        decision time point then the agent checks to see if the state and 
        next state have been observed before, if they haven't then the Q value
        function and visits dictionaries are updated with empty arrays.  Finally
        the next action is chosen using an epsilon greedy algorithm and the 
        value function is updated using the sarsa algorithm.  The visits 
        dictionary records how many times each state has been observed.  The
        next action to take is returned (as an action object, not an index).  
        
        Args:
            s: the current state
            action: the action taken in state s
            reward: the reward from taking the action in state s
            next_s: the next state
            t: current time step, to check if this is a decision horizon or not
        Returns:
            The next action to be taken.  This is an action object, 
            not an index.
            
        """
        
        # Check for terminal state
        if (next_s.terminal):
            # Return a dummy action
            next_a = 20
        else:
            if t in self.control_switch_times:
                
                # If Q[s] has not been seen before, create a table for it.
                if not s.obs in self.Q:
                    self.Q[s.obs] = np.empty(len(self.actions))
                    self.Q[s.obs][:] = 0#np.NAN
                    self.visits[s.obs] = 0
                
                if not next_s.obs in self.Q:
                    self.Q[next_s.obs] = np.empty(len(self.actions))
                    self.Q[next_s.obs][:] = 0#np.NAN
                    self.visits[next_s.obs] = 0
                
                # Find the next action
                next_a = utils.egreedy(self.Q[next_s.obs], \
                                        self.actions, \
                                        self.epsilon)
                
                # Update the Q function
                target = reward + self.gamma * self.Q[next_s.obs][next_a.ind]
                
                self.Q[s.obs][action.ind] = \
                    self.Q[s.obs][action.ind] \
                    + self.alpha*(target - self.Q[s.obs][action.ind])
                
                # Record the observed state
                self.visits[s.obs] += 1
                
            else:
                next_a = action
                
        return next_a
    
    @property
    def control_switch_times(self):
        "control switch times"
        return self._control_switch_times
    
    @control_switch_times.setter
    def control_switch_times(self, value):
        "Allow control_switch_times to be set"
        self._control_switch_times = value
    
    @property
    def starting_action(self):
        "Action with which to start each trial"
        return self._starting_action
    
    @property
    def epsilon(self):
        "Epsilon parameter"
        return self._epsilon
    
    @property
    def gamma(self):
        "Gamma parameter"
        return self._gamma
    
    @property
    def alpha(self):
        "Alpha parameter"
        return self._alpha
    
    @property
    def actions(self):
        "Actions"
        return self._actions


class StaticAgent(SarsaAgentTabular):
    """
    Class representing an agent that does not change action
    
    Several studies have used in-silico methods to test the efficacy of control
    methods by applying one control method from start to finish thoughout a 
    simulated outbreak.  The StaticAgent represents such control.  That is, 
    it applies the same action throughout a simulation.  See the Agent class
    of the rli module for more information on the particular methods of the 
    Agent class.  When a StaticAgent is defined it is passed the single action
    that it always implements.  Usage is along the lines of 
    StaticAgent([Cull(0,5)]).
    
    """
    def __init__(self, action, gamma = 1, **kwargs):
        
        # Call the initialisation method of the parent class
        super(self.__class__, self).__init__(action, epsilon = 0.0, \
            alpha = 0.0, gamma = gamma, **kwargs)
    
    def start_trial(self, state):
        return self.actions[0]
    
    def step(self, s, action, reward, next_s, t, full_s):
        if (next_s == self._terminal_state):
            # Return a dummy action
            next_a = 20
        else:
            if t in self.control_switch_times:
                next_a = action
            else: 
                next_a = Cull(0.0, 0.0)
        
        return next_a


class Kernel(object):
    """Class representing a kernel object.  
    
    See am4fmd.kernels for subclasses of the Kernel class.  
    
    User needs to define the __call__ method.  
    
    """
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, dist_squared):
        return dist_squared


class Landscape(object):
    """
    Class representing a spatial allocation of feedlots/farms
    
    INPUT
    
        c : np.array, 2D
            Coordinates of premises, np.array([[x1, y1], ..., [xn, yn]])
        
        D : np.array, 2D
            Pairwise distance matrix between premises
            Calculated automatically if not given.
        
        Dsq : np.array, 2D
            Pairwise squared distance matrix between premises.
            Calculated automatically if not given.
    """
    
    def __init__(self, c, D = None, Dsq = None, **kwargs):
        self._nfarms = len(c)
        self._coords = np.asarray(c)
        
        if D is None:
            self._D = sp.distance.cdist(c, c, metric = 'euclidean')
            self._Dsq = sp.distance.cdist(c, c, metric = 'sqeuclidean')
        else:
            self._D = D
            self._Dsq = Dsq
    
    @property
    def coords(self):
        "Coordinates of feedlots [(x1, y1), ... , (xn, yn)]"
        return self._coords
    
    @property
    def D(self):
        "Euclidean inter-feedlot distance matrix"
        return self._D
    
    @property
    def Dsq(self):
        "Squared euclidean inter-feedlot distance matrix"
        return self._Dsq
    
    @property
    def nfarms(self):
        "Number of premises on landscape"
        return self._nfarms



class Outbreak(rli.Environment):
    """
    Class representing a foot-and-mouth disease outbreak.  
    
    All delays are listed in days.  All rates are listed as per-farm per day 
    rates.  
    
    Attributes
    ----------
        period_latent: float
            the delay (days) from exposed to infectious
            
        period_nodetect: float
            the delay (days) from exposure to notification
            
        period_culling: float
            the delay (days) from notification to the start of culling
            
        period_confer_immune: float
            the delay (days) from vaccine admin to conferal of immunity
            
        transmiss_sheep: float
            the per-head transmissibility of sheep
            
        transmiss_cows: float
            the per-head transmissibility of cattle
            
        suscept_sheep: float
            the per-head susceptibility of sheep
            
        suscept_cows: float
            the per-head susceptibility of cattle
        
        rate_cull: float
            the per-farm per-day cull rate
        
        rate_vacc: float
            the per-farm per-day vaccination rate
            
        L: Landscape
            a Landscape object of farms on which the outbreak occurs
            
        grid_shape: tuple
            size of the discretized grid used for infection calcs
            
        grid: numpy.array
            grid square index within which each farm is situated
        
        grid_x: numpy.array
            grid square column within which each farm is situated
            
        grid_y: numpy.array
            grid square row within which each farm is situated
            
        kernel: Kernel
            kernel object for use in this outbreak
            
        K: numpy.array
            array squared distances btw farms evaluated using the kernel
            
        Suscept: numpy.array
            array of susceptibility of each grid square
        
        Transmiss: numpy.array
            array of transmissibility of each grid square
            
        MaxRate: numpy.array
            array of the max rate of infection from each grid square to each
            other
            
        Num: numpy.array
            the number of farms in each grid square
            
        first_: numpy.array
            the first farm to consider in each grid square
            
        last_: numpy.array
            the last farm to consider in each grid square
        
        max_sus: numpy.array
            the max susceptibility in each grid square
        
        Dist2all: numpy.array
            
        KDist2all: numpy.array
            
        start: StateFarms
            the starting state of the outbreak
        
    Methods
    
    """
    def __init__(self, Landscape, start, grid_shape, objective, kernel = None, **kwargs):
        
        # Set default values for optional arguments
        self._period_latent = kwargs.pop('period_latent', 4)
        self._period_nodetect = kwargs.pop('period_nodetect', 9)
        self._period_culling = kwargs.pop('period_culled', 4)
        self._period_confer_immune = kwargs.pop('period_confer_immune', 7)
        self._period_ip_culling = kwargs.pop('period_ip_culling', 0)
        self._transmiss_sheep = kwargs.pop('transmiss_sheep', 5.10E-07)
        self._transmiss_cows = kwargs.pop('transmiss_cows', 7.70E-07)
        self._suscept_sheep = kwargs.pop('suscept_sheep', 1.0)
        self._suscept_cows = kwargs.pop('suscept_cows', 10.5)
        self._rate_cull = kwargs.pop('rate_cull', 50)
        self._rate_vacc = kwargs.pop('rate_vacc', 50)
        self._rate_dispose = kwargs.pop('rate_dispose', 50)
        
        # Non-optional arguments
        self._objective = objective
        self.L = Landscape
        self._start = start
        
        # Define grid over the landscape
        if not grid_shape is None:
            self._grid_shape = grid_shape
            
            x = np.array(self.L.coords).transpose()[0]
            y = np.array(self.L.coords).transpose()[1]
            
            d_perim = np.ceil(np.max([np.max(x) - np.min(x), \
                np.max(y) - np.min(y)]))
            
            self._grid_x, self._grid_y, self._grid = \
                utils.Coords2Grid(x, y, d_perim, d_perim, \
                grid_shape[0], grid_shape[1])
        else:
            self._grid = None
            self._grid_x = None
            self._grid_y = None
        
        # Define the kernel, and evaluate squared distances
        # between all farms using this kernel
        self._kernel = kernel
        if not kernel is None:
            self._K = kernel(self.L.Dsq)
        else:
            self._K = None
        
        # Calculate probabilities of transmission between grids
        self.Suscept, self.Transmiss, self.MaxRate, self.Num, self.first_, \
            self.last_, self.max_sus, self.Dist2all, \
            self.KDist2all = self.SetState(start)
    
    @property
    def grid_shape(self):
        "Shape of grid discretization"
        return self._grid_shape
    @grid_shape.setter
    def grid_shape(self, grid_shape):
        """
        Override setting of grid_shape so grid calculations 
        are made when grid_shape is set.
        """
        self._grid_shape = grid_shape
        
        x = np.array(self.L.coords).transpose()[0]
        y = np.array(self.L.coords).transpose()[1]
        
        d_perim = np.ceil(np.max([np.max(x) - np.min(x), \
            np.max(y) - np.min(y)]))
        
        self._grid_x, self._grid_y, self._grid = \
            utils.Coords2Grid(x, y, d_perim, d_perim, \
            grid_shape[0], grid_shape[1])
    
    @property
    def start(self):
        "Starting state of the system"
        return self._start
    
    @property
    def K(self):
        "Distance matrix evaluate using the kernel"
        return self._K
    
    @property
    def goal(self):
        "Goal state of the system"
        return self._goal
    
    @property
    def kernel(self):
        "Kernel function"
        return self._kernel
    @kernel.setter
    def kernel(self, value):
        "Calculate kernel-evaluated distance matrix when kernel is set."
        self._kernel = value
        self._K = value(self.L._Dsq)
    @property
    def grid(self):
        "Grid ID of farm in landscape"
        return self._grid
    
    @property
    def grid_x(self):
        "X grid ID of farm in landscape"
        return self._grid_x
    
    @property
    def grid_y(self):
        "Y grid ID of farm in landscape"
        return self._grid_y
    
    @property
    def period_latent(self):
        "Latency period"
        return self._period_latent
    
    @property
    def period_nodetect(self):
        "Zero detection period"
        return self._period_nodetect
    
    @property
    def period_culling(self):
        "Period until culling occurs"
        return self._period_culling
    
    @property
    def period_confer_immune(self):
        "Period from vaccinating until immunity is conferred"
        return self._period_confer_immune
    
    @property
    def period_silent_spread(self):
        "Silent spread period"
        return self._period_silent_spread
    
    @property
    def rate_cull(self):
        "Per-feedlot daily culling rate"
        return self._rate_cull
    
    @property
    def rate_vacc(self):
        "Per-feedlot daily vaccination rate"
        return self._rate_vacc
    
    @property
    def rate_dispose(self):
        "Per-feedlot daily disposal rate"
        return self._rate_dispose
    
    @property
    def transmiss_sheep(self):
        "Per-sheep transmission rate of FMD"
        return self._transmiss_sheep
    
    @property
    def transmiss_cows(self):
        "Per-cow transmission rate of FMD"
        return self._transmiss_cows
    
    @property
    def suscept_sheep(self):
        "Per-sheep susceptibility rate of FMD"
        return self._suscept_sheep
    
    @property
    def suscept_cows(self):
        "Per-cow susceptibility rate of FMD"
        return self._suscept_cows
    
    @property
    def objective(self):
        "Objective which defines the reward function"
        return self._objective
    
    #############
    ## Methods ##
    #############
    def start_trial(self):
        """
        Ready the environment for the start of a simulation
        
        This method returns the starting state of the environment,
        ready for the start of the simulation.  
        
        Returns:
            The starting state of the environment
        """
        
        state = copy.deepcopy(self.start)
        
        return state
    
    def SetState(self, state):
        """
        Perform tasks relating to setting the state.  
        
        - calculate the suceptibility
        - calculate the transmissibility
        - calculate probabilities of grids infecting each other
        
        """
        
        s_sheep = np.multiply(self.suscept_sheep, state.n_sheep)
        s_cows = np.multiply(self.suscept_cows, state.n_cattle)
        Suscept = np.add(s_sheep, s_cows)
        
        t_sheep = np.multiply(self.transmiss_sheep, state.n_sheep)
        t_cows = np.multiply(self.transmiss_cows, state.n_cattle)
        Transmiss = np.add(t_sheep, t_cows)
        
        x = np.array(self.L.coords).transpose()[0]
        y = np.array(self.L.coords).transpose()[1]
        d_perimeter = np.ceil(np.max([np.max(x) - np.min(x), \
            np.max(y) - np.min(y)]))
        
        MaxRate, Num, first_, last_, max_sus, Dist2all, KDist2all = \
        utils.calc_grid_probs(self.grid, self.grid_shape[0], \
            self.grid_shape[1], Suscept, self.kernel, d_perimeter, d_perimeter)
        
        return(Suscept, Transmiss, MaxRate, Num, first_, last_, max_sus, \
            Dist2all, KDist2all)
        
    def step(self, current_s, current_a, t):
        """
        Main iterate function of the FMD outbreak.  
        
        INPUT
        -------
        current state, current action
        
        OUTPUT
        -------
        reward, next state
        
        """
        
        # Make a copy of the current state object
        next_s = copy.deepcopy(current_s)
        
        #####################################
        # INFECTIOUS -> DETECTED TRANSITION #
        #####################################
        detection_time = self.period_nodetect + self.period_culling
        just_detected = (next_s.status == detection_time)
        
        if np.any(just_detected):
            
            # Implement IP culling
            next_s.being_culled[just_detected] = True
            
            # Transfer any animals that are thought to be immune to be infected
            # (to n_cattle) also, any animals thought to be being vaccinated.
            next_s.n_cattle[just_detected] += np.sum(next_s.n_vacc[:, just_detected], \
                axis = 0).astype(np.int64)
            next_s.n_cattle[just_detected] += next_s.n_i_pseudo[just_detected]
            next_s.n_cattle[just_detected] += next_s.n_i[just_detected]
            
            # Reset numbers being vaccinated, immune, and thought to be immune
            # to zero in all detected farms.  
            next_s.being_vacc[just_detected] = False
            next_s.n_i[just_detected] = 0
            next_s.n_i_pseudo[just_detected] = 0
            next_s.n_vacc[:, just_detected] = 0
        
        ######################################
        # SUSCEPTIBLE -> INFECTED TRANSITION #
        ######################################
        # Can use this if Cython doesn't work
        #status, event = utils.infection_process(next_s.status) 
        
        status, event = cy.evolve_infection(next_s.status, \
            self.MaxRate, self.grid, self.Transmiss, self.Suscept, \
            self.first_, self.last_, \
            self.period_latent, self.Num, self.K)
        
        ######################################
        # UPDATE VACCINE DURATION IN CATTLE  #
        ######################################
        # Those that have been vaccinated 1 day ago, are now 
        # vaccinated 2 days ago and so on
        if next_s.n_vacc.any():
            N_V = next_s.n_vacc
            N_I = next_s.n_i
            N_I_pseudo = next_s.n_i_pseudo
            
            N_V = np.roll(N_V, shift = 1, axis = 0)
            
            # Update the number of immune cattle in each feedlot
            suscept = (status == 0)
            N_I += (N_V[0] * suscept).astype(np.int64)
            
            # Update num cattle in feedlot that look immune but infected
            N_I_pseudo += (N_V[0]*(status > 0)).astype(np.int64)
            
            # Remove 1st col as these animals are yet to be vaccinated
            N_V[0] = 0
            
            # IF Number immune is positive 
            # AND No animals still being vaccinated 
            # AND no other animals in feedlot 
            # AND feedlot was original susceptible
            # THEN whole feedlot is immune.  
            immune = (N_I > 0) & (N_V.sum(axis=0) == 0) & \
                (next_s.n_cattle == 0) & (status == 0)
            
            if any(immune):
                # Removed due to vaccine granted immunity.  
                status[immune] = -1
                next_s.being_vacc[immune] = False
                next_s.being_culled[immune] = False
            
            # Update the state
            next_s.n_vacc = N_V
            next_s.n_i = N_I
            next_s.n_i_pseudo = N_I_pseudo
        
        ###################################################
        # VACCINE ARRIVAL - update vacc bank if necessary #
        ###################################################
        
        
        if np.any(just_detected):
            # Find those farms which we think are susceptible 
            # and upon which culling/vacc isn't occurring
            # (those which are subject to control actions)
            
            s_cond1 = (status < detection_time)
            s_cond2 = (status >= 0)
            s_cond3 = [not farm for farm in next_s.being_culled]
            s_cond4 = [not farm for farm in next_s.being_vacc]
            susceptible_mask = s_cond1 & s_cond2 & s_cond3 & s_cond4
            
            #susceptible_mask = (status < (self.period_nodetect + \
            #    self.period_culling)) & ( status >= 0 ) & \
            #    ([not farm for farm in next_s.being_culled])
            
            # Subset the distance matrix to find dists from all IPs to
            # all farms upon which we can implement control
            D_infectious = self.L.D[just_detected]
            D_inf_to_sus = D_infectious[:, susceptible_mask]
            
            # Find those farms in the ring radius of the IP
            # set their control status to True.  
            if D_inf_to_sus.any():
                nbhd = utils.Neighbourhood(D_inf_to_sus, \
                    current_a.outer_r, current_a.inner_r)
                
                # Control candidates
                cc = np.where(susceptible_mask)[0][nbhd]
                
                ########################
                # FIND VACC CANDIDATES #
                ########################
                if isinstance(current_a, Vacc):
                    next_s.being_vacc[cc] = True
                
                ########################
                # FIND CULL CANDIDATES # beyond those that are IP culled
                ########################
                if isinstance(current_a, Cull):
                    
                    # Boolean list for whether each farms is a cull candidate
                    next_s.being_culled[cc] = True
                    
                    # Turn off any vaccination may be occuring
                    next_s.being_vacc[cc] = False
                    
        #####################################
        # BEING CULLED -> COMPLETELY CULLED #
        #####################################
        # List the conditions for a farm being 'culled':
        # zero cattle, zero being vaccinated, zero thought to be immune
        c_cond1 = (next_s.n_cattle == 0)
        c_cond2 = (next_s.n_i_pseudo == 0)
        c_cond3 = (next_s.n_vacc.sum(axis = 0) == 0)
        
        # Find which farms meet these conditions
        culled = c_cond1 & c_cond2 & c_cond3
        
        # Change culled farms status, stop them being listed as 'culled'
        status[culled] = -1
        next_s.being_culled[culled] = False
        
        ###############################################
        # BEING DISPOSED OF -> COMPLETELY DISPOSED OF #
        ###############################################
        
        disposed = culled & (next_s.n_carcass == 0)
        next_s.being_disposed[disposed] = False
        
        ######################
        # IMPLEMENT DISPOSAL #
        ######################
        
        # If there are any carcasses, turn on disposal
        has_carcasses = (next_s.n_carcass > 0)
        next_s.being_disposed[has_carcasses] = True
        
        # Remove from carcass (smallest of [number of carcasses, rate_dispose])
        remaining_carcass = next_s.n_carcass[next_s.being_disposed]
        n_carcass_to_dispose = np.minimum(remaining_carcass, self.rate_dispose)
        
        # Add these to disposed pile
        next_s.n_carcass[next_s.being_disposed] -= n_carcass_to_dispose
        next_s.n_disposed[next_s.being_disposed] += n_carcass_to_dispose
        
        #####################
        # IMPLEMENT CULLING #
        #####################
        # Remove culled animals (accounting).  
        n_cattle_to_remove = np.minimum(next_s.n_cattle[next_s.being_culled], \
            self.rate_cull)
        next_s.n_cattle[next_s.being_culled] -= n_cattle_to_remove
        next_s.n_carcass[next_s.being_culled] += n_cattle_to_remove
        
        #####################
        # IMPLEMENT VACC    #
        #####################
        n_cattle_to_vacc = np.minimum(next_s.n_cattle[next_s.being_vacc], \
            self.rate_vacc)
        next_s.n_cattle[next_s.being_vacc] -= n_cattle_to_vacc
        next_s.n_vacc[0][next_s.being_vacc] += n_cattle_to_vacc
        next_s.vacc_used += np.sum(n_cattle_to_vacc)
        
        if (self.objective == 'duration'):
            reward = -1.0
        elif (self.objective == 'culling'):
            reward = 0
        else:
            print("Unknown objective")
        
        next_s.status = status
        
        # Check if state is the terminal state
        if self.goal(next_s):
            if (self.objective == 'culling'):
                reward = - np.sum(next_s.n_disposed)
            else:
                reward = 0
            next_s.terminal = True # = self.__init__state
        
        return reward, next_s
    
    def goal(self, s):
        """
        Indicate if there are any infected farms left (the goal state).  
        
        This is a function of the state.  
        When there are no E, I, just reported, or carcasses around, finish the
        trial.
        
        Parameters
        ----------
            s : rli.State
                current state of the environment
            
        Returns
        -------
            Boolean if the goal state has been reached.  The goal state being 
            where all farms have an infection status of zero or less.  
        """
        
        GOAL = False
        
        c1 = (np.all(s.status <= 0))
        c2 = (np.sum(s.n_i_pseudo) == 0)
        c3 = (np.sum(s.n_vacc) == 0)
        c4 = (np.sum(s.n_carcass) == 0)
        
        if c1 & c2 & c3 & c4:
            GOAL = True
        
        return GOAL

