#!/usr/bin/env python3
"""
Run an FMD outbreak with ring culling and vaccination actions.  

W. Probert
"""

import scipy.spatial as sp
import numpy as np, copy, pandas as pd, time, csv
from os.path import join
from scipy.spatial import distance as dist

from . import core, rli, utilities, cy

class FMDSim(rli.Simulation):
    """
    Simulation class for managing interactions between agent and environment
    
    Most methods for this class are defined in the parent class in the
    rli module.  The user defined methods are full2obs which transforms
    the state from a 'full' state that encompasses all the information that 
    is needed for the simulation to evolve, to an 'observed' state, which is the
    state that the agent sees to inform decision making (this may be thought of
    as the 'operational' states and the type of states that make up the 
    dimensions of the policy and value function).  
    
    """
    def __init__(self, Agent, Environment, max_time, full = False, \
            time_save_state = None, trials_save_state = None, \
            save_dir = None, save_name = "state", **kwargs):
        
        # Call the initialisation method of the parent class
        super(self.__class__, self).__init__(Agent, Environment, max_time, **kwargs)
        
        self.agt = Agent
        self.env = Environment
        self.max_time = max_time
        
        full_hull = sp.ConvexHull(self.env.L.coords)
        x = full_hull.points[full_hull.vertices].transpose()[0]
        y = full_hull.points[full_hull.vertices].transpose()[1]
        full_area = cy.hull_area(x,y)
        
        # Resolution at which to discretize the area state variable
        area_res = kwargs.pop('area_resolution', 201)
        self._area_res = area_res
        self.abins = np.linspace(0., full_area, num = area_res)
        
        self.outlist = []
        self.durations = []
        self.culls = []
        self.farms = []
        
        self.current_s = None
        self.current_a = None
        
        if time_save_state is None: 
            self.time_save_state = []
        else:
            self.time_save_state = time_save_state
        
        if trials_save_state is None: 
            self.trials_save_state = []
        else:
            self.trials_save_state = trials_save_state
        
        self.save_dir = save_dir
        self.save_name = save_name
        
        self.full = full
        
        # Set the terminal state
        self._terminal_state = "terminal"
        self.env._terminal_state = self._terminal_state
        self.agt._terminal_state = self._terminal_state
        
        self.start_trial()
    
    def start_trial(self):
        """
        Start a reinforcement learning trial (ie episode).  
        
        The function calls the start_trial method of both the Environment and 
        Agent in the simulation which respectively return the first Sensation
        (ie state) and Action to be taken in the simulation.  This function 
        also resets the timestep counter to zero.  Custom methods may compute 
        particular statistics per trial and/or update any relevant graphics.
        """
        
        # Set the current time step
        self.timestep = 0
        
        # Find the first state
        self.current_s = self.env.start_trial()
        
        self.current_s.area = None
        
        # Find the first action given the current state
        observed_s = self.full2obs(self.current_s)
        self.current_a = self.agt.start_trial(observed_s)
        
        self.df = pd.DataFrame(columns = ('n_exp', 'n_inf', 'n_not', \
            'n_cull', 'n_bcull', 'n_vacc', 'vacc_used', 'area', 'action'))
    
    
    def trials(self, num_trials, max_steps_per_trial = 10000):
        """
        Run the simulation for num_trials trials (aka episodes)
        
        This function runs the simulation for num_trials trials, starting from 
        whatever state the environment is in. Trial are limited to be less than
        max_steps_per_trial steps. Each trial is initialized with start_trial 
        and is completed when the terminal state is reached (or when 
        max_steps_per_trial is reached).  
        
        Args:
            num_trials: number of trials for which to run the simulation
            max_steps_per_trial: maximum number of steps witin any trial
        """
        if (self.current_s == self.terminal_state):
            self.start_trial()
        
        # Find the current time
        start_time = time.time()
        
        # Step through using the environment's function
        for self.trial in range(num_trials):
            if ((time.time() - start_time) >= self.max_time*60*60):
                break
            
            self.timestep = 0
            if self.verbose:
                print("Trial num:", self.trial)
            
            while( (self.timestep < max_steps_per_trial) & (self.current_s != self.terminal_state)):
                
                if (self.timestep in self.time_save_state) & (self.trial in self.trials_save_state):
                    # If this is a time at which we should save the state, 
                    # then save full state of the outbreak in desired folder
                    filename_state = join(self.save_dir, \
                        "{}_trial{:03d}_t{:03d}.csv".format( \
                            self.save_name, self.trial, self.timestep))
                    
                    with open(filename_state, 'w') as csvfile:
                        statewriter = csv.writer(csvfile, delimiter = ',')
                        
                        header = ['id', 'x', 'y', 'n_cattle', 'n_sheep', \
                            'n_carcass', 'n_disposed', 'status', \
                            'being_culled', 'being_vacc', 'whichIP', 'once_an_ip']
                        statewriter.writerow(header)
                        
                        # These are the farms that we're going to output to file
                        non_suscept = (self.current_s.status != 0)
                        
                        # Save the ID of the farm
                        N = len(self.current_s.x)
                        ids = (np.arange(N)+1)[non_suscept]
                        
                        statewriter.writerows(zip(ids, \
                            self.current_s.x[non_suscept], \
                            self.current_s.y[non_suscept], \
                            self.current_s.n_cattle[non_suscept], \
                            self.current_s.n_sheep[non_suscept], \
                            self.current_s.n_carcass[non_suscept], \
                            self.current_s.n_disposed[non_suscept], \
                            self.current_s.status[non_suscept], \
                            self.current_s.being_culled[non_suscept], \
                            self.current_s.being_vacc[non_suscept], \
                            self.current_s.whichIP[non_suscept], \
                            self.current_s.once_an_ip[non_suscept]))
                
                # Step the environment to find a new state
                # this could [instead] have outputs of: r, next_s, terminal (boolean)
                r, next_s = self.env.step(self.current_s, self.current_a, t = self.timestep)
                
                # Save the data
                self.collect_data(self.current_s, self.current_a, r, next_s)
                
                # Find the next action (from the observed state)
                observed_s = self.full2obs(self.current_s)
                observed_next_s = self.full2obs(next_s)
                
                next_a = self.agt.step(observed_s, self.current_a, \
                        r, observed_next_s, self.timestep, self.current_s)
                
                # Reset state and action
                self.current_a = next_a
                self.current_s = next_s
                self.timestep += 1
            
            if not (self.timestep < max_steps_per_trial):
                if self.verbose:
                    print("Max steps per trial reached")
            
            if self.verbose:
                print("Time steps:", self.timestep)
            
            # Restart the next trial (unless it's the last trial)
            if (self.trial != (num_trials-1)):
                self.start_trial()
    
    
    def full2obs(self, current_s):
        """Translate the full state of the environment to the observed state.
        
        In several reinforcement learning tasks the agent will not see the 
        full state of the environment but a reduced form of the state, ie an 
        observed state.  This function translates the full state of the 
        environment (that is output by the environment step function) to the 
        state that is observed by the agent (and input to the agent step 
        function).  By default, the full state and observed state are the same
        so the user need not change this function.  
        
        Args:
            s: current full state of the environment
            
        Returns
            The state observed by the agent (a function of the full state of 
            the environment).  
        """
        
        if (current_s == self._terminal_state):
            s = self._terminal_state
        else:
            if not current_s.area:
                s = (current_s.n_inf, 0)
            else:
                ind = np.digitize([current_s.area], self.abins, right = True)[0]
                s = (current_s.n_inf, self.abins[ind])
        return s
    
    def collect_data(self, current_s, current_a, reward, next_s):
        """ Save data throughout the simulation.  
        
        This function is called once on each step of the simulation to store
        data.  
        
        Arguments
        ---------
            current_s: rli.Sensation
                current state
                
            current_a: rli.Action
                current action
            
            reward: float
                current reward
            
            next_s: rli.Sensation
                next state
        """
        
        if not(current_s == self._terminal_state):
            
            # define conditions for farm inclusion in outbreak area calculation
            cond1 = (current_s.status > self.env.period_nodetect) 
            cond2 = (current_s.status < 0)
            cond3 = current_s.being_culled
            cond4 = current_s.being_vacc
            cond5 = current_s.being_disposed
            all_conds = (cond1 | cond2 | cond3 | cond4 | cond5)
            
            current_s.notified_ind = np.where(all_conds)[0]
            current_s.n_notified = len(current_s.notified_ind)
            notified_vertices = self.env.L.coords[current_s.notified_ind]
            
            # if there are more than 3 points, calculate an area
            if current_s.n_notified > 3:
                # try calculating hull (3 points may still be coplanar)
                try: 
                    current_s.hull = sp.ConvexHull(notified_vertices)
                except:
                    print("Points are coplanar")
                else:
                    pass
                
                #if not current_s.hull:
                #    current_s.hull = sp.ConvexHull(self.env.L.coords[current_s.notified_ind])
                #else:
                #    print("add points to the hull")
                #    current_s.hull = sp.ConvexHull(self.env.L.coords[current_s.notified_ind])
                
                v = current_s.hull.vertices
                x = current_s.hull.points[v].transpose()[0]
                y = current_s.hull.points[v].transpose()[1]
                current_s.area = cy.hull_area(x,y)
        
        # calculate summarise to save to dataframe
        n_exp = np.sum(current_s.status > 0)
        n_inf = np.sum(current_s.status > self.env.period_latent)
        current_s.n_inf = n_inf
        n_not = np.sum(current_s.status > self.env.period_nodetect)
        n_cull = np.sum(current_s.status > self.env.period_culling + \
            self.env.period_nodetect)
        n_bcull = np.sum(current_s.being_culled)
        n_vacc = np.sum(current_s.being_vacc)
        
        if current_s.area:
            area = current_s.area
        else: 
            area = np.nan
        
        a_ind = np.where(self.agt.actions == current_a)[0]
        
        # save data to dataframe
        data = [n_exp, n_inf, n_not, n_cull, n_bcull, n_vacc, \
            current_s.vacc_used, area, a_ind]
        self.df.loc[self.timestep] = data
        
        # save list of all action, states, rewards if desired
        if self.full:
            if self.timestep == 0:
                self.outlist.append([[current_s, \
                            current_a, \
                            reward, \
                            next_s]])
            else:
                self.outlist[self.trial].append([current_s, current_a, \
                    reward, next_s])
            
            if (next_s == self._terminal_state):
                self.durations.append(self.timestep)
                
                culled = (self.current_s.status == -1)
                culls = np.sum(self.current_s.n_cattle_original[culled])
                self.culls.append(culls)
                self.farms.append(np.sum(culled))
        else:
            if (next_s == self._terminal_state):
                self.durations.append(self.timestep)
                
                culled = (self.current_s.status == -1)
                culls = np.sum(self.current_s.n_cattle_original[culled])
                self.culls.append(culls)
                self.farms.append(np.sum(culled))


class Outbreak_cc_vc(core.Outbreak):
    """
    Class representing a foot-and-mouth disease outbreak 
    with a global carcass constraint and a constraint on vaccine capacity.  
    
    The class only overloads the 'step' method of the core.Outbreak class.  
    
    """
    def __init__(self, Landscape, start, grid_shape, objective, cfilter_fn, \
        vfilter_fn, period_silent_spread = 7, kernel = None, **kwargs):
        
        # Call the initialisation method of the parent class
        super(self.__class__, self).__init__(Landscape, start, grid_shape, \
            objective, kernel = kernel, **kwargs)
        
        # Global carcass constraint
        self._carcass_constraint = kwargs.pop('carcass_constraint', 100000)
        
        self._vacc_constraint = kwargs.pop('vacc_constraint', 35000)
        
        # Define the function for filtering/ordering cull carcasses
        self._filter_culls = cfilter_fn
        
        # Define the function for filtering/ordering farms for vaccination
        self._filter_vacc = vfilter_fn
        
        self._period_silent_spread = period_silent_spread
        self._silent_spread_times = range(self.period_silent_spread)
        
    @property
    def filter_culls(self):
        "Function for filtering culls - i.e. who to cull"
        return self._filter_culls
    
    @property
    def filter_vacc(self):
        "Function for filtering farms to vaccinate - i.e. who to cull"
        return self._filter_vacc
    
    @property
    def carcass_constraint(self):
        "Global carcass constraint"
        return self._carcass_constraint
    
    @property
    def vacc_constraint(self):
        "Global vaccine capacity"
        return self._vacc_constraint
    
    @property
    def period_silent_spread(self):
        "silent spread period"
        return self._period_silent_spread
    
    @property
    def silent_spread_times(self):
        "silent spread times"
        return self._silent_spread_times
    
    @silent_spread_times.setter
    def silent_spread_times(self, value):
        "Allow control_switch_times to be set"
        self._silent_spread_times = value
    
    
    #############
    ## Methods ##
    #############
    
    def step(self, current_s, current_a, t):
        """
        See rli.Environment for details on this step method
        """
        
        # Make a copy of the current state object
        next_s = copy.deepcopy(current_s)
        
        n = len(next_s.status)
        
        #####################################
        # INFECTIOUS -> DETECTED TRANSITION #
        #####################################
        detection_time = self.period_nodetect + self.period_culling
        just_detected = (next_s.status == detection_time)
        
        if not (t in self.silent_spread_times):
            if np.any(just_detected):
            
                # Implement IP culling
                #next_s.being_culled[just_detected] = True
            
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
            
                # Save the date of the detection event
                next_s.when_reported[just_detected] = t
            
                # This only accounts for premises becoming IPs via reporting from a 
                # high level of status, what about vaccinated premises that need to
                # be culled?  
                next_s.once_an_ip[just_detected] = True
        
        ######################################
        # SUSCEPTIBLE -> INFECTED TRANSITION #
        ######################################
        # Can use this if Cython doesn't work
        #params = {'period_latent': self.period_latent}
        #status, event = iterate.infection_process(next_s.status, self.period_latent, self.grid, \
        #    self.Transmiss, self.Suscept, self.K, self.MaxRate, self.first_, self.last_, self.Num)
        
        status, event = cy.evolve_infection(next_s.status, self.MaxRate, self.grid, \
            self.Transmiss, self.Suscept, self.first_, self.last_, self.period_latent, \
            self.Num, self.K)
        
        ######################################
        # UPDATE VACCINE DURATION IN CATTLE  #
        ######################################
        # Those that have been vaccinated 1 day ago, are now 
        # vaccinated 2 days ago and so on
        if not (t in self.silent_spread_times):
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
        
            # Start off each step with a clean slate of who is a vacc candidate
        
            # Add IPs (that still have cattle) to list of culling candidates
            vacc_cand_cond1 = (next_s.n_cattle > 0)
            vacc_cand_cond2 = next_s.being_vacc
            vacc_candidate = vacc_cand_cond1 & vacc_cand_cond2
            
            # Perhaps check that they're *not* an IP too.  
            # Check that culling isn't occuring - culling should trump vaccination
        
            ########################
            # FIND VACC CANDIDATES #
            ########################
            if isinstance(current_a, core.Vacc) & np.any(just_detected):
            
                # Find those farms which we think are susceptible 
                # and upon which culling isn't occurring
                # (those which are subject to control actions)
            
                s_cond1 = (status < detection_time)
                s_cond2 = (status >= 0)
                s_cond3 = [not farm for farm in next_s.being_culled]
                susceptible_mask = s_cond1 & s_cond2 & s_cond3
            
                # Subset the distance matrix to find dists from all IPs to
                # all farms upon which we can implement control
                D_infectious = self.L.D[just_detected]
                D_inf_to_sus = D_infectious[:, susceptible_mask]
            
                # Find those farms within the vacc radius of IPs, 
                # output the distance to an IP, which IP was the one closest to the
                # farm in question
                if D_inf_to_sus.any():
                    nbhd, dist2ip, which_ip = utilities.NeighbourhoodDetail(D_inf_to_sus, \
                        current_a.outer_r, current_a.inner_r)
                
                    # Boolean list for whether each farms is a vacc candidate
                    vc = np.where(susceptible_mask)[0][nbhd]
                    vacc_candidate[vc] = True
                    #next_s.being_vacc[np.where(susceptible_mask)[0][nbhd]] = True
                
                    # Record the minimum distance to an IP (at this time step)
                    # Record which IP each farm is closest to.  
                    next_s.dist2IP[vc] = dist2ip
                    next_s.whichIP[vc] = which_ip
                
                    # Record the earliest time that this farm is listed to be vacc'd
                    next_s.when_reported[vacc_candidate] = \
                        np.minimum(next_s.when_reported[vacc_candidate], t)
            
            ########################
            # FIND CULL CANDIDATES #
            ########################
        
            # Start off each step with a clean slate of who is a cull candidate
        
            # Add IPs (that still have cattle) to list of culling candidates
            cull_cand_cond1 = next_s.once_an_ip #(next_s.when_reported < np.inf)
            cull_cand_cond2 = (next_s.n_cattle > 0)
            cull_cand_cond3 = next_s.being_culled
            culling_candidate = (cull_cand_cond1&cull_cand_cond2)|cull_cand_cond3
        
            # RING CULLING #
            if isinstance(current_a, core.Cull) & np.any(just_detected):
            
                # Find those farms which we think are susceptible 
                # (those which are subject to culling actions)
                s_cond1 = (status < detection_time)
                s_cond2 = (status >= 0)
                s_cond3 = [not farm for farm in next_s.being_vacc]
                susceptible_mask = s_cond1 & s_cond2 & s_cond3
            
                # From those that are being culled
                D_infectious = self.L.D[just_detected]
                D_inf_to_sus = D_infectious[:, susceptible_mask]
            
                if D_inf_to_sus.any():
                    # Find which are within the vaccination radius
                    nbhd, dist2ip, which_ip = utilities.NeighbourhoodDetail(D_inf_to_sus, \
                        current_a.outer_r, current_a.inner_r)
                
                    ############################
                    # IDENTIFY CULL CANDIDATES #
                    ############################
                
                    # Boolean list for whether each farms is a cull candidate
                    cc = np.where(susceptible_mask)[0][nbhd]
                    culling_candidate[cc] = True
                
                    next_s.dist2IP[cc] = dist2ip
                    next_s.whichIP[cc] = which_ip
                
                    # Record the earliest time that this farm is listed to be culled
                    next_s.when_reported[culling_candidate] = \
                        np.minimum(next_s.when_reported[culling_candidate], t)
        
            ######################################
            #       FILTER CULL CANDIDATES       # should this go before or after
            ###################################### the 'Being culled -> complete'
            # this doesn't account for those already being culled within
            # being_culled... yes it does
        
            # Need to add when_reported for RC farms
        
            # If 'cull_candidates' exists, then there are changes to be made
            # and we need to cull on these farms.  
            if np.any(culling_candidate):
            
                # Current number of carcasses
                current_carcasses = sum(next_s.n_carcass)
            
                # Subset the results to the size of the culling_candidate list
                c_being_culled = next_s.being_culled[culling_candidate]
                c_n_cattle = next_s.n_cattle[culling_candidate]
                c_once_an_ip = next_s.once_an_ip[culling_candidate]
                c_when_reported = next_s.when_reported[culling_candidate]
                c_dist2ip = next_s.dist2IP[culling_candidate]
            
                # Filter culled candidates based on a 'filter' function
                filtered_cull_candidates = self.filter_culls(c_being_culled, \
                    c_n_cattle, self.carcass_constraint, c_once_an_ip, \
                    c_when_reported, self.rate_cull, current_carcasses, c_dist2ip)
            
                candidate_indices = np.where(culling_candidate)[0]
            
                # Expand this subset back to the original size
                to_cull = np.array([False for ii in range(n)])
                to_cull[candidate_indices[filtered_cull_candidates]] = True
            
                next_s.being_culled = to_cull
            
                # Turn off any vaccination that may be occuring
                # (culling trumps vaccination)
                next_s.being_vacc[to_cull] = False
        
            #####################################
            #   FILTER VACCINATION CANDIDATES   #
            #####################################
        
            if np.any(vacc_candidate):
                # Subset the results to the size of the culling_candidate list
                c_being_vacc = next_s.being_culled[vacc_candidate]
                c_n_cattle = next_s.n_cattle[vacc_candidate]
                c_when_reported = next_s.when_reported[vacc_candidate]
                c_dist2ip = next_s.dist2IP[vacc_candidate]
            
                vacc_indices = np.where(vacc_candidate)[0]
            
                # Filter vacc candidates based on a 'filter' function
                filtered_vacc_candidates = self.filter_vacc(c_being_vacc, \
                    c_n_cattle, self.vacc_constraint, c_when_reported, \
                    c_dist2ip, self.rate_vacc)
                
                # Expand this subset back to the original size
                to_vacc = np.array([False for ii in range(n)])
                to_vacc[vacc_indices[filtered_vacc_candidates]] = True
            
                next_s.being_vacc = to_vacc
        
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
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            # is this correct? ... this needs to switch off when constraint is hit. 
            disposed = (next_s.n_carcass == 0) # culled & 
            next_s.being_disposed[disposed] = False
        
            ######################
            # IMPLEMENT DISPOSAL #
            ######################
        
            # If there are any carcasses, turn on disposal # <<<<< AND there is ability to clear them.  Or perhaps this needs to go BEFORE the 'being disposed of' section?  
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
            next_s = self._terminal_state
        
        return reward, next_s


class MCAgent(rli.Agent):
    """
    Class representing an agent that learns using epsilon-soft Monte Carlo control
    
    If no starting action is given then a random action is chosen.  
    """
    def __init__(self, actions, epsilon, control_switch_times = range(300001), \
            starting_action = None, verbose = False, update_value_fn = True):
        
        self._verbose = verbose
        self._epsilon = epsilon
        self._control_switch_times = control_switch_times
        self._starting_action = starting_action
        self._update_value_fn = update_value_fn
        
        # List all valid actions
        self._actions = actions
        
        self.Q = dict()
        self.visits = dict()
        self.sa_seen = set()
        
        for i, a in enumerate(self._actions):
            a.ind = i # count actions from 0, 1, 2, ...(starting action has no ind attribute)
        
        # Designate the starting action (after silent spread)
        self._starting_action = starting_action
        
        # List of durations resulting from each state-action pair.  
        # Dict keys are states; then each value is a multi-dimensional numpy array
        self.returns = dict()
    
    def start_trial(self, state):
        """
        Return starting action at the start of the trial
        """
        
        # Empty the list of seen state-action pairs
        self.sa_seen = set()
        
        if self.starting_action is None:
            action = random.choice(self.actions)
        else:
            action = self.starting_action
        
        # If Q[s] has not been seen before, create a table for it.
        if not(state in self.Q) and self.update_value_fn:
            self.Q[state] = np.empty(len(self.actions))
            self.Q[state][:] = 0.0
            self.visits[state] = 0
            
            # Create a numpy array to which we can append outbreak durations
            self.returns[state] = [[] for a in self.actions]
        
        return action
    
    def step(self, s, action, reward, next_s, t, *args):
        
        # Check for terminal state
        if(next_s == self._terminal_state):
            # Return a dummy action
            next_a = 20
            out_action = next_a
            
            # Loop through all state action pairs that we've seen, and append the return
            # and update the Q dictionary.  
            if self.update_value_fn:
                for s1, a1 in self.sa_seen:
                    # Record the observed state
                    self.visits[s1] +=1
                
                    # Duration of the outbreak is the variable t, append t to the returns[] list
                    self.returns[s1][a1.ind].append(t)
                
                    # Find the average of the durations for the state-action pair visited
                    self.Q[s1][a1.ind] = np.mean(self.returns[s1][a1.ind])
            
        else:
            # Check that it's time to change the action
            if t in self.control_switch_times:
                # If Q[s] has not been seen before, create a table for it.
                if not(s in self.Q) and self.update_value_fn:
                    self.Q[s] = np.empty(len(self.actions))
                    self.Q[s][:] = 0.0
                    self.visits[s] = 0
                    
                    # Create a numpy array to which we can append outbreak durations
                    self.returns[s] = [[] for a in self.actions]
                
                if not(next_s in self.Q) and self.update_value_fn:
                    self.Q[next_s] = np.empty(len(self.actions))
                    self.Q[next_s][:] = 0.0
                    self.visits[next_s] = 0
                    
                    # Create a numpy array to which we can append outbreak durations
                    self.returns[next_s] = [[] for a in self.actions]
                
                # Determine the list of probabilities of choosing an action
                action_probabilities = utilities.epsilon_soft(self.Q[s], self.actions, self.epsilon)
                
                next_a_idx = np.random.choice(len(self.actions), 1, p = action_probabilities)[0]
                next_a = self.actions[next_a_idx]
                # Convert action to an index
                ind = [i for i, v in enumerate(self.actions) if v == next_a][0]
                
                # Add the current state and action to the set of seen states
                if self.update_value_fn:
                    self.sa_seen.add((s, next_a))
                
            else:
                # Stay with the same action if we're not at a point where actions change
                # or at the final time step. 
                next_a = action
        
        return next_a
    
    @property
    def verbose(self):
        "Should verbose output be turned on?"
        return self._verbose
    
    @property
    def epsilon(self):
        "Epsilon parameter"
        return self._epsilon
    
    @property
    def actions(self):
        "Actions"
        return self._actions
    
    @property
    def update_value_fn(self):
        "Should we update the value function?"
        return self._update_value_fn
    
    @property
    def control_switch_times(self):
        "List of times at which control can switch"
        return self._control_switch_times
    
    @control_switch_times.setter
    def control_switch_times(self, value):
        "Allow control_switch_times to be set"
        self._control_switch_times = value
    
    @property
    def starting_action(self):
        "Starting action"
        return self._starting_action


class StaticAgent(MCAgent):
    def __init__(self, action, epsilon, starting_action, **kwargs):
        
        # Call the initialisation method of the parent class
        super(self.__class__, self).__init__(actions = action, epsilon = epsilon, \
            starting_action = starting_action, update_value_fn = False, **kwargs)
    
    def start_trial(self, state):
        if self.starting_action is None:
            action = random.choice(self.actions)
        else:
            action = self.starting_action
        
        return action
    
    def step(self, s, action, reward, next_s, t, *args):
        
        # Check for terminal state
        if(next_s == self._terminal_state):
            # Return a dummy action
            next_a = 20
            out_action = next_a
            
        else:
            # Check that it's time to change the action
            if t in self.control_switch_times:
                next_a = self.actions[0]
            else:
                # Stay with the same action if we're not at a point where actions change
                # or at the final time step. 
                next_a = action
        return next_a


class MCAgent_robust(MCAgent):
    """
    Agent used for investigating performance under different seeding conditions.  
    If the current state is not seen then the action is chosen according to the closest state
    in the value function (via Euclidean distance).  
    """
    def __init__(self, actions, epsilon, update_value_fn = False, **kwargs):
        
        # Call the initialisation method of the parent class
        super(self.__class__, self).__init__(actions = actions, epsilon = 0.0, \
            update_value_fn = False, **kwargs)
    
    def start_trial(self, state):
        """
        Return starting action at the start of the trial
        """
        
        if self.starting_action is None:
            action = random.choice(self.actions)
        else:
            action = self.starting_action
        
        return action
    
    def step(self, s, action, reward, next_s, t, *args):
        
        # Check for terminal state
        if(next_s == self._terminal_state):
            # Return a dummy action
            next_a = 20
            out_action = next_a
        else:
            # Check that it's time to change the action
            if t in self.control_switch_times:
                
                # If Q[s] has not been seen before, find action that's closest
                if not(s in self.Q):
                    keys = list(self.Q.keys())
                    
                    distances = dist.cdist(keys, [s], 'euclidean')
                    ind = np.argmin(distances)
                    s = tuple(keys[ind])
                
                # Determine the list of probabilities of choosing an action
                action_probabilities = utilities.epsilon_soft(self.Q[s], self.actions, self.epsilon)
                
                next_a_idx = np.random.choice(len(self.actions), 1, p = action_probabilities)[0]
                next_a = self.actions[next_a_idx]
                
            else:
                # Stay with the same action if we're not at a point where actions change
                # or at the final time step. 
                next_a = action
        
        return next_a


def choose_cull_candidates(status_being_culled, n_cattle, carcass_limit_global,
    once_an_ip, when_reported, rate_cull, current_carcasses, *args):
    
    """
    Choose farms from cull candidates to actually cull.  
    
    This function looks at the index of farms listed in culling_candidate and 
    filters them based on sorting rules and the global culling constraint.  A 
    new list is returned of filtered farm indices of those that should be culled
    based on ordering farms on:
    1) Whether they are currently being culled or not
    2) Whether they are an IP (infected premises) or an RC (ring cull)
    3) Day notified/reported
    
    This assumes that all farms input are candidates to be culled.  
    
    Notes...needs to take into account the current number of carcasses... and
    the global limit... what about the current disposal rate?  
    
    
    INPUT
        status_being_culled
            (list; boolean) A list of which farms are currently being culled
        
        n_cattle:
            (list; int)     A list of the number of cattle in each farm
        
        carcass_limit_global
            (int)           The global carcass limit
        
        once_an_ip
            (list; boolean) A list of whether a farm was ever an IP
        
        when_reported
            (list; int)     List of first day each farm became a cull candidate
        
        rate_cull
            (int)           Per-day, per-farm cull rate
        
        current_carcasses
            (int)           Number of carcasses currently
    
    OUTPUT
        to_cull
            (list; boolean) A list of farms to cull
    
    """
    
    # Order based upon 1) IP or not (IPs before ring culls), 2) whether they 
    # are currently being culled or not, and 3) reporting date. 
    
    rev_IP = [not x for x in once_an_ip]
    rev_being_culled = [not x for x in status_being_culled]
    cull_order = np.lexsort([when_reported, rev_IP, rev_being_culled])
    
    cattle_able_to_be_culled = np.minimum(n_cattle, rate_cull)
    
    # Find the cumulative size of cattle to be culled in one time step
    cum_culls = np.cumsum(cattle_able_to_be_culled[cull_order])
    
    # Find which farms can be culled based on current capacity and constraints
    who_to_cull = (cum_culls <= (carcass_limit_global - current_carcasses))
    
    # Transform back to the original shape
    filtered_culls = who_to_cull[np.argsort(cull_order)]
    
    return(filtered_culls)


def choose_cull_candidates2(status_being_culled, n_cattle, carcass_limit_global,
    once_an_ip, when_reported, rate_cull, current_carcasses, dist2ip):
    
    """
    Choose farms from cull candidates to actually cull.  
    
    This builds upon choose_cull_candidates and includes the distance to the IP
    that designated the farm as the cull candidate.  
    """
    
    # Order based upon 1) IP or not (IPs before ring culls), 2) whether they 
    # are currently being culled or not, and 3) reporting date, 4) distance to
    # IP that earmarked them to be culled.  
    
    rev_IP = [not x for x in once_an_ip]
    rev_being_culled = [not x for x in status_being_culled]
    cull_order = np.lexsort([-dist2ip, -when_reported, rev_being_culled, rev_IP])
    
    cattle_able_to_be_culled = np.minimum(n_cattle, rate_cull)
    
    # Find the cumulative size of cattle to be culled in one time step
    cum_culls = np.cumsum(cattle_able_to_be_culled[cull_order])
    
    # Find which farms can be culled based on current capacity and constraints
    who_to_cull = (cum_culls <= (carcass_limit_global - current_carcasses))
    
    # Transform back to the original shape
    filtered_culls = who_to_cull[np.argsort(cull_order)]
    
    return(filtered_culls)


def choose_vacc_candidates(status_being_vacc, n_cattle, vacc_constraint, 
    when_reported, dist2ip, rate_vacc):
    """
    Choose farms from vacc candidates to actually vaccinate.  
    
    This function looks at the index of farms listed in vacc_candidate and 
    filters them based on sorting rules and the global culling constraint.  
    A new list is returned of filtered farm indices of those that should be 
    vaccinated based on ordering farms on:
    1) Whether they are currently being vaccinated or not
        (currently being vacc'd are prioritised over not being vacc'd)
    2) Day at which they were designated to be vaccinated
        ()
    3) Distance from the IP that designated them to be vaccinated
        (distances further away are prioritised first)
    
    This assumes that the input lists are of all farms that are candidates
    to be vaccinated.  
    
    Notes...needs to take into account the current number of carcasses... and
    the global limit... what about the current disposal rate?  
    
    INPUT
        status_being_vacc
            (list; boolean) A list of which farms are currently being vaccinated
        
        n_cattle:
            (list; int)     A list of the number of cattle in each farm
        
        vacc_constraint
            (int)           The global daily capacity for vaccination (head/day)
        
        when_reported
            (list; int)     List of first day each farm became a vacc candidate
        
        rate_vacc
            (int)           Per-day, per-farm vacc rate
        
        dist2IP
            (list; float)   Distance to the IP that designated this farm to be 
                            a vaccination candidate.  
        
    OUTPUT
        to_vacc
            (list; boolean) A list of farms to vaccinate
    
    EXAMPLE
    
    
    """
    
    # Find those not currently being vaccinated
    rev_being_vacc = [not x for x in status_being_vacc]
    
    # Order based upon 1) whether they are currently being vaccinated or not, 
    # 2) day at which they were designated to be vaccinated, 3) distance from 
    # any IP on the day they were designated to be vaccinated
    vacc_order = np.lexsort([-dist2ip, -when_reported, rev_being_vacc])
    
    # Number of cattle on each farm that *can* be vaccinated
    cattle_able_to_be_vax = np.minimum(n_cattle, rate_vacc)
    
    # Find the cumulative size of cattle to be culled in one time step
    cum_vax = np.cumsum(cattle_able_to_be_vax[vacc_order])
    
    # Find which farms can be culled based on current capacity and constraints
    who_to_vacc = (cum_vax <= vacc_constraint)
    
    # Transform back to the original shape
    filtered_vax = who_to_vacc[np.argsort(vacc_order)]
    
    return(filtered_vax)
