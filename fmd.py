"""
Simulation of foot-and-mouth disease outbreaks and various control actions
"""

import numpy as np
import pandas as pd
import copy, time, csv
from collections import defaultdict
import scipy.spatial as sp

from os.path import join # for saving the state

# This is housed locally
import cy

def FMDSpread(x, y, cattle, sheep, start, num_trials = None, \
    control_type = "cull", radius = None, time_max = 1000, verbose = False, \
    DIST = None, kern = None, iterate_fn = None, dcculling = True, \
    
    delay_ipcull = 1, delay_dccull = 2, delay_ctrlcull = 3, \
    delay_ipdisposal = 1, delay_dcdisposal = 1, delay_ctrldisposal = 2, \
    delay_latency = 5, delay_immune = 5, delay_detect = 4, \
    
    dc_f = 0.84, dc_F = 6.0, \
    
    detailed = False, plotting = False, \
    set_params = "KEELING2001", silent_spread = None, \
    rand_index_status = False, maxringcull = 100, cull_eff = 0.8, \
    maxvacc = 35000, vacc_eff = 0.9, \
    
    time_save_state = [], save_dir = None, \
    
    **kwargs):
    
    """
    Run a set of outbreaks of foot-and-mouth disease
    
        Parameters
        -----------
          - x, y : np.array 
              x, y coordinates of premises
          - cattle, sheep : np.array or int
              number of cattle/sheep on each premises; if integer is passed
              then assumed this is the number on each farm
          - start : np.array int
              starting infection status
          - num_trials : int
              number of outbreaks to simulate
          - DIST : np.array or None
              distance matrix of between-premises distances; dim len(x)*len(x);
              calculated using scipy.spatial.distance.cdist if not provided
          - kern : function
              kernel function for evaluating distances (in km2)
          - iterate_fn : function
              function for evolving the infection at each time step
          - set_params : string
              parameter set to use; see below (default is "KEELING2001")
          - silent_spread : int
              number of days of silent spread to simulate (default 0)
          - rand_index_status : int or bool
              choose a random seed case?  
          - **kwargs : 
              further keyword arguments
            
        Control-related parameters:
          - control_type : string
              type of control to implement (see below)
          - radius : float 
              ring radius (km) to be used for vaccination or culling
          - dcculling : bool
              should culling of dangerous contacts be initiated?  
          - maxringcull : int
              max premises that can be ring culled per timestep (default: 100)
          - cull_eff : float
              efficacy of culling, a factor giving the reduced transmissibility
              of carcasses (between culling and disposal) (default: 1.0)
          - maxvacc : int
              max cattle that can be vaccinated per timestep (default: 35000)
          - vacc_eff : float
              vaccine efficacy, a factor giving the reduced transmissibility and
              susceptibility of 'immune' farms (default: 0.9)
            
        Delay-related parameters:
          type: (int | np.array | callable func)
          
          All delays can be passed as an integer (assuming all premises have 
          the same delay) or an array (individual-level delays) or a callable 
          function (a function that returns an array of length func(N) where N 
          is the number of premises in the simulation).  Units for delays are 
          time-steps/days.  For example, a callable function could be:
          
          import numpy as np
          delay_latency = lambda x: np.random.gamma(0.3, 2.4, x)
          
          - delay_ipcull : see above
              delay for IP culling
          - delay_dccull : see above
              delay for dangerous contacts (DC) culling
          - delay_ctrlcull : see above
              delay for control culls
          - delay_ipdisposal : see above
              delay for IP disposal
          - delay_dcdisposal : see above
              delay for dangerous contacts disposal
          - delay_ctrldisposal : see above
              delay for control disposal
          - delay_latency : see above
              delay from exposure until infectious
          - delay_immune : see above
              delay from vaccination until conferment of immunity
          - delay_detect : see above
              delay from becoming infectious to being detected
              (Keeling et al., 2001)
        
        Simulation related parameters:
          - time_max : int
              maximum allowed number of time steps
          - verbose : bool
              should extra information be printed to screen during running?
          - detailed : bool
              should data for each simulated outbreak be output?
          - plotting : bool
              should the current outbreak be plotted while being simulated?
          - time_save_state : list
              list of timesteps at which to output the full state
        
        Available control types:
          - `cull` : ring culling
          - `vacc` : ring vaccination
          - `early` : DEPRECATED, culling of infected premises with early
                     detection (2 days instead of 4 days)
        
        Available parameter sets:
          - `KEELING2001` : Keeling et al. (2001) Science (default)
          - `TILDESLEY2006` : Tildesley et al. (2006) Nature
          - `TILDESLEY2008_CUMBRIA` : Tildesley et al. (2008) Proc. B
          - `INPUT` : 
        
        Returns
        -------
          - pandas DataFrame of summaries from each simulated outbreak.  If 
            detailed = True then returns a detailed summary of each individual
            outbreak (only recommended for outputting one outbreak at a time).
    """
    
    # Process the input arguments
    N = len(x)
    
    # If animal numbers are given as integers, convert to vector
    if isinstance(cattle, int):
        cattle = np.ones(N, dtype = int)*cattle
    
    if isinstance(sheep, int):
        sheep = np.ones(N, dtype = int)*sheep
    
    coords = np.asarray([(a, b) for a, b in zip(x, y)])
    
    # Set up a dictionary to store the results convert to a pandas.DataFrame
    # at the end of the script, using a defaultdict(list) allows using the 
    # .append method for any list that's a value within the dict.  
    results = defaultdict(list)
    
    # Calculate the pair-wise distance matrix btw farms (unless it's given)
    # distance is squared euclidean distance by default
    if DIST is None:
        import scipy.spatial as sp
        if verbose:
            ("Calculating distance matrix ... \n")
        
        COORDS = [(a, b) for a, b in zip(x,y)]
        DIST = sp.distance.cdist(COORDS, COORDS, 'sqeuclidean')
        
        if verbose:
            print(" ... distance matrix calculated\n")
    
    if kern is None:
        kern = UKKernel
    KDIST = kern(DIST)
    
    if iterate_fn is None:
        iterate_fn = IterateWho
    
    if verbose:
        print("Setting up delay parameters ... ")
    
    # Delay parameters (convert to array if they're an int)
    # convert to arrays from int if necessary
    if isinstance(delay_latency, int):
        delay_latency_i = delay_latency * np.ones(N, dtype = int)
    elif isinstance(delay_latency, np.ndarray):
        delay_latency_i = delay_latency
    
    if isinstance(delay_immune, int):
        delay_immune_i = delay_immune * np.ones(N, dtype = int)
    elif isinstance(delay_immune, np.ndarray):
        delay_immune_i = delay_immune
    
    if isinstance(delay_ipcull, int):
        delay_ipcull_i = delay_ipcull * np.ones(N, dtype = int)
    elif isinstance(delay_ipcull, np.ndarray):
        delay_ipcull_i = delay_ipcull
    
    if isinstance(delay_dccull, int):
        delay_dccull_i = delay_dccull * np.ones(N, dtype = int)
    elif isinstance(delay_dccull, np.ndarray):
        delay_dccull_i = delay_dccull
    
    if isinstance(delay_ctrlcull, int):
        delay_ctrlcull_i = delay_ctrlcull * np.ones(N, dtype = int)
    elif isinstance(delay_ctrlcull, np.ndarray):
        delay_ctrlcull_i = delay_ctrlcull
     
    if isinstance(delay_ipdisposal, int):
        delay_ipdisposal_i = delay_ipdisposal * np.ones(N, dtype = int)
    elif isinstance(delay_ipdisposal, np.ndarray):
        delay_ipdisposal_i = delay_ipdisposal
    
    if isinstance(delay_dcdisposal, int):
        delay_dcdisposal_i = delay_dcdisposal * np.ones(N, dtype = int)
    elif isinstance(delay_dcdisposal, np.ndarray):
        delay_dcdisposal_i = delay_dcdisposal
    
    if isinstance(delay_ctrldisposal, int):
        delay_ctrldisposal_i = delay_ctrldisposal * np.ones(N, dtype = int)
    elif isinstance(delay_ctrldisposal, np.ndarray):
        delay_ctrldisposal_i = delay_ctrldisposal
    
    if isinstance(delay_detect, int):
        delay_detect_i = delay_detect * np.ones(N, dtype = int)
    elif isinstance(delay_detect, np.ndarray):
        delay_detect_i = delay_detect
    
    if verbose:
        print("... done")
    
    # Calculate cull/disposal times
    
    #if isinstance(delay_latency, (int, np.ndarray)) & 
    #    isinstance(delay_detect, (int, np.ndarray)):
    #    #time_undetect = delay_latency_i + delay_detect_i
    
    #time_dccull = time_undetect + delay_dccull_i
    #time_ctrldisposal = time_ctrlcull + delay_ctrldisposal_i
    #time_ipdisposal = time_ipcull + delay_ipdisposal_i
    
    # Disposal parameters, Cull -> Disposed
    # Relative transmissibility of carcasses to live animals
    culling_efficacy = (1.0 - cull_eff) * np.ones(N, dtype = float)
    
    spark = 0.0
    
    if verbose:
        print("Setting parameters")
    
    # INPUT params from the Fortran model
    if set_params == "INPUT":
        
        cattle_sus = 9.952; sheep_sus = 1.0
        
        # Per-capita transmissibility
        cattle_trans = 1.589e-6; sheep_trans = 9.16e-7
        
        # Non-linearity terms for sheep
        sheep_pows = 0.201; sheep_powt = 0.489
        
        # Non-linearity terms for cattle
        cattle_pows = 0.412; cattle_powt = 0.424
        
    # INPUT params from the Keeling et al (2001) (and 2003)
    if set_params == "KEELING2001":
        
        cattle_sus = 15.2; sheep_sus = 1.0
        
        # Per-capita transmissibility
        cattle_trans = 4.3e-7; sheep_trans = 2.67e-7
        
        # Non-linearity terms for sheep
        sheep_pows = 1.0; sheep_powt = 1.0
        
        # Non-linearity terms for cattle
        cattle_pows = 1.0; cattle_powt = 1.0
        
    elif set_params == "TILDESLEY2008_CUMBRIA":
        # Infection parameters (See Tildesley et al. (2008))
        # (the model in this paper doesn't use powers on animal numbers)
        # Per-capita susceptibility
        cattle_sus = 180.7581308041645# 9.952 #5.7; 
        sheep_sus = 168.82184642109027#1.0
        
        # Per-capita transmissibility
        cattle_trans = 2.729748128451626E-05# 15.89E-7 #8.2E-4; 
        sheep_trans = 2.7523730108211333E-05# 9.16E-7 #8.3E-4
        
        # Non-linearity terms for sheep
        sheep_pows = 0.201; sheep_powt = 0.489
        
        # Non-linearity terms for cattle
        cattle_pows = 0.412; cattle_powt = 0.424
        
    elif set_params == "TILDESLEY2006":
        # Infection parameters (See Tildesley et al. (2006))
        # (the model in this paper doesn't use powers on animal numbers)
        # Per-capita susceptibility
        cattle_sus = 10.5; sheep_sus = 1.0
        
        # Per-capita transmissibility
        cattle_trans = 7.7e-7; sheep_trans = 5.1e-7
        
        # Non-linearity terms for sheep
        sheep_pows = 1.0; sheep_powt = 1.0
        
        # Non-linearity terms for cattle
        cattle_pows = 1.0; cattle_powt = 1.0
    else:
        if verbose:
            print("Unknown parameter set, using TILDESLEY2006")
        # Infection parameters (See Tildesley et al. (2006))
        # (the model in this paper doesn't use powers on animal numbers)
        # Per-capita susceptibility
        cattle_sus = 10.5; sheep_sus = 1.0
        
        # Per-capita transmissibility
        cattle_trans = 7.7e-7; sheep_trans = 5.1e-7
        
        # Non-linearity terms for sheep
        sheep_pows = 1.0; sheep_powt = 1.0
        
        # Non-linearity terms for cattle
        cattle_pows = 1.0; cattle_powt = 1.0
    
    if verbose: 
        print("Calculating susceptibility and transmissibility")
    
    # Set up a vector of zeros that designate how many days a premises
    # has been waiting to be culled/dispose
    wait_cull = -1 * np.ones(N, dtype = int)
    wait_dispose = -1 * np.ones(N, dtype = int)
    
    # Record the starting number of animals on each farm
    starting_cattle = copy.copy(cattle)
    starting_sheep = copy.copy(sheep)
    
    # Calculate the farm-level susceptibility and transmissibility
    start_Suscept = sheep_sus*(sheep**sheep_pows) + \
        cattle_sus*(cattle**cattle_pows)
    
    start_Transmiss = sheep_trans*(sheep**sheep_powt) + \
        cattle_trans*(cattle**cattle_powt)
    
    gx = kwargs.get('gx', 1)
    gy = kwargs.get('gy', 1)
    if verbose:
        print("Calculating grid ... ")
    grid, xbins, ybins = Coords2Grid(x, y, gx, gy)
    
    if verbose:
        print("... done")
    
    if verbose:
        print("Ordering all farms according to grid cell ... ")
    # Sort all farms according to their grid cell ID
    # so that indexing on grid number is quick
    IND = np.argsort(grid)
    grid = grid[IND]
    
    x = x[IND]; y = y[IND]
    cattle = cattle[IND]; sheep = sheep[IND]
    start = start[IND]
    
    if verbose:
        print("... done")
    
    if verbose:
        print("Calculating grid-to-grid probabilities ... ")
    MaxRate, Num, first_, last_, max_sus, Dist2all, KDist2all = calc_grid_probs(grid, gx, gy, start_Suscept, kern,\
        x.ptp(), y.ptp())
    
    if verbose:
        print("... done")
    
    # Culling parameters (see Tildesley et al. (2009))
    # For UK 2001, f : [0.84, 0.9]; F : [3.5, 9.0]
    #dc_f = 0.84 # accuracy of DC culling (ability to detect routes of trans'n)
    #dc_F = 6.0 # overall level of DC culling per reported case
    # F = 6 is from SI of Tildesley et al. (2006)
    
    # Recording the farm that infected each other farm
    # Default value is -1 (-2 for RC)
    who_infected_me = -1 * np.ones(N, dtype = int)
    
    # Set the variable of candidates
    candidatesRc = np.array([])
    candidatesVc = np.array([])
    
    # Keep a record of the dangerous contacts
    status_dc = np.array([])
    
    # Container to hold the per-farm results for one outbreak
    if detailed:
        output = []
        canimals = []
        sanimals = []
        allcarcasses = []
        alltransmiss = []
    
    # Run a range of outbreak
    for trial in range(num_trials):
        
        if verbose:
            print("Outbreak number : ", trial)
        
        # Reset variables
        t = 0
        outbreak_go = True
        status = copy.copy(start)
        Suscept = copy.copy(start_Suscept)
        Transmiss = copy.copy(start_Transmiss)
        cattle = copy.copy(starting_cattle)
        sheep = copy.copy(starting_sheep)
        
        # Set up vectors showing those designated to be ring culled
        # and those designated to be DC culls
        IP = np.zeros(N, dtype = int)
        RC = np.zeros(N, dtype = int)
        DC = np.zeros(N, dtype = int)
        vstatus = np.zeros(N, dtype = int); V = 0
        carcasses = np.zeros(N, dtype = int)
        
        # Delay-related parameters
        if callable(delay_latency):
            delay_latency_i = delay_latency(N)
        
        if callable(delay_immune):
            delay_immune_i = delay_immune(N)
        
        if callable(delay_detect):
            delay_detect_i = delay_detect(N)
        
        if callable(delay_ipcull):
            delay_ipcull_i = delay_ipcull(N)
        
        if callable(delay_dccull):
            delay_dccull_i = delay_dccull(N)
        
        if callable(delay_ctrlcull):
            delay_ctrlcull_i = delay_ctrlcull(N)
        
        if callable(delay_ipdisposal):
            delay_ipdisposal_i = delay_ipdisposal(N)
        
        if callable(delay_dcdisposal):
            delay_dcdisposal_i = delay_dcdisposal(N)
        
        if callable(delay_ctrldisposal):
            delay_ctrldisposal_i = delay_ctrldisposal(N)
        
        # Seed cases
        if rand_index_status:
            status[np.random.randint(0, len(status), 1)] = rand_index_status
        
        # Run using silent spread
        if silent_spread:
            if verbose:
                print("Running silent spread for %d days", silent_spread)
            
            for i in range(silent_spread):
                status, new_infections, infectors = iterate_fn(status, KDIST, 
                        Suscept, Transmiss, delay_latency_i, spark)
        
        # Count the number of seed cases
        numseeds = np.count_nonzero(status)
        
        # Set the state
        state_area = 0
        
        if plotting:
            from matplotlib import pyplot as plt
            import matplotlib
            matplotlib.interactive(True)
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            ax.set_aspect('equal')
            # Colors
            orange = (255./255, 127./255, 0./255, 1) # Exposed
            red = (228./255, 26./255, 28./255, 1) # Infected
            grey_tint = 200. # Removed or immune
            grey = (grey_tint/255, grey_tint/255, grey_tint/255, 1)
            blue = (18./255, 15/255, 202./255, 1.)
            purple = (152./255, 78./255, 163./255, 1.) # Being disposed of
            cyan = (255./255, 255./255, 51./255, 1.) # Being vaccinated
            sus_col = (65./255, 173./255, 0./255, 1)
            face_colors = [ sus_col for ii in range(N) ]
            
            # Plot the starting state of the outbreak
            msize = 1
            scatter_plot = ax.scatter(x, y, s = msize, linewidths = 2) 
            
            # Set edge and face colors the same
            scatter_plot.set_facecolor(face_colors)
            scatter_plot.set_edgecolor(face_colors)
            fc = face_colors
            
            # Turn all the axes off
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
            
            # Draw the canvas
            fig.canvas.draw()
        
        while ( (t < time_max) & outbreak_go):
            
            if t in time_save_state:
                # If this is a time at which we should save the state, then save full state 
                # of the outbreak in desired folder
                filename_state = join(save_dir, 
                    "state_trial{:03d}_t{:03d}.csv".format(trial, t))
                
                with open(filename_state, 'w') as csvfile:
                    statewriter = csv.writer(csvfile, delimiter = ',')
                    statewriter.writerow(['id', 'x', 'y', 'n_cattle', 'n_sheep', 'status'])
                    statewriter.writerows(zip(np.arange(N)+1, x, y, cattle, sheep, status))
            
            # Find the "notified" vertices
            notified_ind = np.where((status >= delay_latency_i + delay_detect_i) | (status < 0))[0]
            notified_vertices = coords[notified_ind]
            
            if len(notified_vertices) > 3:
                try:
                    # Calculate the observed area of the outbreak
                    hull = sp.ConvexHull(notified_vertices)
                    v = hull.vertices
                    state_area = cy.hull_area(
                        hull.points[v].transpose()[0],
                        hull.points[v].transpose()[1])
                    if t == 7:
                        print("State (time 7):", state_area, 
                            np.sum(status >= delay_latency_i + delay_detect_i))
                except:
                    print("Points are coplanar")
                else:
                    pass
            
            # Iterate the timer
            t += 1
            
            if detailed: 
                # Append the results
                output.append(copy.copy(status))
                canimals.append(copy.copy(cattle))
                sanimals.append(copy.copy(sheep))
                allcarcasses.append(copy.copy(carcasses))
                alltransmiss.append(copy.copy(Transmiss))
            
            if verbose:
                print("Time step: ", t)
            
            # Iterate the infection process
            status, new_infections, infectors = iterate_fn(status, KDIST,
                   Suscept, Transmiss, delay_latency_i, spark)
            # status, new_infections, infectors = IterateGrid(status, grid, delay_latency, \
            #         Transmiss, Suscept, Num, MaxRate, first_, last_, kern, coords)
            
            # Update the vector showing who infected whom
            if len(new_infections) > 0:
                who_infected_me[new_infections.astype(int)] = infectors
                IP[new_infections.astype(int)] = 1
            
            # Update the waiting times
            wait_cull[wait_cull >= 0] += 1
            wait_dispose[wait_dispose >= 0] += 1
            
            # Find the farms of different compartments
            Sus = np.where((status == 0) & (vstatus == 0))[0]
            Exposed = np.sum(status > 0)
            
            #E = np.sum(Exp); I = np.sum(Infd); R2 = np.sum(Rep)
            
            if (control_type == "vacc"):
                Vacc = (vstatus > 0)
                V = np.sum(Vacc)
                Imm = (who_infected_me == -3)
            
            # If any premises that were being vaccinated are now infected, 
            # switch off their vaccination status
            if (control_type == "vacc"):
                if len(new_infections) > 0:
                    if len(Vacc[new_infections]) > 0:
                    
                        vstatus[new_infections] = 0
                        Vacc = (vstatus > 0)
                        V = np.sum(Vacc)
            
            # Update the plot of the outbreak
            if plotting:
                
                Exp = (status > 0) & (status < delay_latency_i)
                Infd = (status >= delay_latency_i) & (status < (delay_latency_i + delay_detect_i))
                Rep = (status >= (delay_latency_i + delay_detect_i))
                Culled = (status == -1)
                
                # Update the colors
                fc = [orange if u else v for u, v in zip(Exp, fc)]
                fc = [red if u else v for u, v in zip(Infd, fc)]
                fc = [grey if u else v for u, v in zip(Culled, fc)]
                
                if (control_type == "vacc"):
                    fc = [grey if u else v for u, v in zip(Imm, fc)]
                    fc = [cyan if u else v for u, v in zip(Vacc, fc)]
                
                # Update activities (sometimes using edge colours)
                ec = fc
                
                #fc = [purple if x else y for x, y in zip(bd, fc)]
                #ec = [blue if x else y for x, y in zip(bc, ec)]
                
                scatter_plot.set_facecolor(fc)
                scatter_plot.set_edgecolors(ec)
                rest_time = 0.02
                time.sleep(rest_time)
                fig.canvas.draw()
            
            # Check for culling of infected farms
            ipculls = np.where(status >= delay_latency_i + \
                delay_detect_i + delay_ipcull_i)[0]
            
            # Check for disposal on infected farms
            ipdispose = np.where(status >= delay_latency_i + \
                delay_detect_i + delay_ipcull_i + delay_ipdisposal_i)[0]
            
            # Iterate vaccination on farms that have been vaccinated
            if (control_type == "vacc"):
                if (np.sum(vstatus) > 0): # is np.any faster here?  
                    vstatus[vstatus > 0] += 1
            
            ########################
            ## Perform IP culling ##
            ########################
            if( (len(ipculls) > 0)):
                
                carcasses[ipculls] += cattle[ipculls] + sheep[ipculls]
                cattle[ipculls] = 0
                sheep[ipculls] = 0
                Transmiss[ipculls] *= culling_efficacy[ipculls]
                wait_cull[ipculls] = -1
                RC[ipculls] = 0
                DC[ipculls] = 0
                
                ########################
                ## Perform DC culling ##
                ########################
                if dcculling:
                    #print("DC culling")
                    # If DC culling is to be performed.  
                    # the prob that farm i is a DC from IP j is :
                    #  1 - f exp(-F . Rate(i,j))
                    #print(who_infected_me)
                    #print(ipculls)
                    # List premises that are both currently being culled
                    # and also infected other farms (candidate DCs)
                    candDCs = np.where(np.in1d(who_infected_me, ipculls))[0]
                    
                    NDC = len(candDCs)
                    
                    # Check that these farms are not yet reported ... 
                    
                    # If there are some candidate DCs
                    if(NDC > 0):
                        
                        # List the IPs that infected each candidate DC
                        candIPs = who_infected_me[candDCs]
                        NIP = len(candIPs)
                        
                        #K = KDIST[np.ix_(candIPs, candDCs)]
                        #T=np.tile(Transmiss[candIPs].reshape(NIP, 1),(1, NDC))
                        #S=np.tile(Suscept[candDCs],(NIP, 1))
                        #P_DC = 1 - dc_f * np.exp(-dc_F * T * S * K)
                        
                        TSK = Transmiss[candIPs][:,None] * \
                                Suscept[candDCs][None,:] * \
                                KDIST[np.ix_(candIPs, candDCs)]
                        
                        P_DC = 1 - dc_f * np.exp(-dc_F * TSK)
                        
                        DC_EVENT = (P_DC > np.random.rand(NIP, NDC))
                        DC_EVENT = candDCs[(np.sum(DC_EVENT, axis = 0) > 0)]
                        
                        # Start the waiting time
                        wait_cull[DC_EVENT] = 0
                        
                        # Designate this premises as a DC cull
                        DC[DC_EVENT] = 1
                
                ##################
                ## Ring culling ##
                ##################
                
                # Designate candidates for ring culling around infected premises
                # if control type is 'cull' and the cull radius is not None
                
                if ((control_type == "cull") & (radius is not None)):
                    if verbose:
                        print("Ring culling")
                    
                    # Choose the ring cull candidates as those susceptibles 
                    # within the cull radius.  
                    
                    inCullRadius = DIST[np.ix_(ipculls, Sus)] < radius
                    inAnyCullRadius = (np.sum(inCullRadius, axis = 0) > 0)
                    
                    # Determine the ring culling candidates
                    candidatesRc = Sus[inAnyCullRadius]
                
                ######################
                ## Ring vaccination ##
                ######################
                
                # Designate candidates for ring vacc'n around infected premises
                # if control type is 'vacc' and the vacc'n radius is not None
                
                if ((control_type == "vacc") & (radius is not None)):
                    if verbose:
                        print("Ring vaccination")
                    
                    inVaccRadius = DIST[np.ix_(ipculls,Sus)] < radius
                    inAnyVaccRadius = (np.sum(inVaccRadius, axis = 0) > 0)
                    n = Sus[inAnyVaccRadius]
                    
                    # If the premises in question are susceptible then they can
                    # be vaccination candidates
                    candidatesVc = n[status[n] == 0]
            
            # If any disposal is to take place (with was checked before the
            # outbreak has been evolved using the 'Iterate' function), 
            # then remove carcasses from the pile update transmissibility
            if len(ipdispose) > 0:
                carcasses[ipdispose] = 0
                Transmiss[ipdispose] = 0
                Suscept[ipdispose] = 0
                status[ipdispose] = -1
            
            ################################
            # Start timers for vaccination #
            ################################
            
            # Choose candidates to vaccinate (if any constraints on vacc)
            # then start their timers
            if (control_type == "vacc"):
                if candidatesVc.any():
                
                    # Choose the actual premises to vaccinate
                    VCs = candidatesVc
                    
                    # Take a subset of the vacc candidates (if needed)
                    if np.sum(cattle[candidatesVc]) > maxvacc:
                        np.random.shuffle(candidatesVc)
                        within_budget = (np.cumsum(cattle[candidatesVc]) <= maxvacc)
                        VCs = candidatesVc[within_budget]
                    
                    # Set a counter on these for how many days they've been 
                    # vaccinated for.  
                    vstatus[VCs] = 1
                
                    # Reset the variable holding the vacc candidates
                    candidatesVc = np.array([])
            
            ####################################
            # Start timers for control culling #
            ####################################
            
            # Choose candidates to ring cull (if any constraints on culling)
            # then start their timers
            if candidatesRc.any():
                
                # Choose the actual premises to ring cull
                RCs = candidatesRc
                
                # Take a subset of the ring-cull candidates if needed
                if len(candidatesRc) > maxringcull:
                    np.random.shuffle(candidatesRc)
                    RCs = candidatesRc[0:maxringcull]
                
                # Set the waiting time to zero
                wait_cull[RCs] = 0
                RC[RCs] = 1
                
                # Reset the variable holding the candidates
                candidatesRc = np.array([])
            
            ###########################
            # Implement vacc immunity #
            ###########################
            if (control_type == "vacc"):
                # Iterate vaccination on premises that have been vaccinated
                immunes = (vstatus >= delay_immune_i)
            
                if np.sum(immunes).any():
                
                    who_infected_me[immunes] = -3
                    vstatus[immunes] = 0
                    #status[immunes] = -2 # leave these as susceptible
                    Transmiss[immunes] *= (1. - vacc_eff)
                    Suscept[immunes] *= (1. - vacc_eff)
            
            #############################
            # Implement control culling #
            #############################
            
            ctrlculls = np.where(wait_cull == delay_ctrlcull_i)[0]
            
            # Perform culling on farms that have been designated as ring culls
            if (len(ctrlculls) > 0):
                
                who_infected_me[ctrlculls] = -2
                carcasses[ctrlculls] += cattle[ctrlculls] + sheep[ctrlculls]
                cattle[ctrlculls] = 0
                sheep[ctrlculls] = 0
                Transmiss[ctrlculls] *= culling_efficacy[ctrlculls]
                status[ctrlculls] = -1
                wait_cull[ctrlculls] = -2
                
                # Set the disposal wait time to zero
                wait_dispose[ctrlculls] = 0
            anycull = np.any(wait_cull >= 0)
            
            ##############################
            # Implement control disposal #
            ##############################
            
            ctrldispose = np.where(wait_dispose == delay_ctrldisposal_i)[0]
            
            if (len(ctrldispose) > 0):
                carcasses[ctrldispose] = 0
                wait_dispose[ctrldispose] = -2
            
            anydispose = np.any(wait_dispose >= 0)
            
            ##########################################
            # Determine if stopping condition is met # 
            ##########################################
            
            # E + I + R2
            if( (Exposed + V + np.sum(carcasses) + anycull + anydispose) == 0 ):
                outbreak_go = False
        
        if detailed:
            # Append the final status
            output.append(status)
            canimals.append(cattle)
            sanimals.append(sheep)
            allcarcasses.append(carcasses)
            alltransmiss.append(copy.copy(Transmiss))
        
        # Save the output
        Culled = (status == -1)
        Imm = (who_infected_me == -3)
        Sus = (status == 0)
        
        if True:
            # Trial specific parameters
            results['trial_num'].append(trial)
            results['control_type'].append(control_type)
            results['radius'].append(radius)
            
            # Duration
            results['duration'].append(t)
            
            # Final total infected farms
            results['nip'].append(np.sum(IP) + numseeds)
            
            # Final total dangerous contacts
            results['ndc'].append(np.sum(DC))
            
            # Final total ring culls
            results['nrc'].append(np.sum(RC))
            
            # Final total culled farms/cattle/sheep
            results['nculled_f'].append(np.sum(Culled))
            nculled_c = np.sum(starting_cattle[Culled])
            nculled_s = np.sum(starting_sheep[Culled])
            total_culls = nculled_c + nculled_s
            results['nculled_c'].append(nculled_c)
            results['nculled_s'].append(nculled_s)
            results['nculled_tot'].append(total_culls)
            
            # Final total vaccinated farms/cattle/sheep
            results['nvacc_f'].append(np.sum(Imm))
            nvacc_c = np.sum(starting_cattle[Imm])
            nvacc_s = np.sum(starting_sheep[Imm])
            total_immune = nvacc_c + nvacc_s
            results['nvacc_c'].append(nvacc_c)
            results['nvacc_s'].append(nvacc_s)
            results['nvacc_tot'].append(total_immune)
            
            # Final total susceptible farms/sheep/cattle 
            nsus_f = np.sum(Sus)
            nsus_c = np.sum(starting_cattle[Sus])
            nsus_s = np.sum(starting_sheep[Sus])
            results['nsus_f'].append(nsus_f)
            results['nsus_c'].append(nsus_c)
            results['nsus_s'].append(nsus_s)
            
            # Total overall farms/animals
            results['ntot_f'].append(N)
            results['ntot_c'].append(np.sum(starting_cattle))
            results['ntot_s'].append(np.sum(starting_sheep))
        
        if plotting:
            plt.close()
        
    if verbose:
        print("\tDuration: ", t)
    
    
    # Convert the per-outbreak information into a dataframe
    results = pd.DataFrame(results)
    
    # Return per-timestep information if it's asked for
    if detailed:
        results = [output, canimals, sanimals, allcarcasses, alltransmiss]
    
    return(results)


def Iterate(status, KDIST, Suscept, Transmiss, delay_latency, spark = 0, **kwargs):
    """
    Function to evolve the infection through time
    
    Parameters
    ----------
      - status : np.array
          infection status of each premises (days since infection event)
      - KDIST : np.array (2D)
          distance matrix evaluated according to the spatial kernel
      - Suscept : np.array
          array of susceptibility of each premises
      - Transmiss : np.array
          array of transmissibility of each premises
      - delay_latency : np.array
          latency of the disease in days (no. of days that premises are latently
          infected after infection event)
      - **kwargs 
          further keyword arguments
    
    Returns
    -------
      - status: np.array
          updated state vector; infection status of each premises
    """
    
    # Create a vector of events that occur this time step
    Event = np.zeros(len(status), dtype = int)
    
    #EXP = np.where((status > 0) & (status <= delay_latency))[0]
    INF = np.where((status >= delay_latency))[0]# & (status < delay_cull))
    SUS = np.where(status == 0)[0]
    
    #nEXP = len(INF)
    nINF = len(INF)
    nSUS = len(SUS)
    
    if ((nINF > 0) & (nSUS > 0)):
        # Loop through all infectious farms.  
        
        # Evaluate the kernel function at the specified distances
        
        K = KDIST[np.ix_(INF, SUS)]
        
        # Find the probability of each susceptible farm becoming infected from
        # each infected farm
        #TRANS = np.tile(Transmiss[INF].reshape(nINF,1), (1, nSUS))
        #SUSCEPT = np.tile(Suscept[SUS], (nINF, 1))
        #P = 1 - np.exp(-TRANS * SUSCEPT * K)
        
        # Use numpy broadcasting for this calculation (create dummy dims)
        TRANSSUSCEPT = Transmiss[INF][:, None] * Suscept[SUS][None, :]
        P = 1 - np.exp(-TRANSSUSCEPT * K + spark)
        
        # Generate random numbers for the infection events
        INFECTION_EVENT = (P > np.random.rand(nINF, nSUS))
        
        # Find the index of susceptible farms which become infected
        INFECTION_ROWS = SUS[(np.sum(INFECTION_EVENT, axis = 0) > 0)]
        
        # Update the event vector
        Event[INFECTION_ROWS] = 1
    
    status[status > 0] += 1
    
    # Update the status vector
    status = status + Event
    
    return(status, [], [])


def IterateWho(status, KDIST, Suscept, Transmiss, delay_latency, spark = 0.0, **kwargs):
    """
    Function to evolve the infection through time, recording whom infected who
    
    Parameters
    ----------
      - status : np.array
          infection status of each premises (days since infection event)
      - KDIST : np.array (2D)
          distance matrix evaluated according to the spatial kernel
      - Suscept : np.array
          array of susceptibility of each premises
      - Transmiss : np.array
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
    
    # Create a vector of events that occur this time step
    Event = np.zeros(len(status), dtype = int)
    
    # Create an empty array of who infected who
    INFECTORS = NEW_INFECTIONS = np.array([])
    
    #EXP = np.where((status > 0) & (status <= delay_latency))[0]
    INF = np.where((status >= delay_latency))[0]# & (status < delay_cull))
    SUS = np.where(status == 0)[0]
    
    nINF = len(INF)
    nSUS = len(SUS)
    
    if ((nINF > 0) & (nSUS > 0)):
        # Loop through all infectious farms.  
        
        # Evaluate the kernel function at the specified distances
        
        #K = KDIST[np.ix_(INF, SUS)]
        #TRANS = np.tile(Transmiss[INF].reshape(nINF,1), (1, nSUS))
        #SUSCEPT = np.tile(Suscept[SUS], (nINF, 1))
        
        # Use numpy broadcasting for this calculation (create dummy dims)
        TSK = Transmiss[INF][:, None] * Suscept[SUS][None, :] * KDIST[np.ix_(INF, SUS)]
        
        #TRANS <- kronecker(matrix(1, 1, nSUS),Transmiss[INF])
        #SUSCEPT <- kronecker(matrix(1, nINF, 1), t(Suscept[SUS]))
        
        # Find the probability of each susceptible farm becoming infected from
        # each infected farm
        #P = 1 - np.exp(-TRANS * SUSCEPT * K + spark)
        P = 1 - np.exp(-TSK + spark)
        
        # Generate random numbers for the infection events
        INFECTION_EVENT = (P > np.random.rand(nINF, nSUS))
        
        #########################
        # FINDING THE INFECTORS #
        #########################
        #INFECTION_EVENT.any(axis = 0)
        
        # Find which susceptibles became infected
        NEW_INFECTIONS = np.sum(INFECTION_EVENT, axis = 0)
        
        if NEW_INFECTIONS.any():
            # Find which premises actually infected the newly infected premises
            
            # Subset matrix of all possible infections to show just 
            # new infections
            NN = INFECTION_EVENT[:,NEW_INFECTIONS.astype(bool)]
            
            # Loop through all the infection events (finding the infector)
            INFECTORS = [np.where(n)[0] for n in NN.T]
            
            # Choose one infector randomly (in cases where multiple IPs
            # could have infected a susceptible premises)
            INFECTORS = [INF[np.random.choice(n)] for n in INFECTORS]
            
            # Expand back to the size of the array
            
            # Find the index of susceptible farms which become infected
            NEW_INFECTIONS = SUS[(np.sum(INFECTION_EVENT, axis = 0) > 0)]
            
            # Update the event vector
            Event[NEW_INFECTIONS] = 1
        else:
            # There may have been the possibility of infections but if they didnt' eventuate 
            # then this array should be empty.  
            NEW_INFECTIONS = np.array([])
    
    status[status > 0] += 1
    
    # Update the status vector
    status = status + Event
    
    # Output 
    return(status, NEW_INFECTIONS, INFECTORS)


def IterateGrid(status, grid, period_latent, Transmiss, Suscept, Num, \
    MaxRate, first_, last_, kern, coords):
    """
    Infection process for spatial array of farms.  
    (see am4fmd.utils.infection_process)
    
    Numpy version of the Cython code.
    
    
    Notes:
        It seems to be the MaxRate and Max_Sus_grid that's slightly
        inconsistent with the Matlab version of this code.  
    
    """
    # Copy the status vetor
    nstatus = copy.copy(status)
    
    # Pre-allocate the resultant vector
    Event = np.zeros(nstatus.shape[0], dtype = int)
    
    # Find the index of infectious farms
    INF = np.where( (nstatus > (1 + period_latent) ))[0]
    #print(INF)
    # Find grid cells of infected farm locations
    IGrids = grid[INF]
    
    # Find the number of infected farms
    NI = INF.shape[0]
    
    # Use broadcasting on this ...     
    #print("----------------------------------------")
    
    MaxProbAll = 1 - np.exp(-Transmiss[INF,None] * Num[None,:] * MaxRate[IGrids,:])
    #rng = np.random.rand(*MaxProbAll.shape)
    #mm = np.where( MaxProbAll - rng > 0 )
    
    ########################################
    # SUSCEPTIBLE TO INFECTIOUS TRANSITION #
    ########################################
    # Loop through all infectious farms
    for ii in range(NI):
        
        # Find the infectious farm's index
        INFi = [INF[ii]]
        
        # Transmissibility of the infectious farm 
        # multiplied by the number of animals in that farm
        trans = np.multiply(-Transmiss[INFi], Num)
        
        maxr = MaxRate[IGrids[ii],:]
        
        # Elementwise multiplication
        rate = np.multiply(trans, maxr) # issues with mult by np.inf here
        
        MaxProb = 1 - np.exp(rate)
        
        #print("MP", MaxProb)
        #print("MaxProbAll", MaxProbAll)
        
        # These are the grids that need further consideration
        rng = np.random.rand(len(MaxProb))
        m = np.where( MaxProb - rng > 0)[0]
        #print(m)
        
        #m = mm[ii]
        for n in range(len(m)):
            s = 1
            
            # Loop through grids where an infection event may have occurred.  
            M = m[n]
            
            #PAB = 1 - np.exp(np.multiply(-Transmiss[INFi], MaxRate[IGrids[ii],M]))
            PAB = 1 - np.exp(-Transmiss[INFi]*MaxRate[IGrids[ii],M])
            
            # For comparison it might be of interest to save all the values
            # all_PAB[ii, n] = PAB
            
            # If the rate is infinite, then PAB is 1.  
            if PAB == 1:
                # Calculate the infection probability for each farm in the 
                # susceptible grid square in question
                ind = np.arange(start = first_[M], \
                    stop = last_[M]+1, \
                    dtype = int)
                #ind = range(first_[M], last_[M]+1)
                #print("INFi", INFi)
                #print("ind", ind)
                K = kern(sp.distance.cdist(coords[INFi], coords[ind], 'sqeuclidean')).flatten()
                #print("K", K)
                #print("K.shape", K.shape)
                #print("Transmiss[INFi].shape", Transmiss[INFi].shape)
                #print("Suscept[ind].shape", Suscept[ind].shape)
                Q = 1 - np.exp(-Transmiss[INFi] * Suscept[ind] * K)
                
                rng1 = np.random.rand(len(Q))
                
                test = (rng1 < Q) & (nstatus[ind]==0)
                #print("ind", ind)
                #print("test", test.shape)
                #print("ind[test]", ind[test])
                Event[ind[test]] = 1
            else:
                # Loop through all susceptible farms in the grids where an
                # infection event occurred.  
                R = np.random.rand(Num[M])
                for j in range(int(Num[M])):
                    ind = [first_[M] + j] #- 1
                    P = 1 - s*(1 - PAB)**(Num[M] - j)
                    
                    K = kern(sp.distance.cdist(coords[INFi], coords[ind], 'sqeuclidean'))
                    if R[j] < (PAB / P):
                        s = 0
                        # Need to use matrix multiplication here... 
                        Q = 1 - np.exp(-Transmiss[INFi]*Suscept[ind]*K)#K[INFi, ind])
                        
                        if (R[j] < Q/P) & (nstatus[ind] == 0):
                            Event[ind] = 1
    
    # Evolve the infection process of those farms which are already exposed
    nstatus[(nstatus > 0)] += 1
    
    # Evolve the infection process of those farms which have just been exposed
    nstatus = np.add(nstatus, Event)
    
    NEW_INFECTIONS = np.zeros(0); INFECTORS = np.zeros(0)
    
    return(nstatus, NEW_INFECTIONS, INFECTORS)


def Coords2Grid(x, y, grid_n_x, grid_n_y):
    """
    Determine grid locations of farms from x, y coordinates (row-major order)
    Grid locations start from smallest-x, largest-y values
    
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
        
        perim_x: float
            width of landscape in x-dimension (default: np.ptp(x))
        
        perim_y: float
            width of landscape in y-dimension (default: np.ptp(y))
        
    Returns
    -------
        grid_x : np.array
            x-location of the grid that each farm is in
            
        grid_y : np.array
            y-location of the grid that each farm is in
            
        grid : np.array
            row-major grid number of farm
    
    
    Example
    -------
    from matplotlib import pyplot as plt
    import numpy as np
    
    # Generate a 4 x 4 grid over a landscape of 10 farms
    N = 100; dim = 20.
    xx = np.random.rand(N)*dim; yy = np.random.rand(N)*dim
    gx, gy, grid = Coords2Grid(xx, yy, 4, 4)
    
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
    
    return(grid, xbins, ybins)


def calc_grid_probs(grid, grid_n_x, grid_n_y, Suscept, kernel_function,\
    d_perimeter_x, d_perimeter_y):
    """
    Calculate the probability of infection from one grid square to the next.
    
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
    fpg = [np.where(grid == index)[0] for index in range(NG)]
    
    # Perhaps be made quicker by only looking at the grids with farms in them.
    
    # Maximum susceptibility of a feedlot per grid square (0 for empty grid)
    max_sus = np.asarray([np.max(Suscept[x]) if len(x) > 0 else 0 for x in fpg])
    # Previous code (slower)
    #max_sus = [np.max(np.append(Suscept[fpg[i]], 0)) for i in range(NG)]
    
    # Find the number of farms per grid square
    Num = np.array([len(x) for x in fpg])
    
    # First farm in grid, -1 if the list is empty
    first_ = np.asarray([x[0] if len(x) > 0 else -1 for x in fpg])
    # This was the previous code (a lot slower): 
    #first_ = map(lambda i: np.nanmin(np.append(i, np.nan)), fpg)
    #first_ = np.asarray(first_)
    #first_[np.isnan(first_)] = -1
    #first_ = first_.astype(int)
    
    # Last farm in grid, -2 if the list is empty
    last_ = np.asarray([x[-1] if len(x) > 0 else -2 for x in fpg])
    # Previous code, much slower
    #last_ = map(lambda i: np.max(np.append(i, -2)), fpg)
    #last_ = np.asarray(last_).astype(int)
    
    NCOL = float(grid_n_x); NROW = float(grid_n_y)
    HEIGHT = float(d_perimeter_y)/NROW; WIDTH = float(d_perimeter_x)/NCOL
    
    # Calculate the minimum sq-distance between each grid square
    # X and Y position of each grid square
    Y, X = np.ogrid[0:NROW*NCOL, 0:NROW*NCOL]
    HDIST = np.abs(Y%NCOL - X%NCOL) - 1
    VDIST = np.abs(Y/NCOL - X/NCOL) - 1
    HDIST[HDIST < 0] = 0; VDIST[VDIST < 0] = 0
    Dist2all = np.add((HDIST*WIDTH)**2, (VDIST*HEIGHT)**2)
    
    KDist2all = kernel_function(Dist2all) # very slow
    
    # Can use Numpy broadcasting for this ... 
    MaxRate = np.multiply(KDist2all, np.tile(max_sus, (grid_n_x*grid_n_y, 1)))
    
    # All grids with no farms should have infinite rate of input infection
    # All grids with no farms should have infinite rate of susceptibility
    MaxRate[Num == 0] = np.inf
    MaxRate[:, Num == 0] = np.inf
    
    # Set rate from one grid to itself to infinity.  
    np.fill_diagonal(MaxRate, np.inf)
    
    return(MaxRate, Num, first_, last_, max_sus, Dist2all, KDist2all)



def ExpKernel(x, g = 4.8, h = 2.4):
    """
    Exponential kernel function
    
    Parameters
    ----------
      - x : float, np.array
          squared distance (km) at which to evaluate the kernel
      - g, h: float, float
          parameters of the exponential kernel
    
    Returns
    -------
      - scalar or np.array of kernel function evaluated at x
    """
    return(g*np.exp(-h*x))


def PowerKernel(x, d = 0.41):
    """
    Power law kernel function
    
    Parameters
    ----------
      - x : float, np.array
          squared distance (km) at which to evaluate the kernel
      - d: float
          parameters of the power-law kernel
    
    Returns
    -------
      - scalar or np.array of kernel function evaluated at x
    """
    return(d/x**2)


def UKKernel(x):
    """
    Spread kernel from Keeling and Rohani (2008)
    
    Parameters
    ----------
      - x : float, np.array
          squared distance (km) at which to evaluate the kernel
    
    Returns
    -------
      - scalar or np.array of kernel function evaluated at x
    """
    
    P = [-9.2123e-5, 9.5628e-4, 3.3966e-3, -3.3687e-2, 
        -1.30519e-1, -0.609262, -3.231772]
    
    K = P[0]*x**6 + P[1]*x**5 + P[2]*x**4 + P[3]*x**3 + P[4]*x**2+P[5]*x+P[6]
    K = np.exp(K)
    
    lower_lim = (x < 0.0138)
    upper_lim = (x > 60*60)
    
    K[lower_lim] = 0.3093
    K[upper_lim] = 0.0
    return(K)
