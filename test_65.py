import numpy as np
from context_matters import core, kernels, fmd
from am4fmd.data.load_data import circ3km

np.random.seed(20)
data = circ3km

test = 65

period_silent_spread = 12
# Delay (days) from vax to conferment of immunity
ci = 7
vc = 10000
cc = 12000

params = dict({\
    'rate_cull': 200, \
    'rate_dispose': 200, \
    'rate_vacc': 200, \
    'period_confer_immune': ci, \
    'carcass_constraint': cc, \
    'vacc_constraint':vc})

# Make a few more cases infectious (10 seed cases)
data.loc[data.status > 0,'status'] = 0
latentcases = [1057, 3886, 3741, 51]
data.loc[latentcases, 'status'] = 1

# Create the seed state, landscape, and outbreak objects
seed_state = core.State(data.status, \
    n_cattle = np.round(data.n_cattle), \
    n_sheep = data.n_sheep, \
    ci = ci, x = data.x, y = data.y)

land = core.Landscape(list(zip(data.x, data.y)))

actions = [core.Cull(0.0, 3.0), core.Vacc(0, 3.0)]

# Generate outbreak objects
#kernel = kernels.UKKernel(scaling_factor = 0.02)
kernel = kernels.ExponentialKernel(g = 0.00626501, h = 1.43359443)

outbreak_dur = fmd.Outbreak_cc_vc(\
    Landscape = land, \
    start = seed_state, \
    grid_shape = (50, 50), \
    objective = 'duration', \
    cfilter_fn = fmd.choose_cull_candidates2, \
    vfilter_fn = fmd.choose_vacc_candidates, \
    period_silent_spread = period_silent_spread, \
    kernel = kernel, \
    **params)

# Generate an agent object
mcagent_dur = fmd.MCAgent_new(\
    actions = actions, \
    epsilon = 0.4, \
    starting_action = core.Cull(0.0, 0.0))

# Only make one decision
mcagent_dur.control_switch_times = [outbreak_dur.period_silent_spread + 7]

save_times = [outbreak_dur.period_silent_spread] + mcagent_dur.control_switch_times

sim_dur = fmd.FMDSim(mcagent_dur, outbreak_dur, max_time = np.inf, \
    full = False, printing = False, time_save_state = save_times, \
    trials_save_state = range(2000), save_dir = "./output/", save_name = "test65")

sim_dur.verbose = True

import time
start = time.time()
N = 100000; sim_dur.trials(N)
end = time.time()
print(end - start)

np.save('./output/test_' + str(test) + '_sim_dur_visits.npy', sim_dur.agt.visits)
np.save('./output/test_' + str(test) + '_sim_dur_Q.npy', sim_dur.agt.Q)
np.save('./output/test_' + str(test) + '_sim_dur_actions.npy', sim_dur.agt.actions)
np.save('./output/test_' + str(test) + '_sim_dur_returns.npy', sim_dur.agt.returns)

# Reload using the following ... etc
#sim_dur_visits = np.load('sim_dur_visits.npy').item()


