#!/usr/bin/env python3
"""
Plotting functions for the context_matters project

W. Probert
"""

import pandas as pd, numpy as np, numpy.ma as ma, time, matplotlib
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
import matplotlib.colors as mcol
import matplotlib.cm as cm
import matplotlib.patches as p


def plot2dpolicy(data, x, y, astar, color_dict, x_lab = "", y_lab = "", a_labels = None, \
    visit_thresh = None, title = "Optimal policy", plot_legend = True, ledge_loc = 0, \
    xlims = None, ylims = None):
    """
    Plot a policy plot that has a two-dimensional state-space
    
    
    Arguments
    ---------
    
    
    """
    # Subset the dataframe if there's a limit on the number of visits 
    # that a state has to have had in order to be plotted
    if visit_thresh is not None:
        data = data.loc[data.visits > visit_thresh]
    
    # Sort unique X and Y values
    X = np.sort(data[x].unique())
    Y = np.sort(data[y].unique())
    
    X = np.append(X, X[-1]+np.abs(X[-1]) - X[-2])
    Y = np.append(Y, Y[-1]+np.abs(Y[-1]) - Y[-2])
    
    # This fails if one action is best across whole state-space (best to use alternative line below)
    #acts = np.asarray(np.sort(data[astar].unique())) 
    acts = np.asarray(np.sort(list(color_dict.keys())))
    
    a, b = np.meshgrid(X, Y)
    b = np.flipud(b)
    
    C = data.pivot_table(values = astar, index = y, columns = x)
    C = np.asarray(C)
    C = np.flipud(C)
    
    customcm = mcol.ListedColormap(color_dict.values())
    boundaries = np.linspace(0, len(acts), len(acts) + 1)
    cnorm = mcol.BoundaryNorm(boundaries, customcm.N)
    
    Cm = ma.masked_where(np.isnan(C), C)
    
    # Create a figure and axis object
    fig, ax = plt.subplots()
    
    # Add the policy heatmap using pcolormesh()
    ax.pcolormesh(a, b, Cm, cmap = customcm, norm = cnorm)
    
    # Adjust x and y limits
    if xlims:
        ax.set_xlim(xlims)
    else:
        ax.set_xlim([np.nanmin(X), np.nanmax(X)])
    
    if ylims:
        ax.set_ylim(ylims)
    else:
        ax.set_ylim([np.nanmin(Y), np.nanmax(Y)])
    
    # Adjust tick marks and frame
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_frame_on(False)
    
    # Draw own axes
    ax.axvline(ax.get_xlim()[0], color = '#2b2b2b', linewidth = 2.0)
    ax.axhline(ax.get_ylim()[0], color = '#2b2b2b', linewidth = 2.0)
    
    # Set labels and title
    ax.set_xlabel(x_lab, fontsize = 16); ax.set_ylabel(y_lab, fontsize = 16)
    ax.set_title(title, fontsize = 18)
    
    # Plot a legend (if necessary)
    if plot_legend:
        
        ledge = []
        for key, value in color_dict.items():
            ledge.append(Line2D([0], [0], linestyle = "none", marker = "s", \
                markersize = 20, markerfacecolor = value))
        
        # Add action labels if none are defined
        if a_labels is None:
            a_labels = range(len(color_dict.keys()))
        
        # Add the legend object
        ax.legend(tuple(ledge), tuple(a_labels), \
            numpoints = 1, loc = ledge_loc, \
            prop = {'size':14}, frameon = True)
    
    # Return the figure and axis object
    return fig, ax


def plot2dvaluefn(data, x, y, value, x_lab = "", y_lab = "", visit_thresh = 10, \
    title = "Optimal value function"):
    """
    Plot the optimal value function for a system that has a two-dimensional state-space
    """
    
    data = data.loc[data.visits > visit_thresh]
    
    X = np.sort(data[x].unique())
    Y = np.sort(data[y].unique())
    
    X = np.append(X, X[-1]+np.abs(X[-1]) - X[-2])
    Y = np.append(Y, Y[-1]+np.abs(Y[-1]) - Y[-2])
    
    cmap = ["#4575b4", "#ffffbf", "#d73027"]
    cm_new = mcol.LinearSegmentedColormap.from_list("custom", cmap)
    cnorm = mcol.Normalize(vmin = np.min(data[value]), \
        vmax = np.max(data[value]))
    cpick = cm.ScalarMappable(norm = cnorm, cmap = cm_new)
    
    fig, ax = plt.subplots()
    
    a, b = np.meshgrid(X, Y)
    b = np.flipud(b)
    
    C = data.pivot_table(values=value, index = y, columns = x)
    C = np.asarray(C)
    C = np.flipud(C)
    
    Cm = ma.masked_where(np.isnan(C),C)
    ax.pcolormesh(a, b, Cm, cmap = cm_new, norm = cnorm)
    
    x_labs = np.array(np.percentile(range(len(X)), [0,25,50,75,100]))
    x_labs = x_labs.astype(int)
    y_labs = np.array(np.percentile(range(len(Y)), [0,25,50,75,100]))
    y_labs = y_labs.astype(int)
    
    ax.set_xlim([np.nanmin(X), np.nanmax(X)])
    ax.set_ylim([np.nanmin(Y), np.nanmax(Y)])
    
    ax.set_xticks(X[x_labs])
    ax.set_xticklabels(np.round(X[x_labs], decimals = 3))
    ax.set_yticks(Y[y_labs])
    ax.set_yticklabels(np.round(Y[y_labs], decimals = 3))
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_frame_on(False)
    
    ax.axvline(ax.get_xlim()[0], color = '#2b2b2b', linewidth = 2.0)
    ax.axhline(ax.get_ylim()[0], color = '#2b2b2b', linewidth = 2.0)
    
    ax.set_xlabel(x_lab, fontsize = 16)
    ax.set_ylabel(y_lab, fontsize = 16)
    
    # Add colormap
    ax1 = fig.add_axes([0.925, 0.25, 0.025, 0.5])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap = cm_new, norm = cnorm)
    
    ax.set_title(title, fontsize = 18)
    
    return fig, ax



def plot2dvaluefn_difference(data, x, y, value_a0, value_a1, x_lab = "", y_lab = "", \
    visit_thresh = 10, title = "Optimal value function", xlims = None, ylims = None, \
    subplots_adjust = None):
    """
    Plot the difference in value function between two actions in a system that has a
    two-dimensional state-space.  
    """
    
    data = data.loc[data.visits > visit_thresh]
    
    X = np.sort(data[x].unique())
    Y = np.sort(data[y].unique())
    
    X = np.append(X, X[-1]+np.abs(X[-1]) - X[-2])
    Y = np.append(Y, Y[-1]+np.abs(Y[-1]) - Y[-2])
    
    data['difference'] = data[value_a0] - data[value_a1]
    
    maximum = np.max([np.abs(np.max(data["difference"])), np.abs(np.max(data["difference"]))])
    
    cmap = ["#4575b4", "#ffffbf", "#d73027"]
    cm_new = mcol.LinearSegmentedColormap.from_list("custom", cmap)
    cnorm = mcol.Normalize(
        vmin = -maximum, \
        vmax = maximum)
    cpick = cm.ScalarMappable(norm = cnorm, cmap = cm_new)
    
    fig, ax = plt.subplots()
    
    a, b = np.meshgrid(X, Y)
    b = np.flipud(b)
    
    C = data.pivot_table(values = 'difference', index = y, columns = x)
    C = np.asarray(C)
    C = np.flipud(C)
    
    Cm = ma.masked_where(np.isnan(C),C)
    ax.pcolormesh(a, b, Cm, cmap = cm_new, norm = cnorm)
    
    # x_labs = np.array(np.percentile(range(len(X)), [0,25,50,75,100]))
    # x_labs = x_labs.astype(int)
    # y_labs = np.array(np.percentile(range(len(Y)), [0,25,50,75,100]))
    # y_labs = y_labs.astype(int)
    #
    # ax.set_xticks(X[x_labs])
    # ax.set_xticklabels(np.round(X[x_labs], decimals = 3))
    # ax.set_yticks(Y[y_labs])
    # ax.set_yticklabels(np.round(Y[y_labs], decimals = 3))
    
    if xlims:
        ax.set_xlim(xlims)
    else:
        ax.set_xlim([np.nanmin(X), np.nanmax(X)])
    
    if ylims:
        ax.set_ylim(ylims)
    else:
        ax.set_ylim([np.nanmin(Y), np.nanmax(Y)])
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_frame_on(False)
    
    ax.axvline(ax.get_xlim()[0], color = '#2b2b2b', linewidth = 2.0)
    ax.axhline(ax.get_ylim()[0], color = '#2b2b2b', linewidth = 2.0)
    
    ax.set_xlabel(x_lab, fontsize = 16)
    ax.set_ylabel(y_lab, fontsize = 16)
    
    # Add colormap
    ax1 = fig.add_axes([0.9, 0.25, 0.025, 0.5])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap = cm_new, norm = cnorm)
    
    ax.set_title(title, fontsize = 18)
    
    if subplots_adjust:
        plt.subplots_adjust(left = subplots_adjust[0], \
            bottom = subplots_adjust[1], \
            top = subplots_adjust[2], \
            right = subplots_adjust[3])
    
    return fig, ax


def plot2dvisits(data, x, y, value, x_lab, y_lab, log = False, visit_thresh = 10, \
    visit_upper_thresh = np.inf, title = "Number of visits", xlims = None, ylims = None, \
    subplots_adjust = None):
    """
    Plot visits to each state as a heatmap for a system that has a two-dimensional state-space
    
    Parameters
    ----------
    data : pandas DataFrame
        
    x : str
    y : str
    value : str
    """
    
    data = data.loc[data.visits > visit_thresh]
    data = data.loc[data.visits < visit_upper_thresh]
    
    X = np.sort(data[x].unique())
    Y = np.sort(data[y].unique())
    
    X = np.append(X, X[-1]+np.abs(X[-1]) - X[-2])
    Y = np.append(Y, Y[-1]+np.abs(Y[-1]) - Y[-2])
    
    cmap = ["#4575b4", "#ffffbf", "#d73027"]
    cm_new = mcol.LinearSegmentedColormap.from_list("custom", cmap)
    
    if log:
        cnorm = mcol.Normalize(vmin = np.log10(np.min(data[value])), \
            vmax = np.log10(np.max(data[value])))
    else:
        cnorm = mcol.Normalize(vmin = np.min(data[value]), \
            vmax = np.max(data[value]))
    
    cpick = cm.ScalarMappable(norm = cnorm, cmap = cm_new)
    
    fig, ax = plt.subplots()
    
    a, b = np.meshgrid(X, Y)
    b = np.flipud(b)
    
    C = data.pivot_table(values=value, index = y, columns = x)
    C = np.asarray(C)
    
    if log:
        C = np.log10(np.flipud(C))
    else:
        C = np.flipud(C)
    
    Cm = ma.masked_where(np.isnan(C),C)
    ax.pcolormesh(a, b, Cm, cmap = cm_new, norm = cnorm)
    
    if xlims is None:
        xlims = [np.nanmin(X), np.nanmax(X)]
    
    if ylims is None:
        ylims = [np.nanmin(Y), np.nanmax(Y)]
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    # x_labs = np.array(np.percentile(range(len(X)), [0,25,50,75,100]))
    # x_labs = x_labs.astype(int)
    # y_labs = np.array(np.percentile(range(len(Y)), [0,25,50,75,100]))
    # y_labs = y_labs.astype(int)
    
    # ax.set_xticks(X[x_labs])
    # ax.set_xticklabels(np.round(X[x_labs], decimals = 3))
    # ax.set_yticks(Y[y_labs])
    # ax.set_yticklabels(np.round(Y[y_labs], decimals = 3))
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_frame_on(False)
    
    ax.axvline(ax.get_xlim()[0], color = '#2b2b2b', linewidth = 2.0)
    ax.axhline(ax.get_ylim()[0], color = '#2b2b2b', linewidth = 2.0)
    
    ax.set_xlabel(x_lab, fontsize = 16)
    ax.set_ylabel(y_lab, fontsize = 16)
    
    # Add colormap
    ax1 = fig.add_axes([0.875, 0.25, 0.025, 0.5])
    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cm_new, norm=cnorm)
    #cb1.set_yticklabels(...)
    
    ax.set_title(title, fontsize = 18)
    
    if subplots_adjust:
        plt.subplots_adjust(left = subplots_adjust[0], \
            bottom = subplots_adjust[1], \
            top = subplots_adjust[2], \
            right = subplots_adjust[3])
    
    return fig, ax



def animateoutbreak_quick(Sim, trial = 0, saver = False, rest_time = 0.01, msize = 1):
    """
    Animate an outbreak given a Simulation object.  
    
    This might be slow as it redraws the whole figure at each ts.      
    This function can also save files to a png and stitch them together for an avi movie.  
    """
    
    NUM_FARMS = Sim.env.L.nfarms
    
    # Set all colours of interest
    
    # Exposed
    orange = (255./255, 127./255, 0./255, 1)
    # Infected
    red = (228./255, 26./255, 28./255, 1)
    # Removed or immune
    grey_tint = 200.
    grey = (grey_tint/255, grey_tint/255, grey_tint/255, 1)
    
    # Being culled
    #(55./255, 126./255, 184./255, 1.) (more grey-blue)
    blue = (18./255, 15/255, 202./255, 1.)
    # Being disposed of
    purple = (152./255, 78./255, 163./255, 1.)
    # Being vaccinated
    cyan = (255./255, 255./255, 51./255, 1.)
    
    sus_col = (65./255, 173./255, 0./255, 1)
    #(44./255, 162./255, 95./255, 1) (more green-grey)
    
    #(40./255, 255./255, 40./255, 1)
    face_colors = [sus_col for ii in range(NUM_FARMS)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    
    # Returns a tuple of line objects, thus the comma 
    scatter_plot = ax.scatter(*zip(*Sim.env.L.coords), s = msize, linewidths = 2)
    
    # Set edge and face colors the same
    scatter_plot.set_facecolor(face_colors)
    scatter_plot.set_edgecolor(face_colors)
    
    x = np.array(Sim.env.L.coords).transpose()[0]
    y = np.array(Sim.env.L.coords).transpose()[1]
    
    # Plot the current action
    action = ax.text(0.27, 0.95, "\n" + str(Sim.outlist[trial][0][1]), \
        ha = "right", va = "top", transform = ax.transAxes)
    
    # Plot the timer
    times = np.arange(len(Sim.outlist[trial]))
    timer = ax.text(0.95, 0.95, "\nDay "+str(times[0])+"\t", \
        ha = "right", va = "top", transform = ax.transAxes)
    
    # Plot a legend
    plt.rc('legend',**{'fontsize': 8})
    
    marker_props = {'marker': '.', 'markersize':15, 'linestyle': 'None', \
        'alpha': 0.7}
    
    state = p.Patch(color=(1,1,1,1), label='State', alpha = 0.0)
    
    ledge_s = Line2D([],[],color=sus_col,label='Susceptible', **marker_props)
    ledge_ex = Line2D([], [], color = orange, label='Exposed', **marker_props)
    ledge_inf = Line2D([], [], color = red, label='Infectious', **marker_props)
    ledge_rem = Line2D([], [], color = grey, label='Removed', **marker_props)
    ledge_immune = p.Patch(color=(1,1,1,1), label='(or immune)', alpha = 0.0)
    
    acts = p.Patch(color=(1,1,1,1), label='Activities', alpha = 0.0)
    ledge_bc = Line2D([], [], color = blue, label = 'Culling', **marker_props)
    ledge_bv = Line2D([], [], color = cyan, label='Vacc\'n', **marker_props)
    ledge_bd = Line2D([], [], color = purple, label='Disposal', **marker_props)
    
    space = p.Patch(color=(1,1,1,1), label='', alpha = 0.0)
    
    all_handles = [state, ledge_s, ledge_ex, ledge_inf, ledge_rem, \
        ledge_immune, space, space, acts, ledge_bc, ledge_bv, ledge_bd]
    
    legend_all = ax.legend(handles = all_handles, \
        loc = 'center right', numpoints = 1, frameon = False, ncol = 1)
    ax.add_artist(legend_all)
    
    # Adjust the axes for the legend
    ax.margins(x = 0.3, y = 0.1)
    
    # Turn all the axes off
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Loop through time and change colour of points accordingly
    for q in times:
        # Update the timer
        timer.set_text("\nDay "+str(q)+"\t")
        
        action.set_text("\n" + str(Sim.outlist[trial][q][1]))
        
        status = Sim.outlist[trial][q][0].status
        status_being_culled = Sim.outlist[trial][q][0].being_culled
        status_being_vaccinated = Sim.outlist[trial][q][0].being_vacc
        status_being_disposed = Sim.outlist[trial][q][0].being_disposed
        
        Exp = ((status > 0) & (status <= (1 + Sim.env.period_latent)))
        Inf = (status > (1 + Sim.env.period_latent))
        Rep = (status > Sim.env.period_latent)
        Removed = (status == -1)
        
        bc = (status_being_culled == 1)
        # Could add this... & (status_being_disposed == 0)
        bv = (status_being_vaccinated == 1)
        
        bd = (status_being_culled == 0) & (status_being_disposed == 1)
        bcd = (status_being_disposed == 1) & (status_being_culled == 1)
        
        # Update the colors
        face_colors = [orange if x else y for x, y in zip(Exp, face_colors)]
        face_colors = [red if x else y for x, y in zip(Inf, face_colors)]
        face_colors = [grey if x else y for x, y in zip(Removed, face_colors)]
        face_colors = [cyan if x else y for x, y in zip(bv, face_colors)]
        face_colors = [purple if x else y for x, y in zip(bd, face_colors)]
        
        # Update edge colours
        edge_colors = face_colors
        edge_colors = [blue if x else y for x, y in zip(bc, edge_colors)]
        edge_colors = [cyan if x else y for x, y in zip(bv, edge_colors)]
        
        scatter_plot.set_facecolor(face_colors)
        scatter_plot.set_edgecolors(edge_colors)
        
        plt.pause(rest_time)
        
        if saver:
            filename = str('%03d' % q) + '.png'
            plt.savefig(filename, dpi=100)
        
    plt.show()
    
    # Stitch all the png files together with mencoder
    # Animation part adapted from code from Josh Lifton, 2004 (see Python old_animation_example)
    # http://matplotlib.org/1.3.0/examples/old_animation/movie_demo.html
    if saver:
        import subprocess
        command = ('mencoder',
                   'mf://*.png',
                   '-mf',
                   'type=png:w=480:h=360:fps=5',
                   '-ovc',
                   'lavc',
                   '-lavcopts',
                   'vcodec=mpeg4',
                   '-oac',
                   'copy',
                   '-o',
                   'output.avi')
        subprocess.check_call(command)


def plot_performance(df, colour_dict, alabels = None, ticks = False, \
    x_lab = "", y_lab = "", title = ""):
    """
    Plot the performance of different actions in a violin plot
    
    Arguments
    ---------
    df : pandas DataFrame
        Data frame with a column for each action with each action showing the performance 
        (in whatever metric is appropriate).  
    colour_dict : dict
        A dictionary mapping action index to face colour for the violin plot
    alabels : list of str
        List of action labels to use along the x-axis for each violin plot
    ticks : bool
        Should ticks be plotted?  
    x_lab, y_lab, title : str
        x-label, y-label, title respectively
    
    Returns
    -------
    The figure and axes objects (as generated by plt.subplots())
    
    Example
    -------
    import pandas as pd, numpy as np
    from matplotlib import pyplot as plt
    from context_matters import plotting as plot
    
    df = pd.DataFrame({"A": np.random.normal(10, 4, 100), "B": np.random.normal(11, 3, 100), \
        "C": np.random.normal(12, 2, 100), "D": np.random.normal(10.5, 5, 100)})
    
    alabels = ["Action A", "Action B", "Action C", "Action D"]
    colours = ["blue", "green", "red", "orange"]
    colour_dict = dict(zip(range(len(alabels)), colours))
    
    fig, ax = plot.plot_performance(df, colour_dict, alabels, ticks = True, \
        x_lab = "Actions", y_lab = "Outbreak duration", title = "Mock plot of action performance")
    plt.show()
    """
    
    fig, ax = plt.subplots()
    
    # Create a list of values for each column in the input data frame
    results = [list(df[c].values) for c in df]
    
    for i, r in enumerate(results):
    
        violins = ax.violinplot(r, positions = [i], showmeans = True, showmedians = False)
    
        for line in ('cbars','cmins','cmaxes','cmeans'):
            v = violins[line]
            v.set_edgecolor(colour_dict[i])
            v.set_linewidth(2)
    
        for v in violins['bodies']:
            v.set_facecolor(colour_dict[i])
            v.set_edgecolor(colour_dict[i])
            v.set_alpha(0.8)
    
    if ticks:
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels(alabels)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
    
    ax.set_xlim([-1, len(results)])
    ax.set_xlabel(x_lab)
    ax.set_ylabel(y_lab)
    ax.set_title(title)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return fig, ax
