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


