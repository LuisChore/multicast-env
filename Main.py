from Graph import Graph
from BRD import Environment
from Utils import *
import numpy as np

#generate_12_8b("Examples/12.8b",0.5,5)
#generate_12_10("Examples/12.10_paths",0.5,5,paths = True)

def print_paths(env):
    for ag in env.agents:
        print(ag.index,end = ":")
        print(ag.path)

def min_dic(Q,state):
    if state in Q:
        min_value = float('inf')
        min_key = None
        for a,v in Q[state].items():
            if v < min_value:
                min_key = a
                min_value = v
        return min_key,min_value
    return np.random.randint(3 * n_edges),0 # again, return 0 if there is no options


def eps_greedy(action):
    p = np.random.random()
    if p < (1-p):
        return action
    return np.random.randint(3 * n_edges)


#everyone starts in 0
def q_value(Q,s,a):
    if s in Q:
        if a in Q[s]:
            return Q[s][a]
        else:
            Q[s][a] = 0
            return Q[s][a]
    else:
        Q[s] = {}
        Q[s][a] = 0
        return Q[s][a]

if __name__ == '__main__':
    agents, source, g = set_graph("Examples/12.8a",paths = True)
    brd = Environment(g,agents,source,step_cost = 100)

    n_edges = len(brd.edge_list)
    GAMMA = 0.9
    ALPHA = 0.1

    Q = {}
    epochs = 1000

    for e in range(epochs):
        s,done = brd.reset()
        a,_ =  min_dic(Q,s)
        while not done:
            a = eps_greedy(a)
            s2,r,done = brd.step(a)
            old_q = q_value(Q,s,a)
            a2,minQ = min_dic(Q,s2)
            TD = r + GAMMA * minQ - old_q # Q-Learning approach
            Q[s][a] = old_q + ALPHA * TD


    #FOLLOWING THE BEST POLICY
    s,done = brd.reset()
    print(s)
    print(done)
    print_paths(brd)
    while not done:
        a,_ = min_dic(Q,s)
        s,r,done = brd.step(a)
        print_paths(brd)


    #brd.plot_graph()
    #brd()
