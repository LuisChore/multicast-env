'''
Q-Learning(QL)   algorithm   designed   to   solve
                             Multicast Environment
'''
from Graph import Graph
from BRD import Environment
from Utils import *
import numpy as np

class QLearning():
    #file_name
    #paths_given - to be able to initialize paths
    #step_cost - hyperparameter for QL algorithm
    #beta -hyperparameter for modify the agent actions
    #eps - hyperparameter for customize e-greddy algorithm
    def __init__(self,file_name,paths_given = False,step_cost = 5 ,beta = 0.9, eps = 0.1):
        self.paths_given = paths_given
        self.eps = eps
        self.agents,self.source,self.g = set_graph(file_name,paths = paths_given)
        self.env = Environment(self.g,self.agents,self.source,
            paths_given = self.paths_given,step_cost = step_cost,beta = beta)
        self.ACTION_SPACE_SIZE = len(self.env.edge_list) * 3

    '''
    It  prints  the current agents paths (for testing)
    '''
    def print_paths(self):
        print("Paths:")
        print("-----------------------------------------------------")
        for ag in self.env.agents:
            print(ag.index,end = ":")
            print(ag.path)
        print("-----------------------------------------------------")


    '''
    it  returns the best possible action ant its value
    given  a  state, if  the states still has not been
    updated  it  returns  a random action with value 0
    '''
    def min_dic(self,Q,state):
        if state in Q:
            min_value = min( Q[state].values())
            min_keys = [key for key,val in Q[state].items() if val == min_value]
            return np.random.choice(min_keys),min_value
        return np.random.randint(self.ACTION_SPACE_SIZE),0

    def eps_greedy(self,action):
        p = np.random.random()
        if p < (1-self.eps):
            return action
        return np.random.randint(self.ACTION_SPACE_SIZE)


    '''
    It returns the Q value for any tuple(state,action)
    if  it  is  not initialized yet, it would have a 0
                                         default value
    '''
    def q_value(self,Q,s,a):
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

    '''
    It  computes  and  returns  the Q values after the
    training,   using   temporal  difference  approach
    '''
    #GAMMA - QL hyperparameter (discount factor)
    #ALPHA - QL hyperparameter (step Learning)
    #epochs - Number of learning epochs
    #test - for testing mode
    def train(self,GAMMA,ALPHA,epochs,test = False):
        Q = {} # Q-values
        if test == True: # test mode
            costs = [] #cost per epoch
            iterations = [] #number of iteration per epoch
            #difference to measure Q-Learning Convergence,it saves the largest
            #difference found in an epoch, theorically it should tends to zero
            differences = []

        for e in range(epochs):
            s,done = self.env.reset() #reset environment
            a,_ =  self.min_dic(Q,s) # initial action (random)

            # testing variables
            cost_per_run = 0
            iterations_per_run = 0
            max_diff = 0

            while not done:
                a = self.eps_greedy(a) # chosing action
                old_q = self.q_value(Q,s,a)  # old Q(s,a) value
                s2,r,done = self.env.step(a) # take an action
                a2,minQ = self.min_dic(Q,s2) # best action for the new state
                TD = r + GAMMA * minQ - old_q # temporal difference
                Q[s][a] = old_q + ALPHA * TD # update Q value

                # testing updates
                if test == True:
                    cost_per_run += r
                    iterations_per_run += 1
                    max_diff = max(max_diff,np.abs(Q[s][a] - old_q))

            # testing updates
            if test == True:
                costs.append(cost_per_run)
                iterations.append(iterations_per_run)
                differences.append(max_diff)

        if test == True:
            return Q,costs,iterations,differences
        else:
            return Q

    '''
    Given  the  Q-values  already trained it solve the
    environment  using the best policy, it returns the
    cost  and  the number of iterations, it contains a
                  render mode to visualize the process
    '''
    def solve(self,Q,render = True,print_policy = False):
        #FOLLOWING THE BEST POLICY
        s,done = self.env.reset()
        if print_policy == True:
            print("Initial State:")
            self.env.print_state()
            self.print_paths()
        if render == True:
            self.env.render()
        cost = 0
        iterations = 0
        while not done:
            a,_ = self.min_dic(Q,s)
            s,r,done = self.env.step(a)
            if print_policy == True:
                self.env.print_state()
                self.print_paths()
            if render == True:
                self.env.render()
            cost += r
            iterations += 1
        return cost,iterations
