from Graph import Graph
from BRD import Environment
from Utils import *
import numpy as np

#generate_12_8b("Examples/12.8b",0.5,5)
#generate_12_10("Examples/12.10_paths",0.5,5,paths = True)
class QLearning():

    def __init__(self,file_name,paths_given = False,step_cost = 1000 ,beta = 5, eps = 0.1):
        self.paths_given = paths_given
        self.eps = eps
        self.agents,self.source,self.g = set_graph(file_name,paths = paths_given)
        self.env = Environment(self.g,self.agents,self.source,
            paths_given = self.paths_given,step_cost = step_cost,beta = beta)
        self.ACTION_SPACE_SIZE = len(self.env.edge_list) * 3

    def print_paths(self):
        for ag in self.env.agents:
            print(ag.index,end = ":")
            print(ag.path)

    def min_dic(self,Q,state):
        if state in Q:
            min_value = float('inf')
            min_key = None
            for a,v in Q[state].items():
                if v < min_value:
                    min_key = a
                    min_value = v
            return min_key,min_value
        return np.random.randint(self.ACTION_SPACE_SIZE),0
        # again, return 0 if there is no options

    def eps_greedy(self,action):
        p = np.random.random()
        if p < (1-self.eps):
            return action
        return np.random.randint(self.ACTION_SPACE_SIZE)

    #everyone starts in 0
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

    def train(self,GAMMA,ALPHA,epochs,test = False):
        Q = {}

        if test == True:
            costs = []
            iterations = []
            differences = []
        for e in range(epochs):
            s,done = self.env.reset()
            a,_ =  self.min_dic(Q,s)

            # test purposes
            cost_per_run = 0
            iterations_per_run = 0
            max_diff = 0

            while not done:
                a = self.eps_greedy(a)
                s2,r,done = self.env.step(a)
                old_q = self.q_value(Q,s,a)
                a2,minQ = self.min_dic(Q,s2)
                TD = r + GAMMA * minQ - old_q # Q-Learning approach
                Q[s][a] = old_q + ALPHA * TD
                if test == True:
                    cost_per_run += r
                    iterations_per_run += 1
                    max_diff = max(max_diff,np.abs(Q[s][a] - old_q))
            if test == True:
                costs.append(cost_per_run)
                iterations.append(iterations_per_run)
                differences.append(max_diff)

        if test == True:
            return Q,costs,iterations,differences
        else:
            return Q

    def solve(self,Q,render = True):
        #FOLLOWING THE BEST POLICY
        s,done = self.env.reset()
        if render == True:
            self.env.render()
        cost = 0
        iterations = 0
        while not done:
            a,_ = self.min_dic(Q,s)
            s,r,done = self.env.step(a)
            if render == True:
                self.env.render()
            cost += r
            iterations += 1
        return cost,iterations
