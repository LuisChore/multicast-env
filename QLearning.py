from Graph import Graph
from BRD import Environment
from Utils import *
import numpy as np

#generate_12_8b("Examples/12.8b",0.5,5)
#generate_12_10("Examples/12.10_paths",0.5,5,paths = True)
class QLearning():

    def __init__(self,file_name,step_cost = 100,alpha = 0.9):
        self.agents,self.source,self.g = set_graph(file_name,paths = False)
        self.env = Environment(self.g,self.agents,self.source,
            step_cost = step_cost,alpha = alpha)
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
        return np.random.randint(self.ACTION_SPACE_SIZE),0 # again, return 0 if there is no options


    def eps_greedy(self,action):
        p = np.random.random()
        if p < (1-p):
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


    def train(self,GAMMA,ALPHA,epochs):
        Q = {}
        for e in range(epochs):
            s,done = self.env.reset()
            a,_ =  self.min_dic(Q,s)
            while not done:
                a = self.eps_greedy(a)
                s2,r,done = self.env.step(a)
                old_q = self.q_value(Q,s,a)
                a2,minQ = self.min_dic(Q,s2)
                TD = r + GAMMA * minQ - old_q # Q-Learning approach
                Q[s][a] = old_q + ALPHA * TD

        return Q

    def test(self,Q):
        #FOLLOWING THE BEST POLICY
        s,done = self.env.reset()
        print(done)
        self.print_paths()
        self.env.render()
        while not done:
            a,_ = self.min_dic(Q,s)
            s,r,done = self.env.step(a)
            self.env.render()
            self.print_paths()
        #brd.plot_graph()
        #brd()
