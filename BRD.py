'''
Reinforcement Learning Environment to find a Nash Equilibrium
'''
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import networkx as nx
import numpy as np
from Agent import Agent
from Graph import Graph
import queue
import copy

class Environment:
    def __init__(self,graph,agents,source,beta = 0.99,step_cost = 100,paths_given = False):
        self.nash_eq = False # done variable
        self.beta = beta # for action values
        self.start = False # to initialize the graphics
        self.step_cost = step_cost # for the reward
        self.W,self.H = 12,8
        self.paths_given = paths_given
        colors = ['blue','red','green','orange','cyan','black','pink','magenta']
        self.cost_by_iteration = []
        self.total_cost = float("inf")
        # initialize agents
        self.source = source # node source
        self.agents = agents
        iterator = 0
        for ag in self.agents:
            ag.set_color(colors[iterator])
            iterator = (iterator + 1) % len(colors)

        if paths_given == True:
            self.agents_copy = copy.deepcopy(self.agents)
        #initialize graph
        self.graph = graph
        self.graphx = self.create_graph(graph)
        #Edges: dictionary to know how many agents are using every edge
        self.Edges = self.initialize_edges()
        #create state (also needed for  BRD altough is always 0)
        self.state = [0] * len(list(self.edge_list))
        self.episode_iterations = 0
        self.harmonic = self.process_harmonic(len(agents))
        self.initialize_agents()

    def process_harmonic(self,n):
        harmonic = [0]
        for i in range(1,n+1):
            harmonic.append(1/i)

        for i in range(1,n+1):
            harmonic[i] += harmonic[i-1]
        return harmonic

    # check in BRD potential_function
    def potential_function(self):
        value = 0
        for e,(w,c) in self.Edges.items():
            value += w * self.harmonic[c]
        return value

    # given an action, change the edge cost
    def modify_state(self,action:int):
        edge_index = int(action / 3) # 3 edge operations
        edge_operation = action % 3 # choose operation
        self.state[edge_index] = edge_operation

    def step(self,action):
        #action integer [0,...,3*|E|-1]
        self.modify_state(action)
        # update costs according to new weights
        for ag in self.agents:
            self.update_agent_cost(ag)
        self.iteration()
        last_potential_value = self.potential_value
        self.potential_value = self.potential_function()
        reward = self.potential_value - last_potential_value + self.step_cost
        self.nash_eq = self.is_NE()
        return tuple(self.state),reward,self.nash_eq

    def reset_edges(self):
        Edges = {}
        for u in range(self.graph.nodes):
            for v,w in self.graph.adj[u]:
                i = min(u,v)
                j = max(u,v)
                Edges[(i,j)] = (w,0)
        return Edges

    def reset(self):
        self.nash_eq = False
        self.episode_iterations = 0
        self.state = [0] * len(list(self.edge_list))
        self.reset_agents()
        #Edges: dictionary to know how many agents are using every edge
        self.Edges = self.reset_edges()
        if self.paths_given == False:
            self.iteration()
        else:
            self.agents = copy.deepcopy(self.agents_copy)
            self.initialize_agents()
        self.potential_value = self.potential_function()
        self.nash_eq = self.is_NE()
        return tuple(self.state),self.nash_eq

    def render(self, W = 6, H = 5):
        if self.start == False:
            self.start = True
            self.figEnv = plt.figure(figsize = (W,H))
            plt.ion()
        else:
            plt.clf()
        self.figEnv.canvas.set_window_title(f"It  {int(self.episode_iterations)}: {self.total_cost}")
        pos = nx.planar_layout(self.graphx)
        nx.draw(self.graphx,pos,with_labels = True)
        edge_labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in self.graphx.edges(data=True)])
        #edge_labels = nx.get_edge_attributes(self.graphx,'weight')
        nx.draw_networkx_edge_labels(self.graphx,pos,edge_labels = edge_labels)

        for ag in self.agents:
            nx.draw_networkx_edges(self.graphx, pos,edgelist = ag.get_path(),width=4,
            alpha=0.5, edge_color=ag.color, style='dashed',label = ag.index)
        plt.show()
        plt.pause(2)
        plt.draw()

    def iteration(self):
        if self.nash_eq == True:
            return
        self.episode_iterations += 1
        np.random.shuffle(self.agents)
        change = False
        for ag in self.agents:
            find = self.find_path(ag)
            if find == True:
                change = True
        for ag in self.agents:
            self.update_agent_cost(ag)
            ag.add_cost()
        self.evaluate_totalcost()
        self.cost_by_iteration.append(self.total_cost)


    def reset_agents(self):
        self.cost_by_iteration = []
        for ag in self.agents:
            ag.edges_used = {}
            ag.path = []
            ag.cost = 0 if ag.index == self.source else float('inf')
            ag.cost_by_iteration = []
        self.evaluate_totalcost()

    # function for simulator (agents already could have paths)
    def initialize_agents(self):
        for ag in self.agents:
            ag.edges_used = {}
            l = len(ag.path)
            for i in range(0,l-1):
                u = min(ag.path[i],ag.path[i + 1])
                v = max(ag.path[i],ag.path[i + 1])
                w,h = self.Edges[(u,v)]
                self.Edges[(u,v)] = (w,h+1)
                ag.edges_used[(u,v)] = True

        for ag in self.agents:
            l = len(ag.path)
            if l > 0 or ag.index == self.source:
                #if it's in the source it doesn't need a path
                self.update_agent_cost(ag)
        self.evaluate_totalcost()

    def create_graph(self,graph):
        G = nx.DiGraph()
        for i in range(graph.nodes):
            G.add_node(i)
        for u in range(graph.nodes):
            for v,w in graph.adj[u]:
                G.add_edge(u,v,weight = w)
        return G

    #returns the current cost according to the new weights configuration
    def current_edge_cost(self,edge):
        w,c = self.Edges[edge]
        index_edge = self.edge_dict[edge]
        if self.state[index_edge] == 0:
            return w
        elif self.state[index_edge] == 1:
            return w + self.beta * w
        return w -self.beta * w

    def update_agent_cost(self,agent):
        # Edges: dictionary  edge -> (real_weight,agents_using_it )
        # state: envirionment state
        cost = 0
        for e in agent.edges_used:
            w,c = self.Edges[e]
            cost +=  self.current_edge_cost(e)/c
        agent.cost = cost

    def initialize_edges(self):
        Edges = {}
        self.edge_list = []
        self.edge_dict = {}
        for u in range(self.graph.nodes):
            for v,w in self.graph.adj[u]:
                i = min(u,v)
                j = max(u,v)
                self.edge_dict[(i,j)] = i
                self.edge_list.append((i,j))
                Edges[(i,j)] = (w,0)
        return Edges

    def plot_graph(self):
        plt.ioff()
        self.fig = plt.figure(figsize = (6,5))
        label = '\n'.join(("Agents: " + str([ag.index for ag in self.agents]), "Source: " + str(self.source)))
        self.fig.canvas.set_window_title('Graph')
        self.fig.text(0.01,0.90, label)
        pos = nx.planar_layout(self.graphx)
        nx.draw(self.graphx,pos,with_labels = True)
        edge_labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in self.graphx.edges(data=True)])
        #edge_labels = nx.get_edge_attributes(self.graphx,'weight')
        nx.draw_networkx_edge_labels(self.graphx,pos,edge_labels = edge_labels)
        plt.show()

    def next_function(self,event):
        self.nash_eq = True
        for ag in self.agents:
            find = self.find_path(ag)
            if find == True:
                self.nash_eq = False
        for ag in self.agents:
            self.update_agent_cost(ag)
            ag.add_cost()
        self.evaluate_totalcost()
        self.cost_by_iteration.append(self.total_cost)

        plt.clf()
        self.plot_button()
        self.plot_paths()
        self.plot_total_cost()
        self.plot_costs()
        plt.draw()

    def plot_button(self):
        axnext = plt.axes([0.82, 0.01, 0.1, 0.065])
        message = "Nash Equilibrium" if self.nash_eq else "Next Step"
        self.bnext = Button(axnext, message)
        self.bnext.on_clicked(self.next_function)

    def plot_paths(self):
        fgraph = self.fig.add_subplot(self.gs[0:2,:])
        pos = nx.planar_layout(self.graphx)
        title = "Source: " + str(self.source) + ", Total cost: " + str(round(self.total_cost,2))
        fgraph.title.set_text(title)
        nx.draw(self.graphx,pos,with_labels = True)
        edge_labels = dict([((u,v,), f"{d['weight']:.2f}") for u,v,d in self.graphx.edges(data=True)])
        #edge_labels = nx.get_edge_attributes(self.graphx,'weight')
        nx.draw_networkx_edge_labels(self.graphx,pos,edge_labels = edge_labels)

        for ag in self.agents:
            nx.draw_networkx_edges(self.graphx, pos,edgelist = ag.get_path(),width=4,
            alpha=0.5, edge_color=ag.color, style='dashed',label = ag.index)

    def plot_total_cost(self):
        fmetrics = self.fig.add_subplot(self.gs[2,0])
        fmetrics.set_ylabel('Total cost')
        fmetrics.set_xlabel('Iterations')
        fmetrics.title.set_text("Total cost: " + str(round(self.total_cost,2)))
        fmetrics.plot(self.cost_by_iteration)

    def is_NE(self):
        for ag in self.agents:
            find = self.find_path(ag,update = False)
            if find == True:
                return False
        return True

    def plot_costs(self):
        fagents = self.fig.add_subplot(self.gs[2,1])
        fagents.set_ylabel('Cost')
        fagents.set_xlabel('Iterations')
        fagents.title.set_text("Cost by agent")
        for ag in self.agents:
            label  = str(ag.index) + ": " +  str(round(ag.cost, 2))
            fagents.plot(ag.cost_by_iteration, color = ag.color, label = label)
            fagents.legend()

    def __call__(self):
        self.fig = plt.figure(figsize = (self.W,self.H))
        self.fig.canvas.set_window_title("BRD")
        self.gs = self.fig.add_gridspec(3, 2)
        self.cost_by_iteration.append(self.total_cost)
        for ag in self.agents:
            ag.add_cost()

        self.plot_button()
        self.plot_paths()
        self.plot_total_cost()
        self.plot_costs()

        plt.show()

    def update_edges(self,agent,change = True):
        l = len(agent.path)
        for i in range(0,l-1):
            u = min(agent.path[i],agent.path[i + 1])
            v = max(agent.path[i],agent.path[i + 1])

            w,h = self.Edges[(u,v)]
            if change == False:
                self.Edges[(u,v)] = (w,h-1)
                del agent.edges_used[(u,v)]
            else:
                self.Edges[(u,v)] = (w,h+1)
                agent.edges_used[(u,v)] = True

    def find_path(self,agent,update = True):
        index = agent.index
        prev_cost = agent.cost
        dist = [float("inf") for i in range(self.graph.nodes)]
        parent = [-1 for i in range(self.graph.nodes)]
        dist[self.source] = 0
        parent[self.source] = self.source
        PQ = queue.PriorityQueue()
        PQ.put((0,self.source))
        while PQ.empty() == False:
            w,u = PQ.get()
            if dist[u] < w:
                continue
            for v,c in self.graph.adj[u]:
                original_cost = True
                if update == True:
                    original_cost = False
                realcost = self.get_realcost(u,v,agent.contain_edge(u,v),original_cost)
                if dist[v] > dist[u] + realcost:
                    parent[v] = u
                    dist[v] = dist[u] + realcost
                    PQ.put((dist[v],v))

        if dist[index] >= prev_cost:
            return False

        #if we are not updating we only return there is a better option
        if update == False:
            return True
        new_path = []
        self.create_path(index,parent,new_path)
        self.update_edges(agent,False)
        agent.path = new_path
        self.update_edges(agent,True)
        agent.cost = dist[index]
        return True

    def create_path(self,u,parent,new_path):
        if parent[u] == u:
            new_path.append(u)
            return
        new_path.append(u)
        self.create_path(parent[u],parent,new_path)

    # real cost if the agent would use this edge
    def get_realcost(self,u,v,contained,original_cost):
        i = min(u,v)
        j = max(u,v)
        w,h = self.Edges[(i,j)]
        cost = w if original_cost else self.current_edge_cost((i,j))
        if contained:
            return (cost/h)
        else:
            return (cost/(h+1))

    def evaluate_totalcost(self):
        ans = 0
        for ag in self.agents:
            ans += ag.cost
        self.total_cost = ans
