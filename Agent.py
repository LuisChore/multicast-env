

class Agent:
    def __init__(self,index):
        self.index = index
        self.cost = float("inf")
        self.path = []
        self.cost_by_iteration = []
        self.edges_used = {}

    def __str__(self):
        S = "Index: " + str(self.index) + "\n"
        S += "Cost: " + str(self.cost) + "\n"
        S += "Path:\n";
        for i in range(len(self.path)):
            S += str(self.path[i])
            if i != len(self.path) -1:
                S += ","
        return S

    def get_cost(self):
        return self.cost

    def set_color(self,c):
        self.color = c

    def get_index(self):
        return self.index

    def add_cost(self):
        self.cost_by_iteration.append(self.cost)

    def get_path(self):
        edgelist = []
        for i in range(len(self.path) - 1):
            edgelist.append((self.path[i],self.path[i+1]))
        return edgelist

    def contain_edge(self,u,v):
        i = min(u,v)
        j = max(u,v)
        if (i,j) in self.edges_used:
            return True
        return False
