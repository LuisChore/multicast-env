
#0-index
class Graph:
    def __init__(self,nodes,directed = False):
        self.nodes = nodes
        self.adj = []
        self.directed = directed
        for i in range(nodes):
            self.adj.append([])

    def add_edge(self,u,v,w):
        self.adj[u].append((v,w))
        if self.directed == False:
            self.adj[v].append((u,w))
            
