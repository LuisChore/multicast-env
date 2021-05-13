from Graph import Graph
from Agent import Agent

#missing evaluation when an agent is in the source
def set_graph(file_name, paths = False):
    f = open(file_name, "r")
    nodes,edges = f.readline().split()
    g = Graph(int(nodes),True)
    for i in range(int(edges)):
        u,v,w = f.readline().split()
        g.add_edge(int(u),int(v),float(w))
    agents = [Agent(int(x)) for x in f.readline().split()]
    source = int(f.readline())
    if paths:
        for ag in agents:
            ag.path = list(map(int,f.readline().split()))
    return  agents,source,g

def generate_12_8b(file_name,eps,k):
    f = open(file_name,"w")
    f.write("2 2\n")
    f.write("1 0 " + str(1 + eps) + "\n")
    f.write("1 0 " + str(k) + "\n")
    for i in range(k):
        f.write("0 ")
    f.write("\n")
    f.write("1\n")

def generate_12_10(file_name,eps,k, paths = False):
    f = open(file_name,"w")
    f.write(str(k + 2) + " " + str(2*k + 1) + "\n")
    f.write(str(k + 1) + " 0 " + str(1 + eps) + "\n")
    for i in range(1,k+1):
        w = 1.0 / i
        f.write(str(k + 1) + " " + str(i) + " " + str(w) + "\n")

    for i in range(1,k+1):
        f.write("0 " + str(i) + " 0\n")

    for i in range(1,k+1):
        f.write(str(i) + " ")
    f.write("\n")
    f.write(str(k + 1) + "\n")
    if paths:
        for i in range(1,k+1):
            f.write(str(i) + " 0 " + str(k + 1) + "\n")
