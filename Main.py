from Graph import Graph
from BRD import Environment
from Utils import *
import numpy as np
#generate_12_8b("Examples/12.8b",0.5,5)
#generate_12_10("Examples/12.10_paths",0.5,5,paths = True)
agents, source, g = set_graph("Examples/12.10_paths ",paths = True)
brd = Environment(g,agents,source)

s,done = brd.reset()
print(done)
print(brd.potential_function())
for ag in brd.agents:
    print(ag.index)
    print(ag.path)
for i in range(5):
    print("-----------------------------------")
    s,r,done = brd.step(np.random.randint(5))
    print(s)
    print(r)
    print(done)
    for ag in brd.agents:
        print(ag.index)
        print(ag.path)
#brd.plot_graph()
#brd()
