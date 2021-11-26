from QLearning import QLearning


if __name__ == '__main__':
    QL = QLearning("Examples/12.9_paths",beta = 0.99,paths_given = True)
    Q = QL.train(GAMMA = 1, ALPHA = 0.20, epochs = 1000,test = False)
    QL.solve(Q,print_policy = False)
    #QL.print_policy(Q)
