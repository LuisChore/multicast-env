from QLearning import QLearning
import matplotlib.pyplot as plt
import numpy as np

ALPHA = 0.1
EPOCHS = 1000

def test(gamma,eps,file,paths):
    QL = QLearning(file,paths_given = paths)
    Q,costs,iterations,differences = QL.train(GAMMA = gamma, ALPHA = ALPHA,
        epochs = EPOCHS,test = True)
    mean = np.mean(costs) # mean cost
    std = np.std(costs)  # standard deviation cost

    iterations_mean = np.mean(iterations)
    cost_policy,iterations_policy,cost_env = QL.solve(Q,render = False)
    return cost_policy,iterations_policy,cost_env,mean,std,differences,iterations_mean


if __name__ == '__main__':
    eps = [0.1,0.15,0.2]
    gammas = [0.9,0.95,0.99]
    file_name = "Examples/12.10_paths"
    paths = True

    best_cost = float("inf")
    best_env_cost = None
    best_e = None
    best_g = None
    best_differences = None
    best_iterations = None
    cost_means = []
    labels = []
    cost_std = []
    for e in eps:
        for g in gammas:
            R = test(g,e,file_name,paths)
            cost_policy,it_policy,cost_env,mean,std,differences,it_mean = R
            print("---Training---")
            print('Epsilon: ' + '{0:.2f}'.format(e))
            print('Gamma: ' + '{0:.2f}'.format(g))
            print('Cost Mean: ' + '{0:.2f}'.format(mean))
            print('Iterations Mean: ' + '{0:.2f}'.format(it_mean))
            print("----Following Best Policy----")
            print("Cost: " + '{0:.2f}'.format(cost_policy))
            print(f"Iterations: {it_policy}")
            print("\n")

            cost_means.append(mean)
            cost_std.append(std)
            labels.append((e,g))
            if best_cost > cost_policy:
                best_env_cost = cost_env
                best_cost = cost_policy
                best_iterations = it_policy
                best_e = e
                best_g = g
                best_differences = differences

    print("Best:")
    print("Cost: " + '{0:.2f}'.format(best_cost))
    print(f"Iterations: {best_iterations}")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Eps: {0:.2f}, Gamma: {1:.2f}".format(best_e,best_g) )

    ax1.plot(best_differences)
    ax2.scatter(cost_std,cost_means)


    for i, txt in enumerate(labels):
        form = '({0:.2f},{1:.2f})'.format(txt[0],txt[1])
        ax2.annotate(form, (cost_std[i], cost_means[i]))

    ax2.set(xlabel='Standard Deviation', ylabel='Mean Cost')
    plt.show()
