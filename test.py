from QLearning import QLearning
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Apply the default theme
sns.set_theme()
EPOCHS = 1000
GAMMA = 1

def test(ALPHA,EPS,file,paths):
    QL = QLearning(file,paths_given = paths,eps = EPS)
    Q,costs,iterations,differences = QL.train(GAMMA = GAMMA, ALPHA = ALPHA,
        epochs = EPOCHS,test = True)
    mean = np.mean(costs) # mean cost
    std = np.std(costs)  # standard deviation cost

    iterations_mean = np.mean(iterations)
    cost_policy,iterations_policy = QL.solve(Q,render = False)
    return costs,cost_policy,iterations_policy,mean,std,differences,iterations_mean

def compute_graph(ALPHA,EPS,EPOCHS,file,paths):
    NUM_ITERATIONS = 20
    return_values = np.zeros(EPOCHS)
    QL = QLearning(file,paths_given = paths, eps = EPS)
    for i in range(NUM_ITERATIONS):
        Q,costs,iterations,differences = QL.train(GAMMA = GAMMA, ALPHA = ALPHA,
            epochs = EPOCHS,test = True)
        for j,c in enumerate(costs):
            temp_value = (return_values[j]  * i  + c ) / (i + 1)
            return_values[j] = temp_value
    plt.plot(return_values)
    plt.xlabel('Episodios')
    plt.ylabel('Retorno')
    plt.show()

def std_mean(EPOCHS,file_name,paths):
    #list of hyperparameters
    eps = [0.05,0.1,0.15,0.2]
    alphas = [0.05,0.10,0.15,0.2]

    #variables to save best results
    best_cost = float("inf")
    best_eps = None
    best_alpha = None
    best_differences = None
    best_iterations = None
    best_mean = None
    best_std = None
    best_costs = None

    #track training process to plot
    cost_means = []
    labels = []
    cost_std = []

    training = {}
    columns = ['epsilon','alpha','mean_cost','mean_iterations','cost_test','iterations_test']
    for c in columns:
        training[c] = []

    for e in eps:
        for a in alphas:
            R = test(a,e,file_name,paths)
            costs,cost_policy,it_policy,mean,std,differences,it_mean = R
            print("---Training Summary---")
            print('Epsilon: ' + '{0:.2f}'.format(e))
            training['epsilon'].append(e)
            print('Alpha: ' + '{0:.2f}'.format(a))
            training['alpha'].append(a)
            print('Cost Mean: ' + '{0:.2f}'.format(mean))
            training['mean_cost'].append(mean)
            print('Iterations Mean: ' + '{0:.2f}'.format(it_mean))
            training['mean_iterations'].append(it_mean)
            print("----Following Best Policy----")
            print("Cost: " + '{0:.2f}'.format(cost_policy))
            training['cost_test'].append(cost_policy)
            print(f"Iterations: {it_policy}")
            training['iterations_test'].append(it_policy)
            print("\n")
            cost_means.append(mean)
            cost_std.append(std)
            labels.append((e,a))
            if best_cost > cost_policy:
                best_mean = mean
                best_std = std
                best_cost = cost_policy
                best_iterations = it_policy
                best_eps = e
                best_alpha = a
                best_differences = differences
                best_costs = costs
            elif best_cost == cost_policy and it_policy < best_iterations:
                best_mean = mean
                best_std = std
                best_cost = cost_policy
                best_iterations = it_policy
                best_eps = e
                best_alpha = a
                best_differences = differences
                best_costs = costs
            elif best_cost == cost_policy and it_policy == best_iterations and mean < best_mean:
                best_mean = mean
                best_std = std
                best_cost = cost_policy
                best_iterations = it_policy
                best_eps = e
                best_alpha = a
                best_differences = differences
                best_costs = costs
            elif best_cost == cost_policy and it_policy == best_iterations and mean == best_mean and best_std > std:
                best_mean = mean
                best_std = std
                best_cost = cost_policy
                best_iterations = it_policy
                best_eps = e
                best_alpha = a
                best_differences = differences
                best_costs = costs

    pd.options.display.float_format = "{:,.3f}".format
    df = pd.DataFrame(training,columns = columns )
    df['mean_cost'] = df['mean_cost'].round(4)
    df['cost_test'] = df['cost_test'].round(4)
    df.to_csv(file_name + '.csv',index = False)
    print(df)
    print("Summary (Best hyperparameters):")
    print("Epsilon: " + '{0:.2f}'.format(best_eps))
    print("Alpha: " + '{0:.2f}'.format(best_alpha))
    print("Cost: " + '{0:.2f}'.format(best_cost))
    print("Mean Cost: " + '{0:.2f}'.format(best_mean))
    print(f"Iterations: {best_iterations}")

    plt.scatter(cost_std,cost_means)
    for i, txt in enumerate(labels):
        form = '({0:.2f},{1:.2f})'.format(txt[0],txt[1])
        plt.annotate(form, (cost_std[i], cost_means[i]), fontsize = 8)

    plt.xlabel('Desviación estandar')
    plt.ylabel('Media del valor de retorno')
    plt.show()

if __name__ == '__main__':
    #compute_graph(0.2,0.1,EPOCHS,"Examples/12.9_paths",True)
    std_mean(EPOCHS,"Examples/12.9_paths",True)
