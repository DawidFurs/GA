import numpy as np
import ga
import time
import matplotlib.pyplot as plt

no_variables = 2
pop_size = 100
crossover_rate = 50
mutation_rate = 50
no_generations = 300
lower_bounds = [-512, -512]
upper_bounds = [512, 512]
step_size = 5
rate = 10
pop = np.zeros((pop_size, no_variables))
for s in range(pop_size):
    for h in range(no_variables):
        pop[s, h] = np.random. uniform(lower_bounds[h], upper_bounds[h])

extended_pop = np.zeros((pop_size+crossover_rate+mutation_rate+2*no_variables*rate,pop.shape[1]))

fig = plt.figure()
ax = fig.add_subplot()
fig.show()
plt.title('Evolutionary process of the objective function value')
plt.xlabel("Iteration")
plt.ylabel("Objective function")

A =[]
a = 5 # adaptive restart
g = 0
global_best = pop
k = 0

while g <= no_generations:
    for i in range(no_generations):
        offspring1 = ga.crossover(pop, crossover_rate)
        offspring2 = ga.mutation(pop, mutation_rate)
        fitness = ga.objective_function(pop)
        offspring3 = ga.local_search(pop, fitness, lower_bounds, upper_bounds, step_size, rate)
        step_size = step_size*0.98
        if step_size < 1:
            step_size = 1
        extended_pop[0: pop_size] = pop
        extended_pop[pop_size: pop_size+crossover_rate] = offspring1
        extended_pop[pop_size+crossover_rate:pop_size+crossover_rate+mutation_rate] = offspring2
        extended_pop[pop_size+crossover_rate+mutation_rate:pop_size+crossover_rate+mutation_rate+2*no_variables*rate] = offspring3
        fitness = ga.objective_function(extended_pop)
        pop = ga.selection(extended_pop, fitness, pop_size)
        print("Generation: ", g, ", Current fitness value: ", 10e6-max(fitness))
        A.append(10e6 - max(fitness))
        g += 1
        if i >= a:
            if sum(abs(np.diff(A[g - a:g]))) <= 0.05:
                index = np.argmax(fitness)
                current_best = extended_pop[index]
                pop = np.zeros((pop_size, no_variables))
                for s in range(pop_size):
                    for h in range(no_variables):
                        pop[s, h] = np.random.uniform(lower_bounds[h], upper_bounds[h])
                step_size = 5
                global_best[k] = current_best
                k += 1
                break

            ax.plot(A, color='r')
            fig.canvas.draw()
            ax.set_xlim(letf=max(0, g - no_generations), right=g + 3)
            if g > no_generations:
                break
        if g > no_generations:
            break
fitness = ga.objective_function(global_best)
index = np.argmax(fitness)
print("Best solution = ", global_best[index])
print("Best fitness value = ", 10e6 - max(fitness))
plt.show()