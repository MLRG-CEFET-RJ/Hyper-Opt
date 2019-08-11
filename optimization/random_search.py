import numpy as np

num_trials = 100
i=0
best_result = 0

def objective_function(l1, l2, l3):
    return l1+l2+l3

while i < num_trials:
    layers = np.random.randint(low=100, high=500, size=3)
    lr = np.around(np.random.uniform(low=0.1, high=0.0001, size=1), decimals=6)
    rr = np.around(np.random.uniform(low=0, high=0.001, size=1), decimals=6)
    print("Iteration number: ", layers)
    max_objective = objective_function(layers[0], layers[1], layers[2])
    
    if max_objective > best_result:
        best_result = max_objective
        best_params = layers
        best_trial = i
        print("Best result Updated!")
    i+=1

print("Random Search final result: ", best_result)
print("Random Search final params: ", best_params)