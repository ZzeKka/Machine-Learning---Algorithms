import numpy as np

def cost_function(X_train, y_train, w, b):
    m = X_train.shape[0]
    cost = 0
    for i in range(m):
        cost += ((np.dot(X_train[i],w) + b) - y_train[i]) ** 2
    total_cost = cost / (2 * m)
    return total_cost
    
    