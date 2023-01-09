import copy
import math
import numpy as np
from cost_function import *
from gradient_function import *

MAX_ITERATIONS = 10000

X_train = np.array([[2104, 5, 1, 45], 
                    [1416, 3, 2, 40],
                    [852, 2, 1, 35]])

y_train = np.array([460, 232, 178])

#print(cost_function(X_train,y_train,[100,100,100,100],300))
#print(gradient_function(X_train,y_train,[100,100,100,100],300))

def gradient_descent(X_train, y_train, w_init, b_init, cost_function, gradient_function, alpha, iterations):
    w = copy.deepcopy(w_init)  #avoid modifying global w within function
    b = b_init
    
    J_history = []
    
    for i in range(iterations):
        j_partial_w, j_partial_b = gradient_function(X_train, y_train, w, b)
        
        w = w - (alpha * j_partial_w)
        b = b - (alpha * j_partial_b)
        
        if i<MAX_ITERATIONS:      
            J_history.append( cost_function(X_train, y_train, w, b))

        
        if i% math.ceil(iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history

if __name__ == "__main__":
    # initialize parameters
    initial_w = np.zeros_like(np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618]))
    print(initial_w)
    initial_b = 0.
    # some gradient descent settings
    iterations = 1000
    alpha = 5.0e-7
    # run gradient descent 
    w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                        cost_function, gradient_function, 
                                                        alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    m,_ = X_train.shape
    for i in range(m):
        print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")