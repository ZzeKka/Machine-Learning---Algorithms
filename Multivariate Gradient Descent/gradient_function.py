import numpy as np

def gradient_function(X_train, y_train, w, b):
    m,n = X_train.shape
    j_partial_w = np.zeros((n,)) #vector
    j_partial_b = 0.
    
    for i in range(m): #each data point
        err = (np.dot(X_train[i], w) + b) - y_train[i]
        for j in range(n):
            j_partial_w[j] += err * X_train[i,j] #each attribute
        j_partial_b += err
    j_partial_w /= m
    j_partial_b /= m
    return j_partial_w, j_partial_b    
    
  