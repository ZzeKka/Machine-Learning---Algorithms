#Hello guys today were gonna implement gradient descent algorithm
#We devide in in 3 functions

from cost_function import compute_cost
from compute_gradient import compute_gradient
import math
import numpy as np

HISTORY_SIZE = 10000
x_train = np.array([1.0, 2.0])   
y_train = np.array([300.0, 500.0])

def gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations, compute_cost,compute_gradient):
    J_history = []
    wb_history = []
    
    w = w_init
    b = b_init
    
    for i in range(iterations):
           
        j_wb_w, j_wb_b = compute_gradient(x_train, y_train, w, b)

        w = w - (alpha * j_wb_w)
        b = b - (alpha * j_wb_b)
        
        if i < HISTORY_SIZE:
            J_history.append(compute_cost(x_train,y_train,w,b))
            wb_history.append([w,b])
        
        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {j_wb_w: 0.3e}, dj_db: {j_wb_b: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history, wb_history    




if __name__ == '__main__':
    w_init = 0
    b_init = 0
    # some gradient descent settings
    iterations = 10000
    tmp_alpha = 1.0e-2
    # run gradient descent
    w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                        iterations, compute_cost, compute_gradient)
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
