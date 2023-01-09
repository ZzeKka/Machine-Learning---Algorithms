def compute_gradient(x_train,y_train, w, b):
    m = x_train.shape[0]
    partial_w_cost = 0 
    partial_b_cost = 0
    
    for i in range(m):
        f_wb = w * x_train[i] + b
        partial_w_cost += (f_wb - y_train[i]) * x_train[i]
        partial_b_cost += (f_wb - y_train[i])  
    j_wb_w = partial_w_cost / m
    j_wb_b = partial_b_cost / m
    return j_wb_w, j_wb_b






    