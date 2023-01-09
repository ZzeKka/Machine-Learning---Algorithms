def compute_cost(x_train,y_train,w,b):
    m = x_train.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x_train[i] + b
        cost += (f_wb - y_train[i]) ** 2
    total_cost = (1 / (2 * m)) * cost
    return total_cost  


