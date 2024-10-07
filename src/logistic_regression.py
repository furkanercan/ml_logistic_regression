import numpy as np
import pandas as pd
import copy, math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from data_processing import *

def sigmoid(z):
    return 1/(1+np.exp(-z))

def compute_cost_logistic(X,y,w,b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    
    cost = cost/m
    return cost

def compute_gradient_logistic(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros(n,)
    dj_db = 0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i*X[i,j]
        dj_db = dj_db + err_i
    
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, alpha, num_iters, epsilon, adjust_alpha_enable):
    """
    Performs batch gradient descent
    
    Args:
      X (ndarray (m,n)   : Data, m examples with n features
      y (ndarray (m,))   : target values
      w_in (ndarray (n,)): Initial values of model parameters  
      b_in (scalar)      : Initial values of model parameter
      alpha (float)      : Learning rate
      num_iters (scalar) : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,))   : Updated values of parameters
      b (scalar)         : Updated value of parameter 
    """

    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)  

        w_tmp = w - alpha * dj_dw
        b_tmp = b - alpha * dj_db

        eval_cost = compute_cost_logistic(X, y, w_tmp, b_tmp)
        if(i > 1):
            if(eval_cost > J_history[-1]):
                print(f"Adjusting alpha back from {alpha} to {alpha/3}")
                alpha = alpha / 3
                adjust_alpha_enable = 0
                #Don't inherit new w and b params, move on with new alpha in the new iteration.
            elif(adjust_alpha_enable):
                print(f"Adjusting alpha forward from {alpha} to {alpha*3}")
                alpha = alpha * 3
                w = w_tmp
                b = b_tmp
                J_history.append(eval_cost)
                # p_history.append([w, b])
            else:
                # print(f"{eval_cost} <= {J_history[-1]} but adjust_alpha_enable:{adjust_alpha_enable}")
                w = w_tmp
                b = b_tmp
                J_history.append(eval_cost)
                # p_history.append([w, b])
        else:
            w = w_tmp
            b = b_tmp
            J_history.append(eval_cost)
            # p_history.append([w, b])

        # if i<100000:      # prevent resource exhaustion 
        #     J_history.append( compute_cost_logistic(X, y, w, b) )

        if(i > 5 & adjust_alpha_enable == 0): #Apply convergence
            if(np.abs(J_history[-2] - J_history[-1]) <= epsilon):
                print(f"Convergence at iteration {i}")
                print(f"J_history[-1]: {J_history[-1]}")
                print(f"J_history[-2]: {J_history[-2]}")
                print(f"difference: {J_history[-2] - J_history[-1]}")
                break

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}")
        
    return w, b, J_history         #return final w,b and J history for graphing


### Train/Test Parameters
test_size = 0.2
random_state = 291192
feature_normalization = 1
adjust_alpha_enable = 1

# Obtain the data from the data/raw:
file_path = "data/raw/framingham.csv"
df = pd.read_csv(file_path)

# df_first_3 = df.head(3)
# print(df_first_3)

# print(f"df.info(): {df.info()}")
# print(f"df.shape: {df.shape}")
# print(f"df.describe(): {df.describe()}")


### DATA PREPROCESSING
# 1. Clear nulls (Handle missing data)
# Although there are several other ways of handling with the problem,
# in this case we are going to fill the missing data with the mean of the column.
column_list=list(df.columns)
clear_nulls(df,column_list)

# 2. Handle outliers.
features = list(df.columns)
outlier_handler_zscore(df, features)


X=df.drop("TenYearCHD",axis=1).values
y=df["TenYearCHD"].values
_, num_features = X.shape

X_train_original, X_test_original, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


# 3. Normalize features.
feature_names = df.columns
feature_types = infer_feature_type(df)

if(feature_normalization):
    mean_X_train = np.zeros(num_features)
    std_X_train = np.zeros(num_features)
    X_train_normalized = np.zeros_like(X_train_original)
    X_test_normalized = np.zeros_like(X_test_original)
    for i in range(num_features):
        if(feature_types[feature_names[i]] == 'continuous'): # do it only for continuous features
            mean_X_train[i] = np.mean(X_train_original[:,i])
            std_X_train[i] = np.std(X_train_original[:,i])   
            X_train_normalized[:,i] = (X_train_original[:,i] - mean_X_train[i])/std_X_train[i]
            X_test_normalized[:,i] = (X_test_original[:,i] - mean_X_train[i])/std_X_train[i]
        else: # and exclude categorical features
            X_train_normalized[:,i] = X_train_original[:,i]
            X_test_normalized[:,i] = X_test_original[:,i]

    X_train = X_train_normalized
    X_test = X_test_normalized
else:
    X_train = X_train_original
    X_test = X_test_original


w_tmp  = np.zeros_like(X_train[0])
b_tmp  = 0.
alpha = 0.0000001
iters = 10000
epsilon = 1e-5

w_final, b_final, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alpha, iters, epsilon, adjust_alpha_enable) 
print(f"\nupdated parameters: w:{w_final}, b:{b_final}")

### Testing Phase
z_pred = np.dot(X_test, w_final) + b_final
y_pred_prob = sigmoid(z_pred)
y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_prob]


# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_prob)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Log Loss: {logloss}")