import numpy as np

def cal_avg_non_diagnoal_elements(x):
    return (x.sum() - x.diagonal().sum())/(x.shape[0]*(x.shape[0]-1))

def cal_avg_upper_triangle(x):
    return (x.sum() - np.tril(x).sum())/(x.shape[0]*(x.shape[0]-1)/2)

def cal_column_avg_upper_triangle(x, bidirection: 1): # bidirection: 0: upper; 1: lower; 2:bidirectional
    if(bidirection==2):
        x += x.T
        x /= 2
    elif(bidirection==1):
        x = x.T
        
    upper_tri = (x - np.tril(x)).T
    # upper_tri = np.tril(x).T
    column_cohs = []
    for i in range(1, len(upper_tri)):
        column_cohs.append(upper_tri[i].sum()/i)
    return np.mean(column_cohs)