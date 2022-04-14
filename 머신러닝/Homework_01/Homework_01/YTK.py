# -*- coding: utf-8 -*-

import numpy as np
      
def feature_normalization(data): # 10 points
    # parameter 
    feature_num = data.shape[1]
    data_point = data.shape[0]
    
    # you should get this parameter correctly
    normal_feature = np.zeros([data_point, feature_num])
    mu = np.zeros([feature_num])
    std = np.zeros([feature_num])
    
    # your code here
    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    #normal_feature = (data-np.min(data, axis=0))/(np.max(data, axis=0) - np.min(data, axis=0))
    normal_feature = (data - mu) / std
    
    # end
    
    return normal_feature
        
def split_data(data, label, split_factor):
    return  data[:split_factor], data[split_factor:], label[:split_factor], label[split_factor:]

def get_normal_parameter(data, label, label_num): # 20 points
    # parameter
    feature_num = data.shape[1]
    
    # you should get this parameter correctly    
    mu = np.zeros([label_num,feature_num])
    sigma = np.zeros([label_num,feature_num])

    # your code here
    
        
    labeled_data = np.zeros([data.shape[0], label_num, feature_num])
    
    for i in range(0, data.shape[0]):
        for k in range(0, feature_num):
            _label = label[i]
            labeled_data[i][_label][k] = data[i][k]
    
    mu = np.mean(labeled_data, axis=0)
    sigma = np.std(labeled_data, axis=0)
    #print(mu)
    #print(sigma)
    # end
    
    return mu, sigma

def get_prior_probability(label, label_num): # 10 points
    # parameter
    data_point = label.shape[0]
    
    # you should get this parameter correctly
    prior = np.zeros([label_num])
    
    # your code here
    
    for i in range(0, data_point):
        idx = label[i]
        prior[idx] += 1;
    
    prior = prior/data_point
    #print(prior)

    # end
    return prior

def Gaussian_PDF(x, mu, sigma): # 10 points
    # calculate a probability (PDF) using given parameters
    # you should get this parameter correctly
    pdf = 0
    
    # your code here
    
    var = float(sigma)**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(float(x)-float(mu))**2/(2*var))
    pdf = num/denom
    # end
    
    return pdf

def Gaussian_Log_PDF(x, mu, sigma): # 10 points
    # calculate a probability (PDF) using given parameters
    # you should get this parameter correctly
    log_pdf = 0
    
    # your code here
    
    var = float(sigma)**2
    denom = (2*np.pi*var)**.5
    num = np.exp(-(float(x)-float(mu))**2/(2*var))
    log_pdf = np.log(num/denom)
    # end
    
    return log_pdf

def Gaussian_NB(mu, sigma, prior, data): # 40 points
    # parameter
    data_point = data.shape[0]
    label_num = mu.shape[0]
    
    # you should get this parameter correctly   
    likelihood = np.zeros([data_point, label_num])
    posterior = np.zeros([data_point, label_num])
    ## evidence can be ommitted because it is a constant
    
    # your code here
        ## Function Gaussian_PDF or Gaussian_Log_PDF should be used in this section
    
    
    for i in range(0, label_num):
        for j in range(0, data_point):
            likelihood[j][i] = 1
            for k in range(0, data.shape[1]):
                likelihood[j][i] += Gaussian_Log_PDF(data[j][k], mu[i][k], sigma[i][k])
                
    
    for i in range(0, label_num):
        for j in range(0, data_point):
            posterior[j][i] = (prior[i]) + likelihood[j][i]
            
        
    # end
    return posterior

def classifier(posterior):
    data_point = posterior.shape[0]
    prediction = np.zeros([data_point])
    
    prediction = np.argmax(posterior, axis=1)
    
    return prediction
        
def accuracy(pred, gnd):
    data_point = len(gnd)
    
    hit_num = np.sum(pred == gnd)

    return (hit_num / data_point) * 100, hit_num

    ## total 100 point you can get 