# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 22:49:36 2017

@author: dell
"""

import os
path = 'D:\Anaconda3\Lib\site-packages\libsvm\python'
os.chdir(path)

from svm import *
from svmutil import *
from grid import *

#rate, param = find_parameters('C:\\Users\\dell\\Desktop\\wd.txt', '-log2c -3,10,1 -log2g -3,3,1')

#from sklearn.cross_validation import train_test_split  
#train_label, train_value = svm_read_problem("C:\\Users\\dell\\Desktop\\traindata.txt")
#predict_label, predict_value = svm_read_problem("C:\\Users\\dell\\Desktop\\testdata.txt")   #预测数
#model = svm_train(train_label,train_value)
#p_label, p_acc, p_val = svm_predict(predict_label, predict_value, model)
#print(p_acc) 

label, value = svm_read_problem("C:\\Users\\dell\\Desktop\\erlei.txt")

sum = 0
times = 1000
for i in range(times):
    X_train, X_test, y_train, y_test = train_test_split(value, label, test_size = 0.2)
    model = svm_train(y_train, X_train, '-s 0   -t 0 -c 500')
    p_label, p_acc, p_val = svm_predict(y_test, X_test, model)
    sum += p_acc[0]
print('average acc is %f' % (sum/times))