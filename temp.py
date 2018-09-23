import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import csv
import random
rand = random.random()
import os


result_path = './Result_possessed/figure2/'
iter = np.loadtxt(open(result_path + '3_1_0.csv'),delimiter=",",skiprows=0)[:,0]
l3_r0_d0 = np.loadtxt(open(result_path + '3_0_0.csv'),delimiter=",",skiprows=0)[:,1]
l3_r0_d1 = np.loadtxt(open(result_path + '3_0_1.csv'),delimiter=",",skiprows=0)[:,1]
l3_r1_d0 = np.loadtxt(open(result_path + '3_1_0.csv'),delimiter=",",skiprows=0)[:,1]
l3_r1_d1 = np.loadtxt(open(result_path + '3_1_1.csv'),delimiter=",",skiprows=0)[:,1]

l5_r0_d0 = np.loadtxt(open(result_path + '5_0_0.csv'),delimiter=",",skiprows=0)[:,1]
l5_r0_d1 = np.loadtxt(open(result_path + '5_0_1.csv'),delimiter=",",skiprows=0)[:,1]
l5_r1_d0 = np.loadtxt(open(result_path + '5_1_0.csv'),delimiter=",",skiprows=0)[:,1]
l5_r1_d1 = np.loadtxt(open(result_path + '5_1_1.csv'),delimiter=",",skiprows=0)[:,1]

l8_r0_d0 = np.loadtxt(open(result_path + '8_0_0.csv'),delimiter=",",skiprows=0)[:,1]
l8_r0_d1 = np.loadtxt(open(result_path + '8_0_1.csv'),delimiter=",",skiprows=0)[:,1]
l8_r1_d0 = np.loadtxt(open(result_path + '8_1_0.csv'),delimiter=",",skiprows=0)[:,1]
l8_r1_d1 = np.loadtxt(open(result_path + '8_1_1.csv'),delimiter=",",skiprows=0)[:,1]

l10_r0_d0 = np.loadtxt(open(result_path + '10_0_0.csv'),delimiter=",",skiprows=0)[:,1]
l10_r0_d1 = np.loadtxt(open(result_path + '10_0_1.csv'),delimiter=",",skiprows=0)[:,1]
l10_r1_d0 = np.loadtxt(open(result_path + '10_1_0.csv'),delimiter=",",skiprows=0)[:,1]
l10_r1_d1 = np.loadtxt(open(result_path + '10_1_1.csv'),delimiter=",",skiprows=0)[:,1]

def Data_Save(epoch,train_dice):
    data = [epoch,float(train_dice)]
    file= "./Result_possessed/figure2/10_0_1_100.csv"
    file_path = os.path.split(file)[0]
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    if not os.path.exists(file):
        os.system(r'touch %s' % file)
    writer = csv.writer(open(file, 'a+')) #a+ do not overwrite origenal data,'wb',overwrite origenal data
    writer.writerow(data)

for i in range(0,1000):
    if i % 10 == 0:
        Data_Save(iter[i]/10,l10_r0_d1[i])
        print(i,l8_r1_d1[i] )