import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import csv
import random
rand = random.random()

result_path = './Result_possessed/all_original/'
iter = np.loadtxt(open(result_path + '3_0_0_100.csv'),delimiter=",",skiprows=0)[:,0]
l3_r0_d0 = np.loadtxt(open(result_path + '3_0_0_100.csv'),delimiter=",",skiprows=0)[:,1]
l3_r0_d1 = np.loadtxt(open(result_path + '3_0_1_100.csv'),delimiter=",",skiprows=0)[:,1]
l3_r1_d0 = np.loadtxt(open(result_path + '3_1_0_100.csv'),delimiter=",",skiprows=0)[:,1]
l3_r1_d1 = np.loadtxt(open(result_path + '3_1_1_100.csv'),delimiter=",",skiprows=0)[:,1]

l5_r0_d0 = np.loadtxt(open(result_path + '5_0_0_100.csv'),delimiter=",",skiprows=0)[:,1]
l5_r0_d1 = np.loadtxt(open(result_path + '5_0_1_100.csv'),delimiter=",",skiprows=0)[:,1]
l5_r1_d0 = np.loadtxt(open(result_path + '5_1_0_100.csv'),delimiter=",",skiprows=0)[:,1]
l5_r1_d1 = np.loadtxt(open(result_path + '5_1_1_100.csv'),delimiter=",",skiprows=0)[:,1]

l8_r0_d0 = np.loadtxt(open(result_path + '8_0_0_100.csv'),delimiter=",",skiprows=0)[:,1]
l8_r0_d1 = np.loadtxt(open(result_path + '8_0_1_100.csv'),delimiter=",",skiprows=0)[:,1]
l8_r1_d0 = np.loadtxt(open(result_path + '8_1_0_100.csv'),delimiter=",",skiprows=0)[:,1]
l8_r1_d1 = np.loadtxt(open(result_path + '8_1_1_100.csv'),delimiter=",",skiprows=0)[:,1]

l10_r0_d0 = np.loadtxt(open(result_path + '10_0_0_100.csv'),delimiter=",",skiprows=0)[:,1]
l10_r0_d1 = np.loadtxt(open(result_path + '10_0_1_100.csv'),delimiter=",",skiprows=0)[:,1]
l10_r1_d0 = np.loadtxt(open(result_path + '10_1_0_100.csv'),delimiter=",",skiprows=0)[:,1]
l10_r1_d1 = np.loadtxt(open(result_path + '10_1_1_100.csv'),delimiter=",",skiprows=0)[:,1]

plt.figure()
line_l3_r0_d0, = plt.plot(iter, l3_r0_d0, color='red', linewidth=1.0, label='Regular_3')
# line_l3_r0_d1, = plt.plot(iter, l3_r0_d1, color='blue', linewidth=1.0, label='Dilated_3')
# line_l3_r1_d0, = plt.plot(iter, l3_r1_d0, color='green', linewidth=1.0, label='Residual_3')
# line_l3_r1_d1, = plt.plot(iter, l3_r1_d1, color='orange', linewidth=1.0, label='Residual_Dilated_3')
#
line_l5_r0_d0, = plt.plot(iter, l5_r0_d0, color='blue', linewidth=1.0, label='Regular_5')
# line_l5_r0_d1, = plt.plot(iter, l5_r0_d1, color='green', linewidth=1.0, label='Dilated_5')
# line_l5_r1_d0, = plt.plot(iter, l5_r1_d0, color='orange', linewidth=1.0, label='Residual_5')
# line_l5_r1_d1, = plt.plot(iter, l5_r1_d1, color='red', linewidth=1.0, label='Residual_Dilated_5')
#
line_l8_r0_d0, = plt.plot(iter, l8_r0_d0, color='green', linewidth=1.0, label='Regular_8')
# line_l8_r0_d1, = plt.plot(iter, l8_r0_d1, color='orange', linewidth=1.0, label='Dilated_8')
# line_l8_r1_d0, = plt.plot(iter, l8_r1_d0, color='red', linewidth=1.0, label='Residual_8')
# line_l8_r1_d1, = plt.plot(iter, l8_r1_d1, color='blue', linewidth=1.0, label='Residual_Dilated_8')

line_l10r0_d0, = plt.plot(iter, l10_r0_d0, color='orange', linewidth=1.0, label='Regular_10')
# line_l10_r0_d1, = plt.plot(iter, l10_r0_d1, color='red', linewidth=1.0, label='Dilated_10')
# line_l10_r1_d0, = plt.plot(iter, l10_r1_d0, color='blue', linewidth=1.0, label='Residual_10')
# line_l10_r1_d1, = plt.plot(iter, l10_r1_d1, color='green', linewidth=1.0, label='Residual_Dilated_10')

# line_l5_r0_d1, = plt.plot(iter, l5_r0_d1, color='black', linewidth=1.0, label='dilated_5_layers')
# line_l5_r1_d0, = plt.plot(iter, l5_r1_d0, color='green', linewidth=1.0, label='residual_5_layers')
# line_l5_r1_d1, = plt.plot(iter, l5_r1_d1, color='red', linewidth=1.0, label='dilated_residual_5')



plt.xlim((0, 100))
plt.ylim(0, 40)
plt.xlabel('Epoch')
plt.ylabel('PSNR(dB)')
plt.legend(loc='lower right')
plt.grid()
# matplotlib.rcParams['xtick.direction'] = 'in'
plt.show()