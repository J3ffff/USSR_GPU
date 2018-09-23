import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import csv
import random
rand = random.random()

result_path = './Result_possessed/figure1/'
iter = np.loadtxt(open(result_path + '3_0_0_100.csv'),delimiter=",",skiprows=0)[:,0]
l3_r0_d0 = np.loadtxt(open(result_path + '3_0_0_100.csv'),delimiter=",",skiprows=0)[:,1]
# l3_r0_d1 = np.loadtxt(open(result_path + '3_0_1.csv'),delimiter=",",skiprows=0)[:,1]
# l3_r1_d0 = np.loadtxt(open(result_path + '3_1_0.csv'),delimiter=",",skiprows=0)[:,1]
# l3_r1_d1 = np.loadtxt(open(result_path + '3_1_1.csv'),delimiter=",",skiprows=0)[:,1]

l5_r0_d0 = np.loadtxt(open(result_path + '5_0_0_100.csv'),delimiter=",",skiprows=0)[:,1]
# l5_r0_d1 = np.loadtxt(open(result_path + '5_0_1.csv'),delimiter=",",skiprows=0)[:,1]
# l5_r1_d0 = np.loadtxt(open(result_path + '5_1_0.csv'),delimiter=",",skiprows=0)[:,1]
# l5_r1_d1 = np.loadtxt(open(result_path + '5_1_1.csv'),delimiter=",",skiprows=0)[:,1]

l8_r0_d0 = np.loadtxt(open(result_path + '8_0_0_100.csv'),delimiter=",",skiprows=0)[:,1]
# l8_r0_d1 = np.loadtxt(open(result_path + '8_0_1_100.csv'),delimiter=",",skiprows=0)[:,1]
# l8_r1_d0 = np.loadtxt(open(result_path + '8_1_0_100.csv'),delimiter=",",skiprows=0)[:,1]
# l8_r1_d1 = np.loadtxt(open(result_path + '8_1_1_100.csv'),delimiter=",",skiprows=0)[:,1]

l10_r0_d0 = np.loadtxt(open(result_path + '10_0_0_100.csv'),delimiter=",",skiprows=0)[:,1]


plt.figure()
line_l3_r0_d0, = plt.plot(iter, l3_r0_d0, color='green', linewidth=1.0, label='Regular_3')
line_l5_r0_d0, = plt.plot(iter, l5_r0_d0, color='orange', linewidth=1.0, label='Regualr_5')
line_l8_r0_d0, = plt.plot(iter, l8_r0_d0, color='red', linewidth=1.0, label='Regualr_8')
line_l10_r0_d0, = plt.plot(iter, l10_r0_d0, color='blue', linewidth=1.0, label='Regular_10')
# line_l8_r0_d0, = plt.plot(iter, l8_r0_d0, color='red', linewidth=1.0, label='Regular_8')
# line_l8_r1_d0, = plt.plot(iter, l8_r1_d0, color='orange', linewidth=1.0, label='Residual_8')
# line_l8_r0_d1, = plt.plot(iter, l8_r0_d1, color='blue', linewidth=1.0, label='Dilated_8')
# line_l8_r1_d1, = plt.plot(iter, l8_r1_d1, color='green', linewidth=1.0, label='Residual_Dilated_8\n(Full model)')



plt.xlim((0, 100))
plt.ylim(10, 40)
ax=plt.gca()
# ax.set_yticklabels( ('10', '15', '20', '25', '30','31','32','33','34','35','36','37','38','39','40'))
plt.xlabel('Epoch')
plt.ylabel('PSNR(dB)')
plt.legend(loc='lower right')
plt.grid()
# matplotlib.rcParams['xtick.direction'] = 'in'
plt.show()