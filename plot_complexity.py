import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

lstmk_time = [  26.82191825,   30.0008688,    45.28180909,   61.48509526,   72.8602612,
   89.20421529,  100.79808164,  114.50133777,  129.60741639,  142.65791416,
  158.22068977,  172.5333147,   186.67399859,  201.42430329,  215.59629464,
  229.39338589,  243.89349675,  259.19675708,  272.59702396]

lstmk_memory = [  9.36886272e+08,   1.90304256e+09,   2.95983104e+09,   3.95080499e+09,
   4.95112602e+09,   5.93902387e+09,   6.94019277e+09,   7.95335885e+09,
   8.93197107e+09,   9.92995328e+09,   1.08673761e+10,   1.18213304e+10,
   1.28185426e+10,   1.38263224e+10,   1.48138721e+10,   1.58026056e+10,
   1.67917896e+10,   1.77778524e+10,   1.87786486e+10]

lstmk_memory[:] = [x //(10**6) for x in lstmk_memory]

convz1_time = [8.50060654,   12.35954332,   18.80981755,   26.51598787,   33.00045085,
   39.23488021,   45.92251635,   52.59452367,   59.04503989,   65.23896742,
   71.65719485,   78.14174128,   84.73481107,   91.67581773,   97.20757842,
  106.02508235,  112.34493589,  118.23698854,  124.20333219]

convz1_memory = [  8.48474112e+08,   1.76457318e+09,   2.67801395e+09,   3.58447923e+09,
   4.48364544e+09,   5.38695270e+09,   6.28857242e+09,   7.19479194e+09,
   8.06432358e+09,   8.94336205e+09,   9.81870592e+09,   1.07136082e+10,
   1.16193812e+10,   1.25345587e+10,   1.34215762e+10,   1.43256125e+10,
   1.52226529e+10,   1.61344102e+10,   1.70151567e+10]

convz1_memory[:] = [x //(10**6) for x in convz1_memory]

convz2_time = [  4.91927385,   5.2031796,    8.14071178,  10.46887636,  12.75016189,
  15.46896434,  18.28139591,  21.95394802,  23.15653539,  26.56936979,
  29.49726868,  32.2347579,   34.92105818,  38.25694799,  42.09882355,
  44.59581065,  45.85227728, 50.23453307,  52.68801618]

convz2_memory = [  8.49141760e+08,   1.72514509e+09,   2.59405824e+09,   3.46281165e+09,
   4.32982426e+09,   5.20445542e+09,   6.07487181e+09,   6.94308454e+09,
   7.81427507e+09,   8.68505190e+09,   9.55578778e+09,   1.04289812e+10,
   1.12991396e+10,   1.21682616e+10,   1.29956413e+10,   1.37900892e+10,
   1.46544763e+10,   1.55285422e+10,   1.63988849e+10]

convz2_memory[:] = [x //(10**6) for x in convz2_memory]
'''
gru_time = [ 217.31313157,  214.76046324,  214.43725657,  214.80735064,  214.77602434,
  214.49478865,  215.55737352,  215.04169154,  214.60414672,  214.79168534,
  214.54164529,  215.02609897,  215.13544083,  214.71523523,  214.75592327,
  214.86979079,  214.9167068,   215.54169536,  214.97919416]
'''
'''
gru_time = [ 16.05109191,  12.51581192,  12.2657876,   12.40634823,  12.7563529,
  12.97329879,  12.66847301,  12.70329475,  13.07625318,  13.04816175,
  13.35834336,  12.67765903,  12.63993478,  12.70707583,  12.90268397,
  13.40738273,  12.67208004,  13.01577163,  12.82825184]
'''
gru_time = [  28.16361895,   25.48470314,   26.46358079,   26.46324241,   28.14100108,
   29.34636409,   29.85737452,   32.25822997,   34.58426345,   36.10980144,
   36.57866422,   40.80127931,   43.83321728,   48.79751778,   55.43821983,
   66.2430613,    82.54796576,  116.79838324,  218.50287032]

'''
gru_memory = [  2.82923827e+09,   2.84532736e+09,   2.84815770e+09,   2.84869427e+09,
   2.85025485e+09,   2.85176218e+09,   2.84450816e+09,   2.84593766e+09,
   2.84641690e+09,   2.84565094e+09,   2.84653978e+09,   2.84599091e+09,
   2.84595405e+09,   2.84622029e+09,   2.84624077e+09,   2.84600320e+09,
   2.84625306e+09,   2.84651520e+09,   2.84599501e+09]
'''
gru_memory = [  1.81284842e+09,   1.83185590e+09,   1.83898522e+09,   1.84311194e+09,
   1.84841148e+09,   1.85426330e+09,   1.85214755e+09,   1.85891499e+09,
   1.86788659e+09,   1.87743027e+09,   1.88931231e+09,   1.90392730e+09,
   1.92450150e+09,   1.94978748e+09,   1.98583091e+09,   2.03995955e+09,
   2.12925099e+09,   2.30770278e+09,   2.84422554e+09]
gru_memory[:] = [x // (10**6) for x in gru_memory]

line_style = ['r--','bs','g^','yo']
labels = ['LSTM-k','CONV-s','CONV-p','proposed']
memories = [lstmk_memory, convz1_memory, convz2_memory, gru_memory]
times = [lstmk_time, convz1_time, convz2_time, gru_time]
opacity = 0.4

ax = plt.subplot(2,1,1)
for i,time in enumerate(times):
	ax.set_ylabel('Time (s)')
	plt.plot(time,line_style[i], label=labels[i], alpha=opacity)

plt.xticks(np.arange(0,19,1))
ax.set_xticklabels(np.arange(1,20,1))
ax = plt.subplot(2,1,2)
for i,memory in enumerate(memories):
	ax.set_ylabel('Memory Consuptiom (MB)')
	plt.plot(memory,line_style[i], label=labels[i], alpha=opacity)



plt.xlabel('Number of appliances')
plt.xticks(np.arange(0,19,1))
ax.set_xticklabels(np.arange(1,20,1))
plt.legend()
plt.show()