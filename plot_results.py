import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


n_groups = 4

mae = (0.021, 0.02, 0.038, 0.013)
#mae_norm = (0.32 , 0.28, 1, 0. )
#mae_norm = (0.42391815, 0.40373158, 0.76708999, 0.26242552)
mae_norm = (55.26314, 52.63156, 100, 34.2105)


mse = (0.0029, 0.0028, 0.008, 0.0015)
#mse_norm = (0.21538462,0.2, 1, 0)        
mse_norm = (36.25,35, 100, 18.75)        
#disaggregation_error = (0.095, 0.09, 0.004, 0.0038)

disaggreation_accuracy = (0.7174, 0.7434, 0.4157, 0.9044)
#acc_norm = (0.61735216,  0.67055453,  0., 1.)
acc_norm = (79.32171066,  82.19676693,  45.96, 100)


plots = [mse, mae, disaggreation_accuracy]
plots2 = [mse_norm, mae_norm, acc_norm]

plots_name = ['MSE', 'MAE', 'Disag Acc']

index = np.arange(n_groups)
bar_width = 0.35


opacity = 0.4
error_config = {'ecolor': '0.3'}

for i,plot in enumerate(plots):
	color = 'tab:blue'
	ax = plt.subplot(1,3,i+1)
	plt.bar(index, plot, bar_width,
	                alpha=opacity, color='b',
	                error_kw=error_config)

	ax.set_xticks(index)
	ax.set_ylabel(plots_name[i], color=color)
	ax.set_xticklabels(('LSTM-k', 'CONV-s', 'CONV-p', 'proposed'), rotation=80)
	ax.legend()
	
	color = 'tab:red'
	ax2 = ax.twinx()
	ax2.set_ylabel('{} normalized'.format(plots_name[i]), color=color)
	plt.bar(index + 0.35, plots2[i], bar_width, alpha=opacity, color='r',
	                error_kw=error_config)


plt.tight_layout()
plt.show()
