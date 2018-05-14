import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple


n_groups = 4

time = (175.99, 151.84, 47.2, 11.2)
time_norm = (100,86.3,26.8,6.364)

index = np.arange(n_groups)
bar_width = 0.35


opacity = 0.4
error_config = {'ecolor': '0.3'}

ax = plt.subplot(1,1,1)

plt.bar(index, time, bar_width,
                alpha=opacity, color='b',
                error_kw=error_config)

color = 'tab:blue'
ax.set_xlabel('Runtime')
ax.set_ylabel('Seconds')
ax.set_xticks(index + (0.35/2))
ax.set_xticklabels(('LSTM-k', 'CONV-s', 'CONV-p', 'proposed'))
ax.legend()

color = 'tab:red'
ax2 = ax.twinx()
ax2.set_ylabel('normalized', color=color)
plt.bar(index + 0.35, time_norm, bar_width,
                alpha=opacity, color='r',
                error_kw=error_config)

plt.tight_layout()
plt.show()
