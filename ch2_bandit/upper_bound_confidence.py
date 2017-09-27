import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(100, 1e+3, num=50)
log_t = np.log(t)

varianceWhen1= np.sqrt(log_t / (t * 1/100))
varianceWhen40 = np.sqrt(log_t / (t * 40/100))
varianceWhen50 = np.sqrt(log_t / (t * 50/100))
varianceWhen99= np.sqrt(log_t / (t * 99/100))

plt.plot(t, varianceWhen1, label='1%')
plt.plot(t, varianceWhen40, label='40%')
plt.plot(t, varianceWhen50, label='50%')
plt.plot(t, varianceWhen99, label='99%')
plt.xlabel('Time step t')
plt.ylabel('Uncertainty')
plt.legend(title='Action selected percent')
plt.show()
plt.close()
