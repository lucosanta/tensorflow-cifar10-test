
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('run_tcomparison-accuracy.csv', delimiter=',', names=['step', 'test', 'train'])
# plt.plot(data['step'][1:], data['train'][1:])
# plt.plot(data['step'][1:], data['test'][1:])
line_up, = plt.plot(data['train'][1:], label='Train Accuracy')
line_down, = plt.plot(data['test'][1:], label='Test Accuracy')
plt.legend(handles=[line_up, line_down])

plt.show()