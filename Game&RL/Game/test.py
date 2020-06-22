import random
import numpy as np


a = np.load('../Neural_Network/actions_345.npy')
s = np.load('../Neural_Network/states_345.npy')

np.save('../Neural_Network/actions_345.npy', a[0: 750])
np.save('../Neural_Network/states_345.npy', s[0: 750])
