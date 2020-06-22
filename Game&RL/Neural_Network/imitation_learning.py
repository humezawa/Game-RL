import numpy as np
from cnn import CNN
import matplotlib.pyplot as plt

code = 2
file_name = 'IL_weights_no_over'

il_nn = CNN()

history = il_nn.train(3 * 121)

plt.plot(history[0])
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost Function Convergence')
plt.grid()
plt.savefig('convergence_{}.png'.format(code), format='png')

plt.figure()
plt.plot(history[1])
plt.xlabel('Epoch')
plt.ylabel('Val Acc')
plt.title('Validation Accuracy Convergence')
plt.grid()
plt.savefig('validation_{}.png'.format(code), format='png')

il_nn.save(file_name)

# history = il_nn.evaluate()
# print('evaluated_accuracy: {}'.format(history))

