import random
import numpy as np
import math

STATES = 0
ACTIONS = 1


class Buffer:
    def __init__(self, shape, buffer_type=np.float32):
        # holds the loaded buffer(np)
        example = np.empty(shape)
        example = np.array([example], dtype=buffer_type)
        self.buffer = np.delete(example,
                                obj=0,
                                axis=0)# empty ndim buffer
        # holds the new buffer
        self.new_buffer = []

    def append(self, sample):
        self.new_buffer.append(sample)

    def mini_batch(self, batch_size):
        random.sample(population=self.buffer, k=batch_size)

    def size(self):
        return len(self.buffer) + len(self.new_buffer)

    def __commit(self):
        self.buffer = np.concatenate([self.buffer, self.new_buffer], axis=0)

    def save(self, file_name):
        self.__commit()
        data_path = '../Neural_Network/{}'.format(file_name)
        np.save(file=data_path, arr=self.buffer)

    def load(self, file_name):
        self.buffer = np.load('../Neural_Network/{}.npy'.format(file_name))


class ILBuffer:
    def __init__(self, test_split=0.8):
        self.state_buffer = Buffer((800, 600, 12), buffer_type=np.uint8)
        self.action_buffer = Buffer(1, buffer_type=np.uint8)
        self.test_split = test_split

    def il_append(self, sample):
        state, action = sample
        self.state_buffer.append(state)
        self.action_buffer.append([action])

    def il_save(self, file_code=None):
        if file_code:
            pass
        else:
            file_code = random.randint(100, 1000)
        self.state_buffer.save('states_{}'.format(file_code))
        self.action_buffer.save('actions_{}'.format(file_code))

    def il_load(self, file_code):
        self.state_buffer.load('states_{}'.format(file_code))
        self.action_buffer.load('actions_{}'.format(file_code))

    def il_mini_batch(self, batch_size, mod='none'):
        if mod == 'test':
            index = range(0, math.floor(len(self.action_buffer.buffer) * self.test_split))
        elif mod == 'validation':
            index = range(math.floor(len(self.action_buffer.buffer) * self.test_split),
                          math.floor(len(self.action_buffer.buffer)))
        else:
            index = range(0, math.floor(len(self.action_buffer.buffer)))
        sample_index = random.sample(index, batch_size)
        states_sample = []
        actions_sample = []
        for i in sample_index:
            states_sample.append(self.state_buffer.buffer[i])
            actions_sample.append(self.action_buffer.buffer[i])

        states_sample = np.array(states_sample, dtype=np.uint8)
        actions_sample = np.array(actions_sample, dtype=np.uint8)

        return states_sample, actions_sample

    def il_extended_mini_batch(self, batch_size, mod='none'):
        states_sample, actions_sample = self.il_mini_batch(batch_size, mod=mod)
        # moving matrix
        states_sample = move_2d(move=5, frame_list=states_sample)
        return states_sample, actions_sample


# takes ndarray like a list of n-1darray
# on each n-1darray like a "image", that moves random-ly the "images" on x and y axis
# the empty space left by the movement in filled with 0
def move_2d(move, frame_list):
    lim_x = frame_list.shape[1]
    lim_y = frame_list.shape[2]
    i = random.randint(- move, move)
    if i >= 0:
        frame_list[:, i: lim_x] = frame_list[:, 0: lim_x - i]
        frame_list[:, 0: i].fill(0)

    else:
        frame_list[:, 0: lim_x + i] = frame_list[:, -i: lim_x]
        frame_list[:, lim_x + i: lim_x].fill(0)

    j = random.randint(- move, move)
    if j >= 0:
        frame_list[:, :, j: lim_y] = frame_list[:, :, 0: lim_y - j]
        frame_list[:, :, 0: j].fill(0)

    else:
        frame_list[:, :, 0: lim_y + j] = frame_list[:, :, -j: lim_y]
        frame_list[:, :, lim_y + j: lim_y].fill(0)

    return frame_list
