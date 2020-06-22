import os
import pygame
import numpy as np
from buffer import ILBuffer
from StateMachine import FiniteStateMachine
from StateMachine import RunPlay

# main settings
extraction_i = 200
taps_per_sec = 7
NN_pack_path = '../Neural_Network'
extract_data = True
SAVE = extract_data
LOAD = extract_data
FILE_CODE = 345
screen_shape = (800, 600)


# translator between a pxa array 2D to a colorRGB 3D
def color_translator(input):
    # if pxa
    if input.ndim == 2:
        output = np.zeros((screen_shape[0], screen_shape[1], 3))
        red = (input // 256 ** 2)
        green = (input - red * 256 ** 2) // 256
        blue = input - red * 256 ** 2 - green * 256
        output[:, :, 0] = red
        output[:, :, 1] = green
        output[:, :, 2] = blue
        output = np.array(output, dtype=np.uint8)
        return output

    # if colorRGB
    if input.ndim == 3:
        input = input.astype(np.float32)
        output = input[:, :, 0] * 256 ** 2 + input[:, :, 1] * 256 + input[:, :, 2]
        return output


# loading or preparing data
buffer = ILBuffer()
if os.path.exists(NN_pack_path + '/actions_{}.npy'.format(FILE_CODE)) and \
        os.path.exists(NN_pack_path + '/states_{}.npy'.format(FILE_CODE)) and\
        LOAD:
    buffer.il_load(FILE_CODE)
    print('loaded data')
else:
    print('new data')
# initialize pygame
pygame.init()
# create a screen
screen = pygame.display.set_mode(screen_shape)
# Title and Icon
pygame.display.set_caption("Virus Invader")
icon = pygame.image.load('image/ufo.png')  # take a icon and load
# (use www.flaticon.com to download icons in 32xp)d
pygame.display.set_icon(icon)
# init fsm
init_state = RunPlay(screen)
fsm = FiniteStateMachine(init_state)
running = 1
# important to extraction
i = 0
color_array = []
# a, d, enter, space
target_array = [0, 0, 0, 0]

while running:
    fsm.update()
    if fsm.state.state_name == "Game Over":
        i = 0

    if fsm.state.state_name == "Game Off":
        running = 0

    # -------------------------------------------------
    # NN extract
    if extract_data:
        if i >= extraction_i:
            pxa = np.array(pygame.PixelArray(screen))
            color_array.append(color_translator(pxa))
            # extracting action
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        target_array[2] = 1
                    if event.key == pygame.K_SPACE:
                        target_array[3] = 1

            stt = fsm.state.player.status()
            if stt["left"]:
                target_array[0] = stt["left"]
            if stt["right"]:
                target_array[1] = stt["right"]

            # sample_get
            if i % extraction_i >= 3:

                # translate in action index
                action = (target_array[0] * 2 ** 3
                          + target_array[1] * 2 ** 2
                          + target_array[2] * 2
                          + target_array[3])

                # concat the color_array
                state = np.concatenate(color_array, -1)

                # sample extraction
                buffer.il_append((state, action))

                # reset to the next extraction
                color_array = []
                i = 0
                target_array = [0, 0, 0, 0]
                print(buffer.action_buffer.size())

    # ----------------------------------------------------
    i += 1

if SAVE:
    buffer.il_save(FILE_CODE)
    print('sample saved with code {}'.format(FILE_CODE))
else:
    print('no sample saved')

'''
86 frames per second(free fluid game)
43 frames per second(fluid game while extracting)
sample is 4 frames ago + action(holded buttons)
'''
