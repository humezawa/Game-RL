import pygame
import numpy as np
import math
from objects import Player, Spawner
from game import Game
from ImageStore import background
import time

player_coord = np.array([400, 450])
spawn_time_0 = 1
font = pygame.font.Font('freesansbold.ttf', 32)
textX = 10
textY = 10
font_color = np.array([0, 255, 255])
breed_0 = 0.5
HIGH_SCORE = 500


def spawn_time(score):
    x = score / HIGH_SCORE
    y = math.tanh(x)
    return spawn_time_0 - 0.5 * y


def breed(score):
    x = score / HIGH_SCORE
    y = math.tanh(x / 2)
    return breed_0 + 0.5 * y


class FiniteStateMachine(object):
    """
    A finite state machine.
    """

    def __init__(self, state):
        self.state = state

    def change_state(self, new_state):
        self.state = new_state

    def update(self):
        self.state.execute(self)


class State(object):
    """
    Abstract state class.
    """

    def __init__(self, state_name):
        """
        Creates a state.

        :param state_name: the name of the state.
        :type state_name: str
        """
        self.state_name = state_name

    def execute(self, fsm):
        """
        Checks conditions and execute a state transition if needed.
        """
        raise NotImplementedError("This method is abstract and must be implemented in derived classes")


class RunPlay(State):
    def __init__(self, screen):
        super(RunPlay, self).__init__('Run Play')
        self.player = Player(coord=player_coord, speed=12, screen=screen)
        self.spawner = Spawner()
        self.game = game = Game(screen=screen, width=800, length=600)
        self.screen = screen
        self.time = time.time()
        self.score = 0
        # pygame.mixer.music.load("sound/bensound-punky.mp3")
        # pygame.mixer.music.play(-1)

    def execute(self, fsm):
        self.screen.blit(background, (0, 0))

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                self.player.read_key_down(event)

            if event.type == pygame.KEYUP:
                self.player.read_key_up(event)

            if event.type == pygame.QUIT:
                fsm.change_state(GameOff())

        self.player.update(self.score)

        # Spawn
        if time.time() - self.time >= spawn_time(self.score) and len(self.game.enemy_list) <= 32:
            self.time = time.time()
            for i in range(1, 2 + math.floor(breed(self.score) * len(self.game.enemy_list))):
                self.game.include_enemy(self.spawner.create_enemy(self.score))

        end = self.game.update_enemy()
        self.score = self.game.bullet_impact(self.player.bullet_shoot, self.score)

        score_frame = font.render("Score : " + str(self.score), True, font_color)
        self.screen.blit(score_frame, (textX, textY))

        if end:
            fsm.change_state(GameOver(self.screen))
            pass

        pygame.display.update()


class GameOver(State):
    def __init__(self, screen):
        super().__init__("Game Over")

        self.screen = screen
        self.font = pygame.font.Font('freesansbold.ttf', 45)
        self.font_coord = np.array([80, 280])

    def execute(self, fsm):
        self.screen.blit(background, (0, 0))

        keep = self.font.render("Press any button to continue", True, font_color)
        self.screen.blit(keep, self.font_coord)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                fsm.change_state(RunPlay(self.screen))

            if event.type == pygame.QUIT:
                fsm.change_state(GameOff())

        pygame.display.update()


class GameOff(State):
    def __init__(self):
        super().__init__('Game Off')

    def execute(self, fsm):
        pass
