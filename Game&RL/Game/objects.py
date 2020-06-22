import numpy as np
import pygame
from ImageStore import playerImg, enemyImg, bulletImg, bolsovirusImg, comunavirusImg, bullet_sound, chargeImg
import random
import math

dfl = np.array([-1, -1])
charge_points = 50
charge_coord = np.array([10, 45])
HIGH_SCORE = 500


def tanh_reg(high_value, score):
    x = score / HIGH_SCORE
    y = math.tanh(x / 2)
    return high_value * y


class Spawner:
    def __init__(self):
        pass

    def create_enemy(self, score):
        """
        enemy = Twisted(speed=random.randint(1, 6), coord_center=np.array([random.randint(250, 550), 200]),
                        coord=np.array([400, random.randint(100, 300)]))
        """

        a = random.randint(0, 2)

        if a % 3 == 0:
            enemy = Invader(coord=np.array([random.randint(100, 700), random.randint(0, 100)]),
                            speed=random.randint(1, 1 + math.floor(tanh_reg(50, score))),
                            direction=random.randint(0, 1))
        elif a % 3 == 1:
            enemy = Invader2(coord=np.array([random.randint(100, 700), random.randint(0, 100)]),
                             speed=random.randint(1, 1 + math.floor(tanh_reg(50 / 3, score))),
                             direction=random.randint(0, 1))
        else:
            enemy = Invader3(coord=np.array([random.randint(100, 700), random.randint(0, 100)]),
                             speed=random.randint(1, 1 + math.floor(tanh_reg(50 / 12, score))))

        return enemy


class Object:
    def __init__(self, coord, image):
        """
        :arg coord: coordenadas em relacao ao canto superior esquerdo.
        :type coord: 2D numpy array
        """
        self.coord = coord
        self.image = image

    def set_coord(self, coord):
        self.coord = coord


class Mobile(Object):
    def __init__(self, coord, image):
        super().__init__(coord, image)

    def move_linear(self, v):
        """

        :param v: 2D numpy array
        :return: void
        """
        self.coord += v

    def move_angular(self, v, coord_center):
        """
        :param w: float which defines angular speed
        (positive w mean a clockwise rotation)
        :param coord_center: 2D numpy array which defines the rotation center
        :return: void
        """
        L = self.coord
        R = self.coord - coord_center
        r = np.linalg.norm(R)
        versor_v = np.array([-R[1], R[0]]) / r
        L = L + v * versor_v

        nR = L - coord_center
        versor_nR = nR / np.linalg.norm(nR)
        self.coord = coord_center + versor_nR * r


class Player(Mobile):
    def __init__(self, coord, speed=0, image=playerImg, screen=None):
        super().__init__(coord, image)
        '''
        :param player_r: flag to move to the right
        :param player_l: flag to mov to the left
        '''
        self.screen = screen
        self.bullet_limit = 5
        self.player_r = 0
        self.player_l = 0
        self.speed = speed
        self.bullet_shoot = []
        self.score_base1 = 0
        self.curr_score = 0
        self.charge = 0
        self.commands_down = {
            pygame.K_a: self.key_dleft,
            pygame.K_d: self.key_dright,
            pygame.K_RETURN: self.key_shoot,
            pygame.K_SPACE: self.key_alcohol_rain
        }

        self.commands_up = {
            pygame.K_a: self.key_uleft,
            pygame.K_d: self.key_uright
        }

    def read_key_down(self, event):
        cmd = self.commands_down.setdefault(event.key, self.Pass)
        cmd()

    def read_key_up(self, event):
        cmd = self.commands_up.setdefault(event.key, self.Pass)
        cmd()

    def key_dleft(self):
        self.player_l = 1
        self.player_r = 0

    def key_dright(self):
        self.player_r = 1
        self.player_l = 0

    def key_uleft(self):
        self.player_l = 0

    def key_uright(self):
        self.player_r = 0

    def key_shoot(self):
        if len(self.bullet_shoot) < self.bullet_limit:
            bullet_sound.play()
            engage_bullet = Bullet(np.copy(self.coord) + np.array([16, 10]))
            engage_bullet.fire(self)
            self.bullet_shoot.append(engage_bullet)

    def key_alcohol_rain(self):
        if self.charge:
            for x in range(1, 800 - 17, 32):
                self.bullet_shoot.append(Bullet(np.array([x, self.coord[1]])))
            self.charge = 0
            self.score_base1 = self.curr_score

    def Pass(self):
        pass

    def update(self, score):
        self.curr_score = score
        if score - self.score_base1 >= charge_points:
            self.charge = 1

        self.move()
        self.screen.blit(self.image, self.coord)
        for bullet in self.bullet_shoot:
            self.screen.blit(bullet.image, bullet.coord)

        if self.charge:
            self.screen.blit(chargeImg, charge_coord)

    def move(self):
        self.move_linear(self.speed * np.array([self.player_r - self.player_l, 0]))
        self.coord[0] = max(0, min(self.coord[0], 735))

        for bullet in self.bullet_shoot:
            bullet.move()
            if bullet.state == 'lost':
                self.bullet_shoot.pop(0)

    def status(self):
        stt = {
            "left": self.player_l,
            "right": self.player_r,
        }
        return stt


class Enemy(Mobile):
    def __init__(self, coord, image=enemyImg):
        if np.array_equal(coord, dfl):
            coord = np.array([random.randint(0, 800), random.randint(0, 600)])

        super().__init__(coord, image)


class Twisted(Enemy):
    def __init__(self, coord=dfl, coord_center=dfl, speed=0):
        super().__init__(coord)
        if np.array_equal(coord_center, dfl):
            coord_center = np.array([random.randint(0, 800), random.randint(0, 600)])

        self.coord_center = coord_center
        self.speed = speed

    def move(self):
        self.move_angular(self.speed, self.coord_center)


class Invader(Enemy):
    def __init__(self, coord=dfl, speed=0, direction=1):
        super().__init__(coord)

        self.speed = speed
        self.direction = direction
        self.command = {
            1: self.move_right,
            0: self.move_left
        }

    def move(self):
        cmd = self.command[self.direction]
        cmd()

    def move_left(self):
        if self.coord[0] <= 0:
            self.move_linear(np.array([0, 48]))
            self.direction = 1
        else:
            self.move_linear(np.array([-self.speed, 0]))

    def move_right(self):
        if self.coord[0] >= 799 - 64 - self.speed:
            self.move_linear(np.array([0, 48]))
            self.direction = 0
        else:
            self.move_linear(np.array([self.speed, 0]))


class Invader2(Enemy):
    def __init__(self, coord=dfl, speed=0, direction=1):
        super().__init__(coord, image=bolsovirusImg)

        self.speed = speed
        self.direction = direction
        self.borders = [0, 799 - self.speed - 64]
        self.command = {
            1: self.move_right,
            0: self.move_left
        }

    def move(self):
        cmd = self.command[self.direction]
        cmd()

    def move_left(self):
        if self.coord[0] <= self.borders[0]:
            self.move_linear(np.array([0, 48]))
            self.direction = 1
            self.borders[1] = random.randint(self.borders[0] + 1, 799 - self.speed - 64)
        else:
            self.move_linear(np.array([-self.speed, 0]))

    def move_right(self):
        if self.coord[0] >= self.borders[1]:
            self.move_linear(np.array([0, 48]))
            self.direction = 0
            self.borders[0] = random.randint(0, self.borders[1] - 1)
        else:
            self.move_linear(np.array([self.speed, 0]))


class Invader3(Enemy):
    def __init__(self, coord=dfl, speed=0):
        super().__init__(coord, image=comunavirusImg)

        self.speed = speed

    def move(self):
        self.move_linear(np.array([0, self.speed]))


class Bullet(Mobile):

    def __init__(self, coord, image=bulletImg, speed=15):
        super(Bullet, self).__init__(coord, image)

        self.state = 'ready'
        self.source_coord = np.copy(coord)
        self.speed = speed

    def fire(self, shooter):
        self.state = 'fire'

    def move(self):
        self.move_linear(np.array([0, -self.speed]))
        r = np.linalg.norm(self.coord - self.source_coord)
        if r > 480:
            self.state = 'lost'
