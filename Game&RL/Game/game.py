import objects
import numpy as np
import pygame
import math
from ImageStore import monster_death

class Game:
    def __init__(self, screen, width, length):
        self.en_loc = np.zeros((length, width))
        self.en_loc.fill(-1)
        self.enemy_list = []
        self.ally_bullet = []
        self.screen = screen

    def include_enemy(self, enemy):
        if len(self.enemy_list) <= 20:
            self.enemy_list.append(enemy)

    def update_enemy(self):
        #move enemy
        end = 0
        for enemy in self.enemy_list:
            enemy.move()
            self.screen.blit(enemy.image, enemy.coord)
            if enemy.coord[1] >= 450:
                end = 1
                break

        #clean loc
        self.en_loc.fill(-1)
        self.update_loc()
        return end

    def update_loc(self):
        radius = 32
        a = math.floor(radius)
        for enemy in self.enemy_list:
            center = enemy.coord + np.array([radius, radius])
            #creates an attack square
            for i in range(int(center[0] - a), int(center[0] + a)):
                self.en_loc[int(center[1] - a), i] = self.enemy_list.index(enemy)
                self.en_loc[int(center[1] + a), i] = self.enemy_list.index(enemy)
            for j in range(int(center[1] - a), int(center[1] + a)):
                self.en_loc[j, int(center[0] - a)] = self.enemy_list.index(enemy)
                self.en_loc[j, int(center[0] + a)] = self.enemy_list.index(enemy)

    def bullet_impact(self, bullet_list, score):
        radius = 16
        a = radius
        number = []
        for bullet in bullet_list:
            center = bullet.coord + np.array([radius, radius])
            for j in range(int(center[1] - a), int(center[1] + a)):
                index = int(self.en_loc[j, int(center[0])])
                if index > -1 and number.count(index) == 0:
                    monster_death.play()
                    number.append(index)
                    bullet_list.remove(bullet)
                    break

        number = sorted(number, reverse=True)
        for i in number:
            self.enemy_list.pop(i)
            score += 1

        return score
