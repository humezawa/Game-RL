import pygame

background = pygame.image.load('image/barbershop.jpg')
playerImg = pygame.image.load('image/player.png')
enemyImg = pygame.image.load('image/enemy.png')
bulletImg = pygame.image.load('image/bullet.png')
bolsovirusImg = pygame.image.load('image/bolsovirus.png')
comunavirusImg = pygame.image.load('image/comunavirus.png')
chargeImg = pygame.image.load('image/natureza.png')

pygame.init()
bullet_hit = pygame.mixer.Sound('sound/waterhit.wav')
bullet_sound = pygame.mixer.Sound('sound/watershoot.wav')
monster_death = pygame.mixer.Sound('sound/monsterdie.wav')
