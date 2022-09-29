import pygame as ga
import pygame.gfxdraw as gfx
from pygame import Vector2 as Vc
import numpy as np
import math
import sys
import time as tm
import random
from json import JSONEncoder
import json
from collections import deque

from scipy.__config__ import show

import NN as nn
# import ga

width = 400
height = 600
N = 255
LIMIT = 10
INC = 5
B_X = 40
B_Y = 20
VITESSE = 2

REPLAY_MEMORY_SIZE = 50_000

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# def picOne(birds):
#     index = 0
#     r =random.random()
#     while r>0:
#         r -= birds[index].fitness
#         index += 1
#     index -= 1
#     bird = birds[index]
#     child = Bird(brain_=bird.brain,color_=bird.color)
#     return child

# def mutate(x):
#     if (random.random() < 0.1):
#         offset = random.normalvariate(0,1) *0.1
#         newx = x + offset
#         return newx
#     else:
#         return x

# def nextGen(birds):
#     fitness(birds)
#     newbirds=[]
#     for i in range(len(birds)):
#         newbirds.append(picOne(birds))
#     return newbirds

# def fitness(birds):
#     cumul = 0
#     row=[]
#     for bird in birds:
#         a=bird.score
#         bird.score = pow(a,2)
#         cumul += bird.score
#         if a>Bird.highScore:
#             Bird.highScore=a
#             Bird.best = bird.brain.copy()

#     for bird in birds:
#         bird.fitness = bird.score/cumul
#         row.append(bird.fitness)
#         if Bird.maxFitness<bird.fitness:
#             Bird.maxFitness = bird.fitness

class Balle():
    surface = ga.Surface((10,10))
    def __init__(self) -> None:
        self.pos = Vc((width//2,height//2))
        self.vel = Vc((random.random()*3,random.random()*3)) #.from_polar((1,random.randint(0,360)))
        self.vel.from_polar((random.random()*1.5+VITESSE,random.randint(45,135)))
        self.color = ga.Color(255,255,255,255)

    def update(self):
        self.pos += self.vel

    def edge(self):
        if self.pos.x <0 or self.pos.x > width :
            self.vel.x *= -1
        if self.pos.y <0 :
            self.vel.y *= -1
    
    def gameOver(self):
        return (self.pos.y > height)

    def hit(self,player_):
        if (self.pos.x > player_.racket.left and self.pos.x < player_.racket.right and self.pos.y > player_.racket.top):
            self.vel.y *= -1
            player_.score += 1

    def show(self):
        gfx.filled_circle(self.surface,int(self.pos.x),int(self.pos.y),5,self.color)

class Brick():
    surface = ga.Surface((10,10))
    w = 20
    h = 10

    def __init__(self,pos_,color_):
        self.pos= Vc(pos_)
        self.color = color_
        self.rect = ga.Rect(self.pos.x-self.w-1,self.pos.y-self.h-1,self.w*2-2,self.h*2-2)

    def hit(self,ball_):
        return ga.Rect.collidepoint(self.rect,ball_.pos.x,ball_.pos.y)

    def show(self):
        gfx.box(self.surface,self.rect,self.color)


class DQNAgent():
    def __init__(self) -> None:
        self.actor = nn.NNs_ep([6,10,3])
        brain_ = self.actor.copy()
        self.critic = nn.NNs_ep([6,10,3])
        self.critic.W = brain_
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.target_update_counter = 0        

class Player():
    surface  = ga.Surface((10,10))
    highScore=0
    maxFitness = 0
    best = None
    w = 25
    h = 5
    valAction = [-1,0,1]

    def __init__(self,color_=(255,255,255,255),brain_=None):
        self.pos = Vc((width/2,height-self.w * 2))
        self.vel = Vc((0,0))
        self.acc = Vc(0,0)
        self.racket = ga.Rect(self.pos.x-self.w,height-self.h,self.w*2,self.h*2)
        self.color = color_
        # if brain_:
        #     self.brain = nn.NNs_ep(brain_.layers)
        #     self.brain.W = brain_.copy()
        #     self.brain.mutate(mutate)
        # else:
        self.brain = nn.NNs_ep([6,10,3])
        self.score=0
        self.fitness = 0
    
    def update(self):
        self.vel.x += self.acc.x
        self.vel.x = self.vel.x * LIMIT /(max(LIMIT,abs(self.vel.x)))
        self.pos.x += self.vel.x
        self.acc.x = 0
        self.vel.x *= 0.95
        if self.pos.x < self.w//2 :
            self.pos.x = self.w//2
        if self.pos.x > width - self.w//2 :
            self.pos.x = width - self.w//2
        self.racket = ga.Rect(self.pos.x-self.w,height-self.h,self.w*2,self.h*2)
            
    def think(self,ball_):
        input=np.zeros((6))
        input[0] = self.pos.x/width
        input[1] = self.vel.x/10.
        input[2] = ball_.pos.x/width
        input[3] = ball_.pos.y/height
        input[4] = ball_.vel.x / 10.
        input[5] = ball_.vel.y / 10.
        action = self.brain.predict(input)
        choix = np.argmax(action[0])
        print(action, choix)
        self.applyForce(self.valAction[choix])

    def applyForce(self,force):
        self.acc.x += force

    def show(self):
        gfx.box(self.surface,self.racket,self.color)

#---------
def main() -> int:
    # init pygame avec la fenetre win
    run = True
    cycle = 0
    ga.init()
    win = ga.display.set_mode([width,height])
    surface= ga.Surface([width,height],ga.SRCALPHA)
    message = ga.freetype.SysFont("couriernew", 14, bold=False, italic=False)
    win.fill(0)
    player = Player()
    balle = Balle()
    bricks = []
    Balle.surface = surface
    Player.surface = surface
    Brick.surface = surface
    Brick.w = B_X//2
    Brick.h = B_Y//2
    for x in range(10):
        for y in range(5):
            bricks.append(Brick((x*B_X+B_X//2,y*B_Y+120),ga.Color(x*25,120,50*y,255)))

    count = 0
    vitesse = 1
    #---
    # Boucle principale
    while run:
        # Gestion interface
        for event in ga.event.get():
            if event.type == ga.QUIT:
                run = False
            if event.type == ga.MOUSEBUTTONUP:
                x,y = ga.mouse.get_pos()
            if event.type == ga.KEYDOWN:
                if event.key == ga.K_ESCAPE:
                    run = False
                if event.key == ga.K_LEFT:
                    player.applyForce(-INC)
                if event.key == ga.K_RIGHT:
                    player.applyForce(INC)
                if event.unicode == 'l':
                    print('load player')
                    with open("best_player.json", "r") as read_file:
                        decodedArray = json.load(read_file)
                        player.brain.W = np.asarray(decodedArray)

                if event.unicode == 'b':
                    # print(Bird.best)
                    print('save Best Player')
                    with open("best_player.json", "w") as write_file:
                        json.dump(player.best, write_file, cls=NumpyArrayEncoder)

                if event.unicode == 's':
                    if vitesse==100:
                        vitesse = 1
                    else:
                        vitesse = 100
        cycle += 1
        start = tm.time()
        # on update
        balle.edge()
        balle.hit(player)
        balle.update()
        player.think(balle)
        player.update()
        for i in range(len(bricks)-1,-1,-1):
            if bricks[i].hit(balle):
                bricks.pop(i)
                player.highScore += 10
                balle.vel.from_polar((random.random()*1.7+VITESSE,random.randint(0,360)))


        if balle.gameOver():
            balle=Balle()
        
        if count%vitesse ==0 :
            # win.fill((0,0,0,128))
            surface.fill((0,0,0,128))
            balle.show()
            player.show()
            for b in bricks:
                b.show()
        count += 1

        #-----------
        end = tm.time()
        delta = round((end-start)*1000)
        mess_nour = 'time = ' + str(delta)
        message.render_to(win, (width-150, 20), mess_nour, (255,255,255,255))
        mess_nour = 'HighScore = ' + str(player.highScore)
        message.render_to(win, (width-150, 60), mess_nour, (255,255,255,255))
        mess_nour = 'Max Fitness = ' + str(player.maxFitness)
        message.render_to(win, (width-150, 80), mess_nour, (255,255,255,255))
        win.blit(surface,(0,0))
        ga.display.flip()
        # ga.time.wait(20)
    return 0
##
if __name__ == '__main__':
    sys.exit(main())