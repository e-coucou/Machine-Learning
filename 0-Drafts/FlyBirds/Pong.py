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

import NN as nn
# import ga

width = 600
height = 600
border = 100
N = 255
LIMIT = 10
INC = 3
B_X = 40
B_Y = 20
VITESSE = 2
CHANCE_GAIN = 0.35

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
        self.pos = Vc((width//2,(height+border)//2))
        self.vel = Vc((random.random()*3,random.random()*3)) #.from_polar((1,random.randint(0,360)))
        self.vel.from_polar((random.random()*1.5+VITESSE,random.randint(135,215)))
        self.color = ga.Color(255,255,255,255)

    def update(self):
        self.pos += self.vel

    def edge(self):
        # if self.pos.x <0 or self.pos.x > width :
        #     self.vel.x *= -1
        if self.pos.y <border or self.pos.y > height:
            self.vel.y *= -1
    
    def gameOver(self):
        r=0
        if (self.pos.x > width):
            r=-1
        elif (self.pos.x <0):
            r=1
        return r

    def hit(self,player_):
        if ga.Rect.collidepoint(player_.racket,self.pos.x,self.pos.y):
            self.vel.x *= -1
            if player_.side == 1:
                self.pos.x -=5
            else:
                self.pos.x += 5
            player_.score += 1

    def show(self):
        gfx.filled_circle(self.surface,int(self.pos.x),int(self.pos.y),5,self.color)


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
    w = 5
    h = 25
    valAction = [-1,0,1]

    def __init__(self,side_=1,color_=(255,255,255,255),brain_=None):
        self.side = side_
        if side_==1:
            self.pos = Vc((width-50,(height+border)//2))
        else:
            self.pos = Vc((50,(height+border)//2))
        self.vel = Vc((0,0))
        self.acc = Vc(0,0)
        self.racket = ga.Rect(self.pos.x-self.w,self.pos.y-self.h,self.w*2,self.h*2)
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
        self.vel.y += self.acc.y
        self.vel.y = self.vel.y * LIMIT /(max(LIMIT,abs(self.vel.y)))
        self.pos.y += self.vel.y
        self.acc.y = 0
        self.vel.y *= 0.95
        if self.pos.y < self.h//2+border :
            self.pos.y = self.h//2+border
        if self.pos.y > (height) - self.h//2 :
            self.pos.y = (height) - self.h//2
        self.racket = ga.Rect(self.pos.x-self.w,self.pos.y-self.h,self.w*2,self.h*2)
            
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
        self.acc.y += force

    def auto(self,ball_):
        if random.random()<CHANCE_GAIN:
            if self.pos.y > ball_.pos.y:
                self.applyForce(-INC)
            else:
                self.applyForce(INC)

    def show(self):
        gfx.box(self.surface,self.racket,self.color)

class Blob():
    def __init__(self) -> None:
        self.balle = Balle()
        self.player = Player(side_=1)
        self.computer = Player(side_=0,color_=ga.Color(255,25,25,255))

    def play(self):
        done = False
        self.balle.edge()
        self.balle.hit(self.player)
        self.balle.hit(self.computer)
        self.balle.update()

        self.player.update()

        self.computer.auto(self.balle)
        self.computer.update()

        reward = self.balle.gameOver()
        Player.highScore += reward

        if reward!=0:
            self.balle=Balle()
            done = True

        obs=np.zeros((6))
        obs[0] = self.player.pos.x/width
        obs[1] = self.player.vel.x/10.
        obs[2] = self.balle.pos.x/width
        obs[3] = self.balle.pos.y/height
        obs[4] = self.balle.vel.x / 10.
        obs[5] = self.balle.vel.y / 10.
        
        return obs, reward, done,'ok'

    def render(self):
        self.balle.show()
        self.player.show()
        self.computer.show()


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
    Balle.surface = surface
    Player.surface = surface

    blob = Blob()

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
                if event.key == ga.K_UP:
                    blob.player.applyForce(-INC)
                if event.key == ga.K_DOWN:
                    blob.player.applyForce(INC)
                if event.unicode == 'l':
                    print('load player')
                    with open("best_player.json", "r") as read_file:
                        decodedArray = json.load(read_file)
                        blob.player.brain.W = np.asarray(decodedArray)

                if event.unicode == 'b':
                    # print(Bird.best)
                    print('save Best Player')
                    with open("best_player.json", "w") as write_file:
                        json.dump(blob.player.best, write_file, cls=NumpyArrayEncoder)

                if event.unicode == 's':
                    if vitesse==100:
                        vitesse = 1
                    else:
                        vitesse = 100
        cycle += 1
        start = tm.time()
        # on update
        obs, r, done, info = blob.play()
        if done:
            print('GameOver')
            print(obs)
            print(r)
        if count%vitesse ==0 :
            # win.fill((0,0,0,128))
            surface.fill((0,0,0,128))
            ga.draw.line(surface,ga.Color(180,180,180,255),(0,border-B_Y//2),(width,border-B_Y//2),B_Y//2)
            blob.render()
        count += 1

        #-----------
        end = tm.time()
        delta = round((end-start)*1000000)
        mess_nour = 'time = ' + str(delta)
        message.render_to(win, (width-150, 20), mess_nour, (255,255,255,255))
        mess_nour = 'HighScore = ' + str(blob.player.highScore)
        message.render_to(win, (width-150, 60), mess_nour, (255,255,255,255))
        mess_nour = 'Max Fitness = ' + str(blob.player.maxFitness)
        message.render_to(win, (width-150, 80), mess_nour, (255,255,255,255))
        win.blit(surface,(0,0))
        ga.display.flip()
        # ga.time.wait(20)
    return 0
##
if __name__ == '__main__':
    sys.exit(main())