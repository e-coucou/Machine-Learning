import pygame as ga
import pygame.gfxdraw as gfx
import numpy as np
import math
import sys
import time as tm
import random
from json import JSONEncoder
import json

import NN as nn
# import ga

width = 900
height = 600
N = 255
LIMIT = 11

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def picOne(birds):
    index = 0
    r =random.random()
    while r>0:
        r -= birds[index].fitness
        index += 1
    index -= 1
    bird = birds[index]
    child = Bird(brain_=bird.brain,color_=bird.color)
    return child

def mutate(x):
    if (random.random() < 0.1):
        offset = random.normalvariate(0,1) *0.1
        newx = x + offset
        return newx
    else:
        return x

def nextGen(birds):
    fitness(birds)
    newbirds=[]
    for i in range(len(birds)):
        newbirds.append(picOne(birds))
    return newbirds

def fitness(birds):
    cumul = 0
    row=[]
    for bird in birds:
        a=bird.score
        bird.score = pow(a,2)
        cumul += bird.score
        if a>Bird.highScore:
            Bird.highScore=a
            Bird.best = bird.brain.copy()

    for bird in birds:
        bird.fitness = bird.score/cumul
        row.append(bird.fitness)
        if Bird.maxFitness<bird.fitness:
            Bird.maxFitness = bird.fitness

class Bird():
    surface  = ga.Surface((10,10))
    highScore=0
    maxFitness = 0
    best = None

    def __init__(self,color_=(255,255,255,255),brain_=None):
        self.pos = ga.math.Vector2((100,height//2))
        self.vel = ga.math.Vector2((0,0.))
        self.color = color_
        self.gravity = 0.8
        if brain_:
            self.brain = nn.NNs_ep(brain_.layers)
            self.brain.W = brain_.copy()
            self.brain.mutate(mutate)
        else:
            self.brain = nn.NNs_ep([5,8,2])
        self.score=0
        self.fitness = 0
    
    def update(self):
        self.vel.y += self.gravity
        self.vel.y = self.vel.y * LIMIT /(max(LIMIT,abs(self.vel.y)))
        self.pos.y += self.vel.y
        if self.pos.y <=0 :
            return True
        else:
            if self.pos.y>= height:
                self.vel.y = 0
                self.pos.y = height
            self.score += 1
            return False
            
    def think(self,pipes_):
        pipeOne = None
        distOne = 2000
        for p in pipes_:
            d = p.bas.left - self.pos.x
            if (d>0 and d<distOne):
                distOne = d
                pipeOne = p
        if pipeOne != None:
            pipeOne.color=(255,100,100,255)
            input=np.zeros((5))
            input[0] = self.pos.x/width
            input[1] = pipeOne.bas.left/width
            input[2] = pipeOne.bas.top/height
            input[3] = pipeOne.haut.bottom/height
            input[4] = self.vel.y / 10.
            action = self.brain.predict(input)
            if (action[0,0]>action[0,1]):
                self.pick()
    def pick(self):
        self.vel.y += -12

    def show(self):
        # ga.draw.circle(self.surface,self.color,self.pos,10)  # type: ignore
        gfx.filled_circle(self.surface, int(self.pos.x), int(self.pos.y), 10, self.color)

class Pipe():
    surface = ga.Surface((10,10))
    inc = 5
    w= 55
    
    def __init__(self) -> None:
        gap = random.randint(150,250)
        h = random.randint(0,height-250)
        self.haut = ga.Rect(width-self.w,0,self.w,h)
        self.bas = ga.Rect(width-self.w,h+gap,self.w,height)
        self.colorBase = ga.Color(255,255,255,200)
        self.color = ga.Color(255,255,255,25)

    def update(self):
        self.bas.move_ip(-self.inc,0)
        self.haut.move_ip(-self.inc,0)

    def edge(self,x):
        return (self.bas.left <= x)

    def hits(self,bird_):
        if (bird_.pos.x>self.bas.left and bird_.pos.x<self.bas.right) and (bird_.pos.y < self.haut.bottom or bird_.pos.y>self.bas.top):
            self.colorBase = (255,25,25,220)
            return True
        return False

    def show(self):
        gfx.box(self.surface,self.bas,self.colorBase)
        gfx.box(self.surface,self.haut,self.colorBase)
        # ga.draw.rect(self.surface, self.colorBase,self.bas)  # type: ignore
        # ga.draw.rect(self.surface, self.colorBase,self.haut)  # type: ignore
        ga.draw.circle(self.surface, self.color,(self.haut.left,self.haut.bottom),5)  # type: ignore
        ga.draw.circle(self.surface, self.color,(self.bas.left,self.bas.top),5)  # type: ignore
        gfx.vline(self.surface,self.bas.left,0,height,self.color)  # type: ignore

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
    pipes = []
    birds = []
    savedbirds =[]
    for i in range(N):
        color_ = (0,i%128+128,96+i%128,200)
        birds.append(Bird(color_=color_))
    savedbirds = []
    pipes.append(Pipe())
    Pipe.surface = surface  # type: ignore
    Bird.surface = surface  # type: ignore

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
                if event.unicode == ' ':
                    birds[0].pick()
                if event.unicode == 'l':
                    print('load bird')
                    with open("best_bird.json", "r") as read_file:
                        decodedArray = json.load(read_file)
                        birds[0].brain.W = np.asarray(decodedArray)

                if event.unicode == 'b':
                    # print(Bird.best)
                    print('save Best Bird')
                    with open("best_bird.json", "w") as write_file:
                        json.dump(Bird.best, write_file, cls=NumpyArrayEncoder)

                if event.unicode == 's':
                    if vitesse==100:
                        vitesse = 1
                    else:
                        vitesse = 100
        cycle += 1
        if cycle%90 == 0:
            pipes.append(Pipe())
        start = tm.time()
        # on update tous les Birds
        for i in range(len(birds)-1,-1,-1):
            birds[i].think(pipes)
            if birds[i].update():
                savedbirds.append(birds.pop(i))
                break
            for j in range(len(pipes)):
                if pipes[j].hits(birds[i]):
                    savedbirds.append(birds.pop(i))
                    break
                # break
        # on update tous les Pipes
        for i in range(len(pipes)-1,-1,-1):
            pipes[i].update()
            if pipes[i].edge(30):
                pipes.pop(i)
        
        if count%vitesse ==0 :
            # win.fill((0,0,0,128))
            surface.fill((0,0,0,128))
            for pipe in pipes:
                pipe.show()
            for bird in birds:
                bird.show()
        count += 1

        if len(birds)==0:
            birds = nextGen(savedbirds)
            savedbirds = []
            pipes=[]
        #-----------
        end = tm.time()
        delta = round((end-start)*1000)
        mess_nour = 'time = ' + str(delta)
        message.render_to(win, (width-150, 20), mess_nour, (255,255,255,255))
        mess_nour = 'Nb = ' + str(len(birds))
        message.render_to(win, (width-150, 40), mess_nour, (255,255,255,255))
        mess_nour = 'HighScore = ' + str(Bird.highScore)
        message.render_to(win, (width-150, 60), mess_nour, (255,255,255,255))
        mess_nour = 'Max Fitness = ' + str(Bird.maxFitness)
        message.render_to(win, (width-150, 80), mess_nour, (255,255,255,255))
        win.blit(surface,(0,0))
        ga.display.flip()
        # ga.time.wait(20)
    return 0
##
if __name__ == '__main__':
    sys.exit(main())