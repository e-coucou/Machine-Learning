import pygame as ga
import numpy as np
import math
import sys
import time as tm
import random

import NN as nn
# import ga

width = 900
height = 600
N = 255

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
    # print('max Fitness',max,row)
    # print(row,sum(row))
    # birds.sort()

class Bird():
    surface  = None
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
        self.fitness = 0 #score normalisé
    
    def update(self):
        self.vel.y += self.gravity
        self.vel.y = self.vel.y * 10 /(max(10,abs(self.vel.y)))
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
            # print(action)
            # if action:
            # print(action[0], action[0,0])
            if (action[0,0]>action[0,1]):
                self.pick()
                # self.applyF((0,-4))
    def pick(self):
        self.vel.y += -12

    def show(self):
        ga.draw.circle(self.surface,self.color,self.pos,10)

class Pipe():
    surface = None
    inc = 5
    w= 55
    
    def __init__(self) -> None:
        gap = random.randint(150,250)
        h = random.randint(0,height-250)
        self.haut = ga.Rect(width-self.w,0,self.w,h)
        self.bas = ga.Rect(width-self.w,h+gap,self.w,height)
        self.color = ga.Color(255,255,255,25)

    def update(self):
        self.bas.move_ip(-self.inc,0)
        self.haut.move_ip(-self.inc,0)

    def edge(self,x):
        return (self.bas.left <= x)

    def hits(self,bird_):
        if (bird_.pos.x>self.bas.left and bird_.pos.x<self.bas.right) and (bird_.pos.y < self.haut.bottom or bird_.pos.y>self.bas.top):
            self.color = (0,255,255,120)
            return True
        return False

    def show(self):
        ga.draw.rect(self.surface, self.color,self.bas)
        ga.draw.rect(self.surface, self.color,self.haut)
        ga.draw.circle(self.surface, (255,0,0,128),(self.haut.left,self.haut.bottom),3)
        ga.draw.circle(self.surface, (255,0,0,128),(self.bas.left,self.bas.top),3)

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
        color_ = (255,i,0,120)
        birds.append(Bird(color_=color_))
    savedbirds = []
    pipes.append(Pipe())
    Pipe.surface = surface
    Bird.surface = surface

    count = 0
    vitesse = 1

    # #test
    a = np.random.randn(100)
    print(a)
    s = np.array(list(map(mutate,a)))
    print(s-a)

    print('init')
    a= nn.NNs_ep([5,8,2])
    b= nn.NNs_ep(a.layers)
    b.W = a.copy()
    b.mutate(mutate)
    print((b.W) - np.array(a.W))
    # c = a.copy()
    # d = np.vectorize(mutate)
    # print(d(c[0])-b.W[0])
    # b.mutate(mutate)
    # print(b.W[0].shape,len(b.W))
    # print(b.W[0])
    # print(a.W - b.W)
    # print(list(map(mutate,c[0])))
    # s0 = np.array(list(map(mutate,c[0][0])))
    # print(s0-c[0][0])


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
                if event.unicode == 'b':
                    print(Bird.best)
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