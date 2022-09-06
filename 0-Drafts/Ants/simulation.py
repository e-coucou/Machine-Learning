import numpy as np
import sys
from Ant import *
import time as tm
import pygame as ga
import pygame.gfxdraw as gfx
from math import *
from random import *
import quadtree as qt
import timeit

width = 800
height = 600
N = 100
DEC = 1
GRID =30

np_v1 = np.array([2.35,5.56])
np_v2 = np.array([-3.35,1.56])
v1 = [2.35,5.56]
v2 = [-3.35,1.56]
ga_v1 = ga.math.Vector2(2.35,5.56)
ga_v2 = ga.math.Vector2(-3.35,1.56)

def pure_add():
    return v1+v2

def numpy_add():
    return np.add(np_v1,np_v2)

def pygame_add():
    return ga_v1+ga_v2

#---------
def main() -> int:
    # init pygame avec la fenetre win
    ga.init()
    win = ga.display.set_mode([width,height])
    message = ga.freetype.SysFont("couriernew", 14, bold=False, italic=False)
    win.fill(0)
    # Essai
    n = 10000
    t1 = timeit.timeit(pure_add, number = n)
    print('Pure Python:', t1)
    t2 = timeit.timeit(numpy_add, number = n)
    print('Numpy:', t2)
    t3 = timeit.timeit(pygame_add, number = n)
    print('Pygame:', t3)
    # init de la boucle
    run = True
    mode=0 # en mode behavior
    mousse = False
    pause = False
    speed = 100
    affichage = 0
    nbGenese = 0

    #debug
    # pos=np.array([100,100])
    target = ga.math.Vector2(500,400)
    base = ga.math.Vector2(100,100)
    colonie = []
    for _ in range(N):
        colonie.append(Ant(base.x,base.y))
    Ant.surface = win
    Ant.width = width
    Ant.height = height
    Ant.grid = np.zeros((width,height),dtype=np.int64)
    Ant.phe_food = np.zeros((width,height),dtype=np.int64)
    Ant.phe_base = np.zeros((width,height),dtype=np.int64)
    ga.pixelcopy.surface_to_array(Ant.grid,win,'P')
    Ant.food.append(target)
    Ant.getFood = 10
    while run:
        if not(pause):
            win.fill(0)

    # Gestion interface
        for event in ga.event.get():
            if event.type == ga.QUIT:
                run = False
            if event.type == ga.MOUSEBUTTONUP:
                # p = ga.mouse.get_pos()
                x,y = ga.mouse.get_pos()
                # Ant.phe_base[int(x),int(y)] += 255
                # gfx.pixel(win,x,y,(255,255,255))
                Ant.food.append(ga.math.Vector2(x,y))
                # # for f in colonie:
                # #     f.newSet(target)
            if event.type == ga.KEYDOWN:
                if event.key == ga.K_ESCAPE:
                    run = False
                if event.key == ga.K_SPACE:
                    affichage = (affichage + 1) % 4
                    # f = colonie[0]
                    # choix=[]
                    # for i in range(3):
                    #     p=colonie[0].look[i]
                    #     a = int(p.x)
                    #     b = int(p.y)
                    #     w = 20
                    #     # print(int(p.x),int(p.y), Ant.PheroSq)
                    #     print('le val',a,b,w)
                    #     arr = (Ant.phe_base[a:(a+w),b:(b+20)])
                    #     n = np.sum(arr)
                    #     choix.append(n)
                    #     # print(arr) 
                    #     # print(n)
                    # print('choix:',np.argmax(choix),choix)
                    # print(Ant.phe_food[47:67][74:94])
                if event.unicode == 'n' or event.unicode == 'N':
                    # new set
                    # colonie = []
                    pos = ga.math.Vector2(randint(0,width),randint(0,height))
                    for _ in range(N):
                        colonie.append(Ant(pos.x,pos.y))
                if event.unicode == '0':
                    mode = 0
                if event.unicode == '1':
                    mode = 1
                if event.unicode == '2':
                    mode = 2
                if event.unicode == '<':
                    speed -= 10
                    print(speed)
                if event.unicode == '>':
                    speed += 10
                if event.unicode == 'd' or event.unicode == 'D':
                    Ant.debug = not Ant.debug
                if event.unicode == 'a' or event.unicode == 'A':
                    Ant.anim = not Ant.anim
                if event.unicode == 'p':
                    Ant.wanderProjection -= 0.1
                if event.unicode == 'P':
                    Ant.wanderProjection += 0.1
                if event.unicode == 'r':
                    Ant.wanderRadius -= 1
                if event.unicode == 'R':
                    Ant.wanderRadius += 1
                if event.unicode == 'f':
                    Ant.maxForce -= 0.1
                if event.unicode == 'F':
                    Ant.maxForce += 0.1
                if event.unicode == 's':
                    Ant.setMaxSpeed( -0.1)
                if event.unicode == 'S':
                    Ant.setMaxSpeed( 0.1)

        start = tm.time()
        if affichage==0:
            Ant.grid = Ant.phe_food + 65536 * Ant.phe_base
        elif affichage==1:
            Ant.grid = Ant.phe_food
        elif affichage==2:
            Ant.grid =  65536 * Ant.phe_base
        else:
            Ant.grid = 0 * Ant.phe_food
        ga.surfarray.blit_array(win,Ant.grid)
        Ant.phe_food = (Ant.phe_food - 1).clip(0,255)
        Ant.phe_base = (Ant.phe_base - 1).clip(0,255)
        for f in Ant.food:
            gfx.filled_circle(win,int(f.x),int(f.y),5,(0,255,0))
        gfx.filled_circle(win,int(base.x),int(base.y),6,(255,0,0))
        nF = len(colonie)
        for i in range(nF-1,-1,-1):
            f = colonie[i]
            if mode == 0:
                f.Behavior()
            elif mode == 1:
                f.Force(f.Wander())
            # f.Force(f.Seek(target))
            elif mode == 2:
                f.Force(f.Arrive(target))
            f.Update()
            # f.Edge()
            # f.Behavior()
            # f.HandleBase()
            f.Show()
            if f.aLive():
                colonie.pop(i)
            if f.getFood:
                f.getFood = 0
                nbGenese += 1
        if (nbGenese >= 20 and nF <=500):
            nbGenese = 0
            colonie.append(Ant(f.base.x,f.base.y))
        end = tm.time()
        delta = round((end-start)*1000)
        mx = width-150
        mess_nour = 'Speed = ' + str(Ant.maxSpeed)
        message.render_to(win, (mx, 10), mess_nour, 0xFFFFFF)
        mess_nour = 'Force = ' + str(Ant.maxForce)
        message.render_to(win, (mx, 22), mess_nour, 0xFFFFFF)
        mess_nour = 'Projection = ' + str(Ant.wanderProjection)
        message.render_to(win, (mx, 34), mess_nour, 0xFFFFFF)
        mess_nour = 'Radius = ' + str(Ant.wanderRadius)
        message.render_to(win, (mx, 46), mess_nour, 0xFFFFFF)

        mess_nour = 'time = ' + str(delta)
        message.render_to(win, (mx, 70), mess_nour, 0xFFFFFF)
        mess_nour = 'colonie = ' + str(len(colonie))
        message.render_to(win, (mx, 82), mess_nour, 0xFFFFFF)
        mess_nour = 'Génèse = ' + str(nbGenese)
        message.render_to(win, (mx, 94), mess_nour, 0xFFFFFF)
        ga.display.flip()
        if speed > 1:
            ga.time.wait(speed)
    return 0
##
if __name__ == '__main__':
    sys.exit(main())