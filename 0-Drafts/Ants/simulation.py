import numpy as np
import sys
from Ant import *
import time as tm
import pygame as ga
import pygame.gfxdraw as gfx
from math import *
from random import *
import quadtree as qt
from Maze import *
import timeit
from perlin_noise import PerlinNoise

width = 800
height = 800
N = 1
DEC = 1
GRID =70
LIMITE = 1000
WALL = (80,80,80,255)

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

class Colony():
    surface = None
    def __init__(self,x,y,color) -> None:
        self.pos = ga.math.Vector2(x,y)
        self.fourmi_list = []
        self.color = color
        for _ in range(N):
            self.fourmi_list.append(Ant(self.pos.x,self.pos.y,self.color))

    def Show(self,message):
        gfx.circle(self.surface,int(self.pos.x),int(self.pos.y),30,self.color)
        mess_nour = str(len(self.fourmi_list))
        # text_rect = message.get_rect(mess_nour, size = 12)
        # text_rect.center = self.surface.get_rect().center 
        message.render_to(self.surface,self.pos, mess_nour, 0xFFFFFFFF)


def randomCave(noise,res_x,res_y):
    field = np.zeros((res_x+1,res_y+1))
    offset=random.random()
    for x in range(res_x+1):
        for y in range(res_y+1):
            field[x,y] = noise([x/res_x/1000, y/res_y/1000,offset]) #randint(0,4)
            # if field[x,y] < seuil:
            #     gfx.filled_circle(fond,int(x*GRID),int(y*GRID),3,(255,255,255))
            # else:
            #     gfx.filled_circle(fond,int(x*GRID),int(y*GRID),3,(55,55,55))
            offset += 0.1
    print('field 0',field)
    return field

def printCave(field,res_x,res_y):
    # fond = ga.Surface([width,height],ga.SRCALPHA)
    win = ga.Surface((width,height))
    print('field2',field)
    print(win)
    for i in range(res_x):
        for j in range(res_y):
            x= i*GRID
            y = j*GRID
            l=field[i,j]+1 # intervalle [0,2] car noise de (-1,1)
            m=field[i+1,j]+1
            n=field[i+1,j+1]+1
            o=field[i,j+1]+1
            A = ga.math.Vector2(x,y)
            B = ga.math.Vector2(x+GRID,y)
            C = ga.math.Vector2(x+GRID,y+GRID)
            D = ga.math.Vector2(x,y+GRID)
            a = ga.math.Vector2(x+GRID*m/(l+m),y)
            b = ga.math.Vector2(x+ GRID,y+GRID*m/(n+m))
            c = ga.math.Vector2(x+GRID*n/(n+o),y+GRID)
            d = ga.math.Vector2(x,y+GRID*l/(l+o))
            id = getPid(field,i,j)
            if (id==14):
                ga.draw.polygon(win,(80,80,80,255),[c,d,(x,y+GRID)])
            elif (id==1):
                ga.draw.polygon(win,(80,80,80,255),[d,(x,y),(x+GRID,y),(x+GRID,y+GRID),c])
            elif (id==2):
                ga.draw.polygon(win,(80,80,80,255),[A,B,b,c,D])
            elif (id==13):
                ga.draw.polygon(win,(80,80,80,255),[b,C,c])
            elif (id==12):
                ga.draw.polygon(win,(80,80,80,255),[d,b,C,D])
            elif (id==3):
                ga.draw.polygon(win,(80,80,80,255),[d,b,B,A])
            elif (id==11):
                ga.draw.polygon(win,(80,80,80,255),[a,B,b])
            elif (id==4):
                ga.draw.polygon(win,(80,80,80,255),[a,b,C,D,A])
            elif (id==10):
                ga.draw.polygon(win,(80,80,80,255),[a,B,b])
                ga.draw.polygon(win,(80,80,80,255),[d,c,D])
            elif (id==6): #ok
                ga.draw.polygon(win,(80,80,80,255),[A,a,c,D])
            elif (id==9): #ok
                ga.draw.polygon(win,(80,80,80,255),[a,B,C,c])
            elif (id==7): #ok
                ga.draw.polygon(win,(80,80,80,255),[A,a,d])
            elif (id==8): #ok
                ga.draw.polygon(win,(80,80,80,255),[a,B,C,D,d])
            elif (id==5):
                ga.draw.polygon(win,(80,80,80,255),[a,d,A])
                ga.draw.polygon(win,(80,80,80,255),[b,c,C])
            elif (id==0):
                ga.draw.polygon(win,(80,80,80,255),[A,B,C,D])
            # if field[i,j] < seuil:
            #     gfx.filled_circle(fond,int(i*GRID),int(j*GRID),3,(255,255,255))
            # else:
            #     gfx.filled_circle(fond,int(i*GRID),int(j*GRID),3,(55,55,55))
    cadre = ga.Rect(0,0,width,height)
    ga.draw.rect(win,WALL,cadre,20)
    fond = np.zeros((width,height),dtype=np.int64)
    ga.display.flip()
    ga.pixelcopy.surface_to_array(fond,win,'P')
    return fond

def getPid(f,i,j):
    return 8*ceil(f[i,j])+4*ceil(f[i+1,j])+2*ceil(f[i+1,j+1])+ceil(f[i,j+1])
#---------
def main() -> int:
    # init pygame avec la fenetre win
    ga.init()
    noise = PerlinNoise(octaves=2, seed=1)
    res_x, res_y = (width//GRID, height//GRID)
    field = np.zeros((res_x+1,res_y+1))
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
    cycle = 0
    pause = False
    speed = 100
    affichage = 0
    nbGenese = 0

    foods = []
    Food.surface=win
    # fond d'écran marching squared
    noise = PerlinNoise(octaves=12, seed=1)

    #genartion du labyrinthe
    Maze.width =width
    Maze.height = height

    Ant.monde = np.zeros((width,height),np.int64)

    colonies = []
    Colony.surface = win
    Ant.surface = win
    Ant.width = width
    Ant.height = height
    Ant.wall = WALL
    Ant.grid = np.zeros((width,height),dtype=np.int64)
    Ant.phe_food = np.zeros((width,height),dtype=np.int64)
    Ant.phe_base = np.zeros((width,height),dtype=np.int64)
    ga.pixelcopy.surface_to_array(Ant.grid,win,'P')
    Ant.food = foods

    while run:
        # if not(pause):
        #     # win.fill(0)
        #     win.blit(fond,(0,0))
        # Gestion interface
        for event in ga.event.get():
            if event.type == ga.QUIT:
                run = False
            if event.type == ga.MOUSEBUTTONUP:
                # p = ga.mouse.get_pos()
                x,y = ga.mouse.get_pos()
                # Ant.phe_base[int(x),int(y)] += 255
                # gfx.pixel(win,x,y,(255,255,255))
                foods.append(Food(x,y))
                foods[len(foods)-1].pid = foods[len(foods)-1]
                # Ant.food.append(ga.math.Vector2(x,y))
                # # for f in colonie:
                # #     f.newSet(target)
            if event.type == ga.KEYDOWN:
                if event.key == ga.K_ESCAPE:
                    run = False
                if event.key == ga.K_SPACE:
                    affichage = (affichage + 1) % 4
                # new Colonie
                if event.unicode == 'n' or event.unicode == 'N':
                    x,y = ga.mouse.get_pos()
                    pos = ga.math.Vector2(x,y)
                    colonies.append(Colony(pos.x,pos.y,(randint(1,4)*63,randint(1,4)*63,randint(1,4)*63,255)))
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
                    Ant.anim = (Ant.anim + 1)%3
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
                if event.unicode == 'b' or event.unicode == 'B':
                    monde = randomCave(noise,res_x,res_y)
                    Ant.monde = np.zeros((width,height),np.int64)
                if event.unicode == '8':
                    res = 100
                    maze = Maze(width//res,height//res,res)
                    maze.Build(0)
                    Ant.monde = maze.Show()
                if event.unicode == '9':
                    monde = randomCave(noise,res_x,res_y)
                    Ant.monde = printCave(monde,res_x,res_y)

        start = tm.time()
        Ant.grid = Ant.monde.copy()
        if affichage==0:
            Ant.grid += Ant.phe_food + 65536 * Ant.phe_base
        elif affichage==1:
            Ant.grid += Ant.phe_food
        elif affichage==2:
            Ant.grid +=  65536 * Ant.phe_base
        ga.surfarray.blit_array(win,Ant.grid)
        cycle += 1
        if (cycle % 2) == 0:
            Ant.phe_food = (Ant.phe_food - DEC).clip(0,255)
            Ant.phe_base = (Ant.phe_base - DEC).clip(0,255)
        # la barre
        # ga.draw.line(win,WALL,(width/2,200),(width/2,400),30)

        for colonie in colonies:
            nFo = len(foods)
            for i in range(nFo-1,-1,-1):
                foods[i].Show()
                if foods[i].fin:
                    foods.pop(i)
            nF = len(colonie.fourmi_list)
            for i in range(nF-1,-1,-1):
                f = colonie.fourmi_list[i]
                if mode == 0:
                    nb.jit(f.Behavior())
                elif mode == 1:
                    f.Force(f.Wander())
                elif mode == 2:
                    f.Force(f.Arrive(target))
                elif mode == 3:
                    f.Edge()
                    f.HandleBase()
                nb.jit(f.Update())
                nb.jit(f.Show())
                if f.aLive():
                    colonie.fourmi_list.pop(i)
                if f.getFood:
                    f.getFood = 0
                    nbGenese += 1
            if (nbGenese >= 20 and nF <=LIMITE):
                nbGenese = 0
                colonie.fourmi_list.append(Ant(f.base.x,f.base.y,colonie.color))
            end = tm.time()
            delta = round((end-start)*1000)
            mx = width-150
            # mess_nour = 'Speed = ' + str(Ant.maxSpeed)
            # message.render_to(win, (mx, 10), mess_nour, 0xFFFFFF)
            # mess_nour = 'Force = ' + str(Ant.maxForce)
            # message.render_to(win, (mx, 22), mess_nour, 0xFFFFFF)
            # mess_nour = 'Projection = ' + str(Ant.wanderProjection)
            # message.render_to(win, (mx, 34), mess_nour, 0xFFFFFF)
            # mess_nour = 'Radius = ' + str(Ant.wanderRadius)
            # message.render_to(win, (mx, 46), mess_nour, 0xFFFFFF)

            # mess_nour = 'colonie = ' + str(len(colonie))
            # message.render_to(win, (mx, 82), mess_nour, 0xFFFFFF)
            # mess_nour = 'Génèse = ' + str(nbGenese)
            # message.render_to(win, (mx, 94), mess_nour, 0xFFFFFF)
            colonie.Show(message)
        end = tm.time()
        delta = round((end-start)*1000)
        mess_nour = 'time = ' + str(delta)
        message.render_to(win, (width-150, 20), mess_nour, 0xFFFFFF)
        # win.blit(fond,(0,0))
        ga.display.flip()
        if speed > 1:
            ga.time.wait(speed)
    return 0
##
if __name__ == '__main__':
    sys.exit(main())