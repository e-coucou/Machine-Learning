from pickletools import uint8
import sys
import numpy as np
import pygame as ga
from math import *
from random import *
import quadtree as qt


width = 600
height = 400
N = 100
DEC = 1

class Nourriture():
    def __init__(self,pos,nb):
        self.pos = pos
        self.nb = nb
        self.couleur = 'green'
        self.count = 255

    def Croc(self,arr,x,y):
        arr[x][y] = 0
        # arr = (arr-16777216).clip(0,4294967296) #4278190080)
        # return arr

    def show(self,win):
        ga.draw.circle(win,self.couleur,self.pos,self.nb//13)

class Ant():
    def __init__(self,pos,couleur='black',debug=False) -> None:
        self.pos = pos
        self.vel = np.zeros((2))
        self.acc = np.zeros((2))
        self.couleur = couleur
        self.maxSpeed = 10
        self.mag = 1
        self.rho = 30
        self.nourriture = False
        self.fourmiliere = False
        self.cible = 0 # 0 cherche la nourriture / 1 cherche la fourmilière

        self.debug = debug
        self.radius = 20

    def Mag(self) -> float:
        return self.mag

    def Move(self):
        self.vel = self.acc + self.vel
        self.mag = np.linalg.norm(self.vel)
        coef = min(self.maxSpeed,self.mag)/self.mag
        self.vel = self.vel * coef
        self.pos = (self.pos + self.vel).astype(int)
        self.pos[0] = self.pos[0].clip(0,width-1)
        self.pos[1] = self.pos[1].clip(0,height-1)
        self.acc = np.zeros((2))
    
    def Force(self,f):
        self.acc = f

    def Edge(self):
        if (self.pos[0]<=0 or self.pos[0]>=width-2):
            self.vel[0] *= -1
        if (self.pos[1]<=0 or self.pos[1]>=height-2):
            self.vel[1] *= -1

    def Croc(self,nour):
        if (nour[self.pos[0], self.pos[1]] != 0):
            return True, self.pos[0], self.pos[1]
        return False, 0, 0

    def Show(self,win,arr):
        # ga.draw.circle(win,self.couleur,self.pos,1)
        arr[(self.pos[0]),(self.pos[1])]=255

#-----
def copy2array(win,arr):
    ga.pixelcopy.surface_to_array(arr,win,'P')
    return arr

def copy2surface(arr,win):
    ga.pixelcopy.array_to_surface(win,arr)

#--- Main -
def main() -> int:
    ga.init()
    win = ga.display.set_mode([width,height])
    win.fill((0,0,0))
    array = np.empty(win.get_size(),np.int64)
    a_nour = np.empty(win.get_size(),np.int64)
    # a_nour = ga.surfarray.pixels_green(win)
    # aa_red = a_red + DEC
    # print(a_red.shape)
    run = True
    fourmis = []
    pos = [np.random.randint(0,width),np.random.randint(0,height)]
    for n in range(N):
        fourmis.append(Ant(pos,'red',False))
    nourriture = Nourriture([100,100],255)
    nourriture.show(win)
    # a_nour = ga.surfarray.pixels_red(win)
    a_nour = copy2array(win,a_nour)
    boundary = qt.Rectangle(width/2,height/2,width/2,height/2)
    qt_Nour = qt.Quadtree(boundary,4)
    for x in range(width):
        for y in range(height):
            if a_nour[x][y] != 0:
                qt_Nour.insert(qt.Point(x,y))
    print('Init done')
    #---Boucle principale
    while run:
    # Did the user click the window close button?
        for event in ga.event.get():
            if event.type == ga.QUIT:
                run = False
            if event.type == ga.KEYDOWN:
                # arr = copy2array(win)
                print(a_nour)
        # copy2surface(array,win)
        for f in fourmis:
            force = (np.random.rand(2) * 2 - 0.96) * max(0.5,f.Mag())
            f.Force(force)
            f.Move()
            f.Edge()
            f.Show(win,array)
            croc, x, y = f.Croc(a_nour)
            if croc:
                nourriture.Croc(a_nour,x,y)
        # array = copy2array(win,array)
        array = (array - DEC).clip(0,255)
        # a_nour = nourriture.Croc(a_nour)
        # aa_red = ((aa_red - DEC)).clip(0,255).astype(int)
        # a_red = np.random.randint(0,255,size=(width,height))
        ga.surfarray.blit_array(win,(array*65536 + a_nour))
        # ga.surfarray.blit_array(win,(a_nour))
        ga.display.flip()
        ga.time.wait(10)
    return 0
##
if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit