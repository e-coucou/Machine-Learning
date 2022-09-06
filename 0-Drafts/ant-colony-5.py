from array import array
from locale import ABMON_10
import sys
from xml.etree.ElementTree import PI
import numpy as np
import pygame as ga
from math import *
from random import *
import quadtree as qt


width = 600
height = 600
N = 100
DEC = 1

def lena():
    import pickle, os
    fname = os.path.join(os.path.dirname(__file__),'lena.dat')
    f = open(fname,'rb')
    lena = np.array(pickle.load(f))
    f.close()
    return lena

def sector_mask(shape,centre,radius,angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """
    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    #ep tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    # if tmax < tmin:
    #         tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - angle_range[0] #tmin
    theta = np.where(theta<0,2*pi+theta,theta)
    # wrap angles between 0 and 2*pi
    # theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
#    anglemask = theta <= (tmax-tmin)
    anglemask = theta <= (angle_range[1]-angle_range[0])

    return circmask*anglemask

class Monde():
    def __init__(self) -> None:
        self.food = np.zeros((width,height),np.int64)
        self.colonies = []
        self.fourmiliere = []
        self.nourriture = []
        self.idCol = 0

    def Count(self):
        return np.count_nonzero(self.food)
    
    def Detect(self, x,y):
        return np.count_nonzero(self.food[x-5:x+5][y-5:y+5])

    def Add(self):
        colony = np.zeros((width,height),np.int64)
        phero_F = np.zeros((width,height),np.int64)
        phero_N = np.zeros((width,height),np.int64)
        self.colonies.append(colony)
        self.fourmiliere.append(phero_F)
        self.nourriture.append(phero_N)
        self.idCol += 1
        return (self.idCol - 1)

    def Update(self):
        for i in range(self.idCol):
            self.fourmiliere[i] = (self.fourmiliere[i] - DEC).clip(0,255)
            self.nourriture[i] = (self.nourriture[i] - DEC).clip(0,255)

    def UpdateN(self, id, x,y , val):
        self.nourriture[id][x][y] = val
    def UpdateF(self, id, x,y , val):
        self.fourmiliere[id][x][y] = val
    
    def Croc(self,x,y):
        self.food[x][y] = 0

class Grid():
    def __init__(self,surface,world) -> None:
        self.grid = np.zeros((width,height),np.int64)
        self.surface = surface
        self.world = world        

    def Live(self,mode):
        self.grid = self.world.food
        if mode==1:
            for c in self.world.colonies:
                self.grid = self.grid + c
        elif mode==2:
            self.grid = self.grid + self.world.fourmiliere[0]*65536
        elif mode==3:
            self.grid = self.grid + self.world.nourriture[0]*256
        ga.surfarray.blit_array(self.surface,self.grid)

class Nourriture():
    def __init__(self,pos,nb):
        self.pos = pos
        self.nb = nb
        self.couleur = 'green'
        self.count = 0

    def Get(self):
        self.count += 1

    def show(self,win):
        ga.draw.circle(win,self.couleur,self.pos,self.nb//13)

class Point():
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y

    def Plot(self,array):
        array[self.x][self.y] = 255
    
    def Show(self,surface):
        ga.draw.circle(surface,'red',[self.x,self.y],2)

class Ant():
    def __init__(self,base,world,couleur='red'):
        self.base = base
        self.world = world
        self.vect = np.array([self.base.x,self.base.y],dtype=np.int16)
        self.couleur = couleur
        self.vel = np.array([0,0])
        self.maxSpeed = 5
        self.mag = 0
        self.nour = 0 # 0 pas 

    def Update(self):
        # desire = self.DetectFood()
        # self.vel = self.vel + desire
        dx = round((random()-0.49)*4)
        dy = round((random()-0.49)*4)
        self.vel = [dx,dy] + self.vel
        self.mag = np.linalg.norm(self.vel)
        if self.mag>0.01:
            coef = min(self.maxSpeed,self.mag)/self.mag
        else:
            coef = 1
        self.vel = self.vel * coef
        self.vect = (self.vect + self.vel).clip(0,width-1).astype(int)
        if self.nour == 0:
            self.Take()

    def Take(self):
        if (self.world.food[self.vect[0], self.vect[1]] != 0):
            self.world.Croc(self.vect[0], self.vect[1])
            self.nour = 1 # 0 cherche N, 1 cherche Fourmiliere
            return True
        else:
            return False
    
    def DetectFood(self):
        a0 = np.arctan2(self.vel[0],self.vel[1])-pi/4
        s = []
        for i in range(4):
            m = a0 + i*pi/8
            M = a0 + (i+1)*pi/8
            mask = sector_mask(self.world.food.shape,[self.vect[0],self.vect[1]],50,[m,M])
            s.append(np.count_nonzero(self.world.food[mask]))
        c = np.argmax(s)
        a = a0 + (2*c+1)*pi/16
        return np.array([sin(a),cos(a)])

    def DetectFood2(self):
        a0 = np.arctan2(self.vel[0],self.vel[1])
        if a0<-2.35619:
            return
        elif a0<-1.57079:
            return
        elif a0<-0.78539:
            return
        elif a0<0:
            return
        elif a0<0.78539:
            return
        elif a0<1.57079:
            return
        elif a0<2.35619:
            return
        else:
            return
        return 0

    def Show(self,surface):
        ga.draw.circle(surface,self.couleur,self.vect,2)

    def Plot(self,array):
        self.pos.Plot(array)

class Colony():
    def __init__(self,base,surface,world,couleur='red') -> None:
        self.base = base
        self.couleur = couleur
        self.coulVal = 255
        self.world = world
        self.colony = []
        self.id = 0
        self.surface=surface
        self.base_vc = np.array([self.base.x, self.base.y])
        self.worldId = self.world.Add()
    
    def Add(self):
        self.colony.append(Ant(self.base,world=self.world,couleur=self.couleur))
    
    def Couleur(self,couleur,coulVal):
        self.coulVal = coulVal
        for f in self.colony:
            f.couleur=couleur

    def Update(self):
        for f in self.colony:
            f.Update()
            if f.nour == 0:
                self.world.UpdateF(self.worldId,f.vect[0],f.vect[1],255)
            elif f.nour == 1:
                self.world.UpdateN(self.worldId,f.vect[0],f.vect[1],255)

    def Print(self):
        for f in self.colony:
            print(f.world,f.vect)

    def Plot(self):
        # self.plot = np.zeros((width,height),dtype=np.int64)
        plot = np.zeros((width,height),dtype=np.int64)
        for f in self.colony:
            plot[f.vect[0]][f.vect[1]] = self.coulVal
        self.world.colonies[self.worldId] = plot

    def Show(self):
        for f in self.colony:
            f.Show(self.surface)


def main() -> int:
    # Essai
    print(np.arctan2(1,1) )
    print(np.arctan2(-1,1) )
    print(np.arctan2(-1,-1))
    print(np.arctan2(1,-1) )
    print(pi/4, -3*pi/4, pi/2)
    # init pygame avec la fenetre win
    ga.init()
    win = ga.display.set_mode([width,height])
    win.fill(0)
    world=Monde()
    grille=Grid(win,world)
    message = ga.freetype.SysFont("couriernew", 12, bold=False, italic=False)
    # print(ga.font.get_default_font())
    # print(ga.font.get_fonts())
    # init de la nourriture
    nourriture = Nourriture([250,250],255)
    nourriture.show(win)
    ga.pixelcopy.surface_to_array(world.food,win,'P')
    # init de la colonie de fourmis
    fourmis = Colony(Point(200,200),couleur='blue',surface=win,world = world)
    for i in range(N):
        fourmis.Add()
    fourmis.Show()
    # init de la boucle
    run = True
    mode=1 # affiche les fourmis
    while run:
    # Gestion interface
        for event in ga.event.get():
            if event.type == ga.QUIT:
                run = False
            if event.type == ga.KEYDOWN:
                if event.key == ga.K_r:
                    fourmis.Couleur('red',0xFF0000)
                elif event.key == ga.K_b:
                    fourmis.Couleur('blue',0x0000FF)
                elif event.key == ga.K_v:
                    fourmis.Couleur('green',0x00FF00)
                elif event.key == ga.K_j:
                    fourmis.Couleur('yellow',0xFFFF00)
                elif event.key == ga.K_w:
                    fourmis.Couleur('white',0xFFFFFF)
                elif event.key == ga.K_c:
                    fourmis.Couleur('cyan',0x00FFFF)
                elif event.key == ga.K_1:
                    mode=1
                elif event.key == ga.K_2:
                    mode=2
                elif event.key == ga.K_3:
                    mode=3
                elif event.key == ga.K_4:
                    mode=4
                elif event.key == ga.K_ESCAPE:
                    run = False
        fourmis.Update()
        fourmis.Plot()
        world.Update()
        grille.Live(mode)
        mess_nour = 'croc = ' + str(world.Count())
        message.render_to(win, (500, 30), mess_nour, 0xFFFFFF)
        ga.display.flip()
        ga.time.wait(100)
    return 0
##
if __name__ == '__main__':
    sys.exit(main())