# from array import array
# from asyncio import set_event_loop
# from distutils.log import debug
# from importlib.resources import path
import sys
from telnetlib import DM
# from telnetlib import TM
import numpy as np
import pygame as ga
import pygame.gfxdraw as gfx
from math import *
from random import *
# import quadtree as qt
import timeit


width = 600
height = 600
N = 1000
DEC = 1
GRID =30

def setHeading(v,a,mag=1):
    return (np.array([mag*cos(a),-mag*sin(a)]) + v).astype(int)

def heading(v):
    return (np.arctan2(-v[1],v[0]))

def setMag(v,mag):
    coef = np.linalg.norm(v)
    if coef >0:
        return (v/coef*mag)
    else:
        return v

def norm(v):
    coef = np.linalg.norm(v)
    if coef>0:
        return ( v/coef)
    else:
        return v

def dist(v1,v2):
    d = np.linalg.norm(v2-v1)
    return d

def limit(v,lim):
    return norm(v)*lim

def findProjection(pos,a ,b):
    v1 = pos -a
    v2 = b - a
    v2 = norm(v2)
    sp = np.dot(v1,v2)
    return (a+v2*sp).astype(int),sp

# Class HitBox
class HBox():
    def __init__(self,) -> None:
        self.liste = []

    def Add(self,id):
        self.liste.append(id)

    def Remove(self,id):
        self.liste.remove(id)

# Class Point
class Point():
    def __init__(self,x,y) -> None:
        self.x = x
        self.y = y

    def Plot(self,array):
        array[self.x][self.y] = 255
    
    def Show(self,surface):
        ga.draw.circle(surface,'red',[self.x,self.y],2)

class Path():
    def __init__(self,x1,y1,x2,y2) -> None:
        self.start = np.array([x1,y1],dtype=np.int16)
        self.end = np.array([x2,y2],dtype=np.int16)
        self.largeur = 40
    
    def Show(self,surface):
        ga.draw.line(surface,(120,120,120),self.start,self.end,self.largeur)
        gfx.line(surface,self.start[0],self.start[1],self.end[0],self.end[1],(255,255,255))

class Ant():
    def __init__(self,base,surface,id=id,couleur='white',size=7):
        self.base = base
        self.id = id
        self.size=size
        self.pos = np.array([self.base.x,self.base.y],dtype=np.int16)
        self.vel = np.array([randint(-3,3),randint(-3,3)])
        self.acc = np.array([0,0])
        self.desired = self.vel
        self.maxSpeed = randint(3,7)
        self.maxForce = .7
        self.wanderTheta =  uniform(-pi/4,pi/4)
        self.couleur = couleur
        self.mass = 1
        self.debug = False
        self.surface = surface
        self.isol = 15
        self.link = 40

    def Behavior(self,colonie,path,liste):
        if len(liste)>0:
            self.Force(self.Separate(colonie,liste) * 2)
            self.Force(self.Align(colonie,liste) * 1)
        self.Force(self.Follow(path) * 2 )

    def Force(self,force):
        self.acc = self.acc + force

    def Update(self):
        self.vel = setMag(self.acc + self.vel,self.maxSpeed)
        self.pos = (self.pos + self.vel).astype(int)
        self.acc = np.array([0,0])

    def WanderDebug(self):
        wPoint = (self.pos + self.vel*7).astype(int)
        gfx.pixel(self.surface,wPoint[0],wPoint[1],(255,255,255))
        r = 15
        gfx.circle(self.surface,int(wPoint[0]),int(wPoint[1]),r,(00,255,00))
        theta = self.wanderTheta + heading(self.vel)
        wPoint = wPoint + np.array([r * cos(theta),-r * sin(theta)]).astype(int)
        gfx.filled_circle(self.surface,wPoint[0],wPoint[1],2,(0,255,255))
        gfx.line(self.surface,self.pos[0],self.pos[1],wPoint[0],wPoint[1],(255,255,255))
        self.wanderTheta += uniform(-0.5,0.5)
        self.Force(self.Seek(wPoint))

    def Wander(self):
        wPoint = (self.pos + self.vel*7).astype(int)
        r = 15
        theta = self.wanderTheta + heading(self.vel)
        wPoint = setHeading(wPoint,theta,r)
        self.wanderTheta += uniform(-0.5,0.5)
        self.Force(self.Seek(wPoint))

    def WanderRandom(self):
        r = 2
        a = round(random()*2*pi)
        dx = round(r*sin(a))
        dy = round(r*cos(a))
        self.vel = self.vel + np.array([dx,dy])+self.acc
        self.vel = setMag(self.vel,self.maxSpeed)
        self.pos = (self.pos + self.vel)
        self.acc = np.array([0,0])

    def Separate(self, colonie, liste):
        steer = np.array([0.,0.])
        cpt = 0
        for l in liste:
            # d = dist(self.pos,colonie[l].pos)
            # if (d<self.isol and d>0):
            steer += (norm(self.pos - colonie[l].pos))
            # steer += s
            cpt += 1
        if (cpt>0):
            steer = steer/cpt
            steer = setMag(steer,self.maxSpeed)
            steer = setMag(steer-self.vel, self.maxForce)
        return steer

    def Align(self, colonie, liste):
        steer = np.array([0.,0.])
        cpt = 0
        for l in liste:
            # d = dist(self.pos,colonie[l].pos)
            # if (d<self.link and d>0):
            steer += colonie[l].vel
            cpt += 1
        if (cpt>0):
            steer = steer/cpt
            steer = setMag(steer,self.maxSpeed)
            steer = setMag(steer-self.vel, self.maxForce)
        return steer

    def Edge(self):
        # if (self.pos[0] > width-100 or self.pos[0] < 100):
        #     self.Force([-self.vel[0]*2,0])
        # if (self.pos[1] > height-100 or self.pos[1] < 100):
        #     self.Force([0,-self.vel[0]*2])
        if (self.pos[0] >= width-1):
            self.pos[0] = 2
        if self.pos[0] <= 1:
            self.pos[0] = width-2
        if self.pos[1] <= 1:
            self.pos[1] = height-2
        if self.pos[1] >= height-1:
            self.pos[1] = 2

    def Seek(self,target):
        self.desired = setMag(target - self.pos,self.maxSpeed)
        steer = setMag(self.desired - self.vel,self.maxForce)
        return (steer)

    def Follow(self,path):
        futur = (self.pos + self.vel * 4).astype(int)
        dir = np.array([0,0])
        target = np.array([0,0])
        if self.debug:
            gfx.circle(self.surface,self.pos[0],self.pos[1],3,(255,0,255))
            gfx.circle(self.surface,futur[0],futur[1],3,(255,255,255))
        dMin = 100000000
        for i in range(len(path)):
            a=path[i].start
            b=path[i].end
            dir = b-a
            tg,sp = findProjection(futur,a,b)
            if (tg[0]<min(a[0],b[0])) or (tg[0]>max(a[0],b[0])) or (tg[1]<min(a[1],b[1])) or (tg[1]>max(a[1],b[1])):
                tg = b
                i_ = (i+1)%len(path)
                a = path[i_].start
                b = path[i_].end
                dir = b-a
                if self.debug:
                    print('not in',i,tg,target,futur,a,b)
            d = np.sum(np.square(tg-futur))
            if self.debug:
                 print('yes',i,tg,target,futur,a,b,d,dMin)
            if d<dMin:
                target = tg
                dMin=d
                dir=norm(dir).astype(int)*4
                target = target+dir
        if self.debug:
            print(target)
            gfx.filled_circle(self.surface,target[0],target[1],3,(255,255,255))
        if dMin > 400:
            return (self.Seek(target))
        else:
            return(np.array([0,0]))

    def Show(self):
        gfx.pixel(self.surface,self.pos[0],self.pos[1],(255,255,255))
        # gfx.filled_circle(self.surface,self.pos[0],self.pos[1],self.size,(255,0,255))
        # s1 = self.size/2
        # s2=self.size/5
        # angle = heading(self.vel)
        # a_iris = heading(self.desired)
        # oeil1 = setHeading(self.pos,angle-0.7,s1)
        # oeil2 = setHeading(self.pos,angle+0.7,s1)
        # iris1 = setHeading(oeil1,a_iris,s2)
        # iris2 = setHeading(oeil2,a_iris,s2)
        # gfx.filled_circle(self.surface,oeil1[0],oeil1[1],int(s1),(255,255,255))
        # gfx.filled_circle(self.surface,oeil2[0],oeil2[1],int(s1),(255,255,255))
        # gfx.filled_circle(self.surface,iris1[0],iris1[1],2,(0,0,0))
        # gfx.filled_circle(self.surface,iris2[0],iris2[1],2,(0,0,0))
#

class Colony():
    def __init__(self,base,surface,couleur='red') -> None:
        self.base = base
        self.couleur = couleur
        self.coulVal = 255
        self.colony = []
        self.id = 0
        self.surface=surface
        self.base_vc = np.array([self.base.x, self.base.y])
        self.hitbox = []
        self.l = width//GRID
        h = height//GRID
        for i in range(self.l*(h+1)):
            self.hitbox.append(HBox())
    
    def Add(self):
        id = len(self.colony)
        self.colony.append(Ant(self.base,couleur=self.couleur,surface=self.surface,id=id))
        x = self.colony[id].pos[0]//GRID
        y = self.colony[id].pos[1]//GRID
        id_ = self.l * y + x
        self.hitbox[id_].Add(id)
    
    def Couleur(self,couleur,coulVal):
        self.coulVal = coulVal
        for f in self.colony:
            f.couleur=couleur

    def Debug(self):
        for f in self.colony:
            f.debug = not(f.debug)

    def Size(self,inc):
        for f in self.colony:
            f.size += inc
            if f.size < 5:
                f.size=5

    def Update(self,path,mode=1):
        for f in self.colony:
            x = f.pos[0]//GRID
            y = f.pos[1]//GRID
            id_1 = self.l * y + x
            self.hitbox[id_1].Remove(f.id)
            id_c = int((f.pos[0]+f.vel[0]*5)%width//GRID + (f.pos[1]+f.vel[1]*5)%height//GRID * self.l)
            f.Edge()
            if (mode==1):
                f.WanderDebug()
            elif mode==2:
                f.Wander()
            elif mode==3:
                f.Behavior(self.colony,path,self.hitbox[id_c].liste) #,self.hitbox[id_c].liste)
            elif mode==5:
                f.Follow(path)
            f.Update()
            x = f.pos[0]%width//GRID
            y = f.pos[1]%height//GRID
            id_1 = self.l * y + x
            # print(id_1,x,y,f.pos)
            self.hitbox[id_1].Add(f.id)

            f.Show()

    def Show(self):
        for f in self.colony:
            f.Show()


def main() -> int:
    # init pygame avec la fenetre win
    ga.init()
    win = ga.display.set_mode([width,height])
    win.fill(0)
    speed = 30
    # Essai
    pos = np.array([100,100])
    vel = np.array([5,-2])
    # world=Monde()
    # grille=Grid(win,world)
    # message = ga.freetype.SysFont("couriernew", 12, bold=False, italic=False)
    # # init de la nourriture
    # nourriture = Nourriture([250,250],255)
    # nourriture.show(win)
    # ga.pixelcopy.surface_to_array(world.food,win,'P')
    # # init de la colonie de fourmis
    fourmis = Colony(Point(200,200),couleur='white',surface=win)
    for i in range(N):
        fourmis.Add()
    fourmis.Show()
    # init de la boucle
    run = True
    mode=1 # affiche les fourmis
    paths = []
    paths.append(Path(100,100,width-100,100))
    paths.append(Path(width-100,100,width-100,height - 100))
    paths.append(Path(width - 100, height - 100,100,height-100))
    paths.append(Path(100, height - 100,width//4,height//2))
    paths.append(Path(width//4,height//2,100,100))
    mousse = False
    pause = False
    while run:
        if not(pause):
            win.fill(0)
            for p in paths:
                p.Show(win)
        if mode==4:
            gfx.filled_circle(win,pos[0],pos[1],15,(255,0,255))

    # Gestion interface
        for event in ga.event.get():
            if event.type == ga.QUIT:
                run = False
            if event.type == ga.MOUSEBUTTONUP:
                if mode == 4:
                    pos=ga.mouse.get_pos()
                else:
                    fourmis.Add()
            if event.type == ga.KEYDOWN:
                # print(event.key)
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
                elif event.unicode == ">":
                    speed -= 5
                    if speed < 0:
                        speed=1
                elif event.unicode == "<":
                    speed += 5
                elif event.unicode == "+":
                    fourmis.Size(1)
                elif event.unicode == "-":
                    fourmis.Size(-1)
                elif event.unicode == "d" or event.unicode == 'D':
                    fourmis.Debug()
                elif event.key == ga.K_ESCAPE:
                    run = False
                elif event.key == ga.K_SPACE:
                    pause = not(pause)
        # pos = ga.mouse.get_pos()
        if not(pause):
            fourmis.Update(paths,mode)
            fourmis.Show()
        # fourmis.Plot()
        # world.Update()
        # grille.Live(mode)
        # mess_nour = 'croc = ' + str(world.Count())
        # message.render_to(win, (500, 30), mess_nour, 0xFFFFFF)
            ga.display.flip()
        ga.time.wait(speed)
    return 0
##
if __name__ == '__main__':
    sys.exit(main())