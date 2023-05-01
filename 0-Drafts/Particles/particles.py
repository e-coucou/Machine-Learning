import pygame as ga
import pygame.gfxdraw as gfx
from pygame import Vector2 as Vc
import numpy as np
import math, random
import sys
import time as tm
from matplotlib import pyplot as plt

width = 600
height = 600
Origine = Vc(width//2,150)
GRID = 20
SIZE = width // GRID

def dist(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def getRainbow(t):
    r = math.sin(t);
    g = math.sin(t + 0.33 * 2.0 * math.pi);
    b = math.sin(t + 0.66 * 2.0 * math.pi);
    return ga.Color(int(255.0 * r * r),int(255.0 * g * g),int(255.0 * b * b),255)

class Solver():

    def __init__(self,p,m) -> None:
        self.particules = p
        self.monde = m
        self.gravite = Vc(0,1.0)
        self.dt = 0.2
        self.step = 4

    def update(self):
        sub_dt = self.dt/self.step
        for _ in range(self.step):
            grid = np.zeros(shape=(GRID,GRID),dtype=object)
            for id,p in enumerate(self.particules):
                p.force(self.gravite)
                p.limite(self.monde.pos,self.monde.radius)
                i = int(p.pos.x // SIZE)
                j = int(p.pos.y // SIZE)
                k = grid[i,j]
                if k==0:
                    grid[i,j] = [id]
                else:
                    grid[i,j].append(id)
            self.collision_2(grid.copy())
            for p in self.particules:
                p.update(sub_dt)

    def collisions_3(self,grid):
        for p in self.particules:
            i = int(p.pos.x // SIZE)
            j = int(p.pos.y // SIZE)
            g = grid[max(i-1,0):min(i+1,GRID),max(j-1,0):min(j+1,GRID)]
            s = (g[g != 0])
            if (s.size >1):
                s = np.concatenate(s)
                self.collision_3(s)

    def collision_2(self,grid):
        for i in range(GRID):
            for j in range(GRID):
                g = grid[i:i+2,j:j+2]
                s = (g[g != 0])
                if (s.size >1):
                    s = np.concatenate(s)
                    self.collisions(s)

    def collisions(self, s):
        nb = s.size
        for i in range(nb):
            p=self.particules[s[i]]
            # print(i,nb)
            for j in range(i,nb):
                o=self.particules[s[j]]
                if (o != p):
                    o_p = p.pos - o.pos
                    d = o_p.length()
                    if (d<(o.r+p.r)):
                        n = o_p/d
                        corr = o.r + p.r - d
                        # print('collide',d,corr,n)
                        p.pos += 0.5 * n * corr
                        o.pos -= 0.5 * n * corr

    def collision(self):
        for p in self.particules:
            for o in self.particules:
                if (o != p):
                    o_p = p.pos - o.pos
                    d = o_p.length()
                    if (d<(o.r+p.r)):
                        n = o_p/d
                        corr = o.r + p.r - d
                        p.pos += 0.5 * n * corr
                        o.pos -= 0.5 * n * corr
    def collision_3(self,s):
        for p in s:
            for o in s:
                if (o != p):
                    o_p = p.pos - o.pos
                    d = o_p.length()
                    if (d<(o.r+p.r)):
                        n = o_p/d
                        corr = o.r + p.r - d
                        p.pos += 0.5 * n * corr
                        o.pos -= 0.5 * n * corr

    def render(self,win):
        for p in self.particules:
            win.blit(p.surface,(p.pos.x-p.r,p.pos.y-p.r))
        return win

class Grid():

    def setGrid(self,l,c):
        box = ga.Rect(c*SIZE,l*SIZE,SIZE,SIZE)
        gfx.box(self.surface,box,ga.Color(30,30,50,255))
        return self.surface

class Monde():
    surface = ga.Surface([1,1],ga.SRCALPHA)

    def __init__(self,w,h,x,y,r):
        self.pos = Vc(x,y)
        self.radius = r
        self.couleur = ga.Color(0,0,0,255)
        self.surface = ga.Surface([w,h],ga.SRCALPHA)

    def render(self,grid_on=False):
        gfx.filled_circle(self.surface,int(self.pos.x),int(self.pos.y),int(self.radius),self.couleur)
        if grid_on:
            for c in range(GRID):
                for l in range(GRID):
                    gfx.hline(self.surface,0,width,c*SIZE,(255,255,255,255))
                    gfx.vline(self.surface,l*SIZE,0,height,(255,255,255,255))
        return self.surface

class Particule():
    surface = ga.Surface([1,1],ga.SRCALPHA)

    def __init__(self,x,y,acc,couleur):
        self.pos = Vc(x,y)+acc
        self.anc = Vc(x,y)
        self.acc = acc
        self.r = 10
        # self.r = random.randint(10,15)
        # r = random.randint(0,255)
        # g = random.randint(0,255)
        # b = random.randint(0,255)
        # self.couleur = ga.Color(r,g,b,255)
        self.couleur = couleur
        self.surface = ga.Surface([self.r*2+1,self.r*2+1],ga.SRCALPHA)
        self.render()

    def update(self,dt):
        vel  = self.pos - self.anc
        self.anc = self.pos.copy()
        self.pos += vel + self.acc * dt * dt
        self.acc = (0,0)

    def force(self,f):
        self.acc = self.acc + f

    def limite(self,c,r):
        p = self.pos - c
        d = p.length()
        if (d>(r-self.r)):
            n = p / d
            self.pos = c + n * (r - self.r)

    def setCouleur(self,c):
        self.couleur=c
        self.render()

    def render(self):
        gfx.filled_circle(self.surface,int(self.r),int(self.r),int(self.r),self.couleur)

def main()->int:
    # init pygame avec la fenetre win
    ga.init()
    win = ga.display.set_mode([width,height])
    monde = Monde(width,height,width//2,height//2,250)
    monde_s = monde.render()
    run = True
    iter = 0
    nb = 0
    grid_on = False
    direction = Vc()
    # print(direction)

    particules = []
    # for i in range(1):
    #     particules.append(Particule(400,120))

    solver = Solver(particules,monde)
    start = tm.time()
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
                if event.unicode == 'g':
                    grid_on = not(grid_on)
                    monde_s = monde.render(grid_on)

        #-----
        # Ajout des particules ...
        iter += 1
        if (iter%7 == 0 and nb<500):
            nb += 1
            dir = math.sin(iter/100)*90 + 90
            direction.from_polar((.9,dir))
            # particules.append(Particule(Origine.x-100, Origine.y,direction,getRainbow(iter/1000)))
            particules.append(Particule(Origine.x+100, Origine.y,direction,getRainbow(iter/1000)))
            if (nb==500):
                end = tm.time()
                print(end-start, GRID)

        solver.update()
        win.fill(ga.Color(120,120,120,120))
        win.blit(monde_s,(0,0))

        solver.render(win)
        # ga.time.wait(10)
        
        # display('(c) eCoucou',win,font,width-60,height-7,fontsize=10)
        # win.blit(graphe_s,(15,15))
        ga.display.flip()

    return 0
if __name__ == '__main__':
    sys.exit(main())
    # sys.exit(main(True,A_file='a00_Actor.h5',C_file='a00_Critic.h5',J_file='a00_Game.json'))
    # sys.exit(run(A_file='00_Actor.h5',C_file='00_Critic.h5'))