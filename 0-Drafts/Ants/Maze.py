import pygame as ga
import numpy as np
from math import *
import pygame.gfxdraw as gfx
import random

class Cell():
    surface = None
    color = (80,80,80)
    res = 20

    def __init__(self,i,j) -> None:
        self.i = i
        self.j = j
        x = self.i * self.res 
        y = self.j * self.res
        self.border = [1,1,1,1]
        self.visited = False
        self.wall = [(x,y),(x+self.res,y),(x+self.res,y+self.res),(x,y+self.res)] # N E S O

    def Show(self):
        # if self.visited:
        #     ga.draw.rect(self.surface,(255,0,255),ga.Rect(self.wall[0][0],self.wall[0][1],self.res,self.res))
        for i in range(4):
            if self.border[i]:
                ga.draw.line(self.surface,self.color,self.wall[i],self.wall[(i+1)%4],20)

    def getRC(self):
        return self.i,self.j

class Maze():
    width = 0
    height = 0

    def __init__(self,l,c,res,color=(80,80,80)):
        Cell.surface = ga.Surface((self.width,self.height))
        Cell.color = color
        self.res = res
        Cell.res = self.res
        self.cells = []
        self.l = l
        self.c = c

        for j in range(self.c):
            for i in range(self.l):
                self.cells.append(Cell(i,j))

    def Show(self):
        img = np.zeros((self.width,self.height),np.int64)
        for c in self.cells:
            c.Show()
        ga.display.flip()
        ga.pixelcopy.surface_to_array(img,Cell.surface,'P')
        return img

    def getId(self,v):
        i,j = v
        if (i>=0 and i<self.c and j>= 0 and j<self.l):
            return (i+self.c*j)
        else:
            return -1 

    def findVoisin(self,id):
        i,j = self.cells[id].getRC()
        v = [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
        liste = []
        for k in range(4):
            nId = self.getId(v[k])
            if nId>0 :
                if not(self.cells[nId].visited):
                    liste.append(nId)
        if len(liste)>0:
            return random.choice(liste)
        else:
            return None

    def removeWall(self,a,b):
        x = b.i - a.i
        if x == 1:
            a.border[1] = 0
            b.border[3] = 0
        elif (x == -1):
            a.border[3] = 0
            b.border[1] = 0
        y = b.j - a.j
        if y ==1:
            a.border[2] = 0
            b.border[0] = 0
        elif y == -1:
            a.border[0] = 0
            b.border[2] = 0
    
    def Build(self,id):
        next = id
        stack = []
        stack.append(next)
        while next != None:
            self.cells[next].visited = True
            next = self.findVoisin(next)
            if next != None:
                stack.append(next)
                self.removeWall(self.cells[id],self.cells[next])
                id = next
            else:
                print(stack)
                if not stack==[]:
                    # print(next)
                    next = stack.pop()
                    self.removeWall(self.cells[id],self.cells[next])
                    id = next
