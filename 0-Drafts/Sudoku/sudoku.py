import pygame as ga
import pygame.gfxdraw as gfx
from pygame import Vector2 as Vc
import numpy as np
import numpy.ma as ma
import math
import sys
import time as tm
from matplotlib import pyplot as plt

MAX_ITER = 30000
width = 390
height = 390
test = np.zeros((9,9))

m = np.array([0,5,8,6,0,5,0,0,0,0,0,0,0,3,0,6,0,0,3,0,0,0,6,0,0,9,0,0,4,0,0,8,0,0,3,0,0,0,3,0,0,0,0,0,0,3,0,0,0,9,4,0,0,0,0,6,0,0,0,0,3,2,0,0,0,0,0,0,8,0,0,4,0,0,1,0,0,0,9,0,6])

s = np.array([6,2,7,1,5,4,8,3,9,3,8,1,6,9,7,2,4,5,4,9,5,2,3,8,1,6,7,2,1,4,7,8,5,6,9,3,8,6,9,4,1,3,5,7,2,7,5,3,9,6,2,4,8,1,1,3,2,8,7,6,9,5,4,9,7,6,5,4,1,3,2,8,5,4,8,3,2,9,7,1,6])

'''
test = np.array([[0,0,2,0,0,6,0,0,5],
                 [0,9,0,0,0,0,8,0,0],
                 [0,3,0,7,0,0,4,1,0],
                 [0,0,0,2,0,0,6,0,0],
                 [5,0,0,4,0,3,0,0,7],
                 [0,0,4,0,0,7,0,0,0],
                 [0,1,3,0,0,2,0,8,0],
                 [0,0,8,0,0,0,0,4,0],
                 [9,0,0,5,0,0,2,0,0]
                 ],dtype=np.int8)
'''
test_i = np.array([[0,6,1,0,0,3,0,0,5],
                 [2,0,0,0,1,4,0,0,0],
                 [0,0,7,0,0,2,0,0,0],
                 [0,4,2,0,0,0,0,6,9],
                 [0,0,9,0,0,0,7,0,0],
                 [7,3,0,0,0,0,2,4,0],
                 [0,0,0,3,0,0,5,0,0],
                 [0,0,0,2,4,0,0,0,8],
                 [8,0,0,7,0,0,3,9,0]
                 ],dtype=np.int8)

def getMouse(x,y):
    rx = x*9//width
    ry = y*9//height
    # print(x,y,rx,ry)
    return rx, ry

class Grid():
    surf = ga.Surface((1,1))
    mess = None
    mess_mini = None
    cx,cy = -1, -1
    mask = np.zeros((9,9))

    def __init__(self) -> None:
        self.pas = width//9
        self.possible = []
        for x in range(9):
            self.possible.append([])
            for y in range(9):
                self.possible[x].append( np.arange(9)+1)
        # print(self.possible)

    def set(self,x,y,value):
        test[y,x] = value
        self.mask[y,x]=1

    def update(self):
        self.mask = np.zeros((9,9))
        for x in range(9):
            for y in range(9):
                v = test[y,x]
                if v==0:
                    i = (x//3)*3
                    j = (y //3)*3
                    v_ = np.unique(test[y,:])
                    h_ = np.unique(test[:,x])
                    c_ = np.unique(test[j:j+3,i:i+3])
                    a_ = np.delete(np.unique(np.concatenate((v_,h_,c_))),[0])
                    b_ = np.setdiff1d(self.possible[x][y],a_)
                    self.possible[x][y] = b_
                else:
                    self.possible[x][y] = np.array([0])
        for x in range(9):
            for y in range(9):
                v=self.possible[x][y]
                if (len(v) != 0):
                    v = np.stack(self.possible[x][y])
                    # if (x==0 and y==0):
                    #     print(' - ',np.stack(self.possible[x][y]))
                    # print(v, len(v))
                    if (len(v)==1 and v!=0):
                        # print(x,y,v)
                        self.set(x,y,v[0])
                        # test[y,x]=v[0]
        # vérification des carrés
        for i in range(3):
            for j in range(3):
                t = []
                x=i*3
                y=j*3
                for a in range(3):
                    for b in range(3):
                        v = np.array(self.possible[x+a][y+b],dtype=np.int8)
                        t = (np.concatenate((v,t)))
                        # t = np.delete(np.concatenate((v,t)),[0])
                u,c = np.unique(t,return_counts=True)
                # print(i,j,t,u,c)
                if (len(c[c==1]) > 0):
                    ww= np.where(c == 1)
                    for w in ww[0]:
                        # print(w, u[w], i ,j )
                        c= int(u[w])
                        # print('index Carre ',i,j,' ->',c)
                        for a in range(3):
                            for b in range(3):
                                v = np.array(self.possible[x+a][y+b],dtype=np.int8)
                                ww2 = (v[v==c])
                                if ww2:
                                    for w in ww2:
                                        # print(i*3+a,j*3+b,w)
                                        self.set(i*3+a,j*3+b,w)

        # vérification des colonnes
        for i in range(9):
            t = []
            for j in range(9):
                v = self.possible[i][j]
                t = (np.concatenate((v,t)))
            u,c = np.unique(t,return_counts=True)
            # print(i,j,t,u,c)
            if (len(c[c==1]) > 0):
                ww= np.where(c == 1)
                # print(t,w, u[w], i ,j )
                for w in ww[0]:
                    c=int(u[w])
                    # print('index Colonne',i,' ->', c)
                    for j in range(9):
                        v = np.array(self.possible[i][j],dtype=np.int8)
                        ww2 = (v[v==c])
                        if ww2:
                            for w in ww2:
                                # print(i,j,w)
                                self.set(i,j,w)
                                # test[j,i] = w
        # vérification des lignes
        for j in range(9):
            t = []
            for i in range(9):
                v = self.possible[i][j]
                t = (np.concatenate((v,t)))
            u,c = np.unique(t,return_counts=True)
            # print(i,j,t,u,c)
            if (len(c[c==1]) > 0):
                ww= np.where(c == 1)
                # print(t,w, u[w], i ,j )
                for w in ww[0]:
                    c=int(u[w])
                    # print('index Ligne',j,' ->',c)
                    for i in range(9):
                        v = np.array(self.possible[i][j],dtype=np.int8)
                        ww2 = (v[v==c])
                        if ww2:
                            for w in ww2:
                                # print(i,j,w)
                                # test[j,i] = w
                                self.set(i,j,w)
        # print(test.sum())
        # for i in range(9):
        #     print(test[:,i].sum())
        # for j in range(9):
        #     print(test[j,:].sum())
            

    def show(self):
        self.surf.fill((0,0,0)) #type: ignore
        # affichage de la grille
        for i in range(10):
            gfx.vline(self.surf,i*self.pas,0,height,(255,255,255))  # type: ignore
            gfx.hline(self.surf,0,width,i*self.pas,(255,255,255))  # type: ignore
            if ( i%3 == 0):
                gfx.vline(self.surf,i*self.pas+1,0,height,(255,255,255))  # type: ignore
                gfx.hline(self.surf,0,width,i*self.pas+1,(255,255,255))  # type: ignore
        # affichage du jeux
        r = ga.Rect((self.cx)*self.pas+1,(self.cy)*self.pas+1,self.pas-2,self.pas-2)
        gfx.box(self.surf,r,(70,70,70)) #type: ignore
        for x in range(9):
            for y in range(9):
                v = test[y,x]
                r = ga.Rect(x*self.pas+1,y*self.pas+1,self.pas-2,self.pas-2)
                # est ce que la cellule est dans la selection
                if (self.cx==x or self.cy==y or (x//3*3 ==self.cx//3*3 and y//3*3 == self.cy//3*3)):
                    gfx.box(self.surf,r,(70,70,70))
                # est ce que la cellule possède un nombre
                if v>0:
                    if (self.mask[y,x]==1):
                        gfx.box(self.surf,r,(0,70,70))
                    # if (v<4):
                    #     gfx.box(self.surf,r,(70,70,0))
                    # elif (v<7):
                    #     gfx.box(self.surf,r,(0,70,70))
                    # else:
                    #     gfx.box(self.surf,r,(70,0,70))
                    value=str(v)
                    self.mess.render_to(self.surf, ((x+0.35)*self.pas, (y+0.33)*self.pas), value, (255,255,255,255)) # type: ignore 
                # sinon on affiche toutes les options
                else:
                    p = self.possible[x][y]
                    # s'il n(y plus d'option erreur !)
                    if (len(p)==0):
                        gfx.box(self.surf,r,(200,0,0))
                    else:
                        if (len(p)==1):
                            gfx.box(self.surf,r,(0,200,0))
                        for c in p:
                            value=str(c)
                            i=c-1
                            # print(value)
                            self.mess_mini.render_to(self.surf, ((x+(i%3)*(0.33)+0.1)*self.pas, (y+0.1+(i//3)*(0.33))*self.pas), value, (255,255,255,255))
            

def main()->int:
    global test,t
    # init pygame avec la fenetre win
    ga.init()
    win = ga.display.set_mode([width,height])
    win.fill(0)
    run = True
    iter = 0

    n = m > 0
    t2 = (n * s).reshape((9,9))
    print(t2)

    test = t2.copy()
    # test = test_i.copy()
    # print(test)

    grille = Grid()
    grille.surf = ga.Surface([width,height],ga.SRCALPHA)
    grille.mess = ga.freetype.SysFont("couriernew", 32, bold=False, italic=False)
    grille.mess_mini = ga.freetype.SysFont("couriernew", 12, bold=False, italic=False)

    while run:
        # Gestion interface
        for event in ga.event.get():
            if event.type == ga.QUIT:
                run = False
            if event.type == ga.MOUSEBUTTONUP:
                mx,my = ga.mouse.get_pos()
                grille.cx, grille.cy = getMouse(mx,my)
            if event.type == ga.KEYDOWN:
                if event.key == ga.K_ESCAPE:
                    run = False
                if event.unicode=='1':
                    test[grille.cy, grille.cx] = 1
                if event.unicode=='2':
                    test[grille.cy, grille.cx] = 2
                if event.unicode=='3':
                    test[grille.cy, grille.cx] = 3
                if event.unicode=='4':
                    test[grille.cy, grille.cx] = 4
                if event.unicode=='5':
                    test[grille.cy, grille.cx] = 5
                if event.unicode=='6':
                    test[grille.cy, grille.cx] = 6
                if event.unicode=='7':
                    test[grille.cy, grille.cx] = 7
                if event.unicode=='8':
                    test[grille.cy, grille.cx] = 8
                if event.unicode=='9':
                    test[grille.cy, grille.cx] = 9
                if event.unicode == 'r':
                    n = m > 0
                    t2 = (n * s).reshape((9,9))
                    test = t2.copy()
                    grille.__init__()
                    iter = 0
        iter +=1

        # affiche en pygame
        # if (iter%20 == 0):
        #     ga.surfarray.blit_array(surface,image) # type: ignore
        #     win.blit(surface,(0,0))
        #     # ga.time.wait(10)
        
        # display('(c) eCoucou',win,font,width-60,height-7,fontsize=10)
        if (test.sum()<405):
            grille.update()
            grille.show()
            win.blit(grille.surf,(0,0))
            ga.time.wait(500)
            ga.display.flip()
            print(iter)
    return 0
if __name__ == '__main__':
    sys.exit(main())
    # sys.exit(main(True,A_file='a00_Actor.h5',C_file='a00_Critic.h5',J_file='a00_Game.json'))
    # sys.exit(run(A_file='00_Actor.h5',C_file='00_Critic.h5'))