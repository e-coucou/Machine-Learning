from random import random
from tkinter import *
from turtle import clear
import numpy as np
from scipy.spatial import distance

import quadtree as qt

canvas_width = 800
canvas_height = 600
HOLD = False
it = 0
N = 100
Nourriture = np.array([450,400])
Nou_dist = 50
Foumiliere = np.array([650,500])
Fou_dist = 50


boundary = qt.Rectangle(canvas_width/2,canvas_height/2,canvas_width/2,canvas_height/2)
qt_P = [qt.Quadtree(boundary,4),qt.Quadtree(boundary,4)]
qt_F = [qt.Quadtree(boundary,4),qt.Quadtree(boundary,4)]


class ant:
    def __init__(self,pos,vel,couleur='green',debug=False):
        self.pos = pos
        self.vel = vel
        self.acc = np.zeros((2))
        self.couleur = couleur
        self.maxSpeed = 10
        self.mag = 1
        self.rho = 30
        self.pheromone = False
        self.nourriture = False
        self.fourmiliere = False
        self.cible = 0 # 1 cherche la nourriture

        self.debug = debug
        self.radius = 20

        self.path = []

    def Update(self,qtP,qtF):
        self.vel = self.acc + self.vel
        self.mag = np.linalg.norm(self.vel)
        coef = min(self.maxSpeed,self.mag)/self.mag
        self.vel = self.vel * coef
        self.pos = self.pos + self.vel
        self.acc = np.zeros((2))
        if (self.nourriture):
            self.path.append(self.pos)
            if (len(self.path) > self.rho):
                self.path.pop(0)
            for p in self.path:
                qtP.insert(qt.Point(p[0],p[1]))
        if (self.fourmiliere):
            self.path.append(self.pos)
            if (len(self.path) > self.rho):
                self.path.pop(0)
            for p in self.path:
                qtF.insert(qt.Point(p[0],p[1]))

    def Force(self,f,qtP,qtF):
        if self.cible == 0:
            bary, chg=self.Bary(qtP,f)
            if chg:
                self.cible=1
            else:
                bary, chg=self.Bary(qtF,f)
                if chg:
                    self.cible = 2
        elif self.cible == 1:
            bary,chg = self.Bary(qtP,f)
        else:
            bary,chg = self.Bary(qtF,f)
        self.acc = bary

    def Edge(self):
        x=self.pos[0]
        y=self.pos[1]
        if ((x<0) | (x>canvas_width)):
            self.vel[0] = -self.vel[0]
        if ((y<0) | (y>canvas_height)):
            self.vel[1] = -self.vel[1]

    def Nourriture(self):
        if (distance.euclidean(self.pos,Nourriture) < Nou_dist):
            self.couleur = 'blue'
            self.pheromone = True
            self.nourriture = True
            self.fourmiliere = False
            self.cible = 2

    def Fourmiliere(self):
        if (distance.euclidean(self.pos,Foumiliere) < Fou_dist):
            self.couleur = 'red'
            self.pheromone = True
            self.nourriture = False
            self.fourmiliere = True
            self.cible = 1

    def Magnitude(self):
        return self.mag

    def Print(self):
        print('position: ',self.pos)
        print('velocité: ',self.vel)
        print('acceleration: ',self.acc)
    
    def Debug(self,qtP,qtF):
        centre = [self.pos[0],self.pos[1]]
        canvas.create_oval(self.pos[0]-self.radius,self.pos[1]-self.radius,self.pos[0]+self.radius,self.pos[1]+self.radius)
        found = []
        found = qtP.queryRadius(self.pos,self.radius,found)
        for f in found:
            canvas.create_oval(f.x-3,f.y-3,f.x+3,f.y+3,fill='violet')
        found = []
        found = qtF.queryRadius(self.pos,self.radius,found)
        for f in found:
            canvas.create_oval(f.x-3,f.y-3,f.x+3,f.y+3,fill='black')

    def Bary(self,qtL,bary):
        barycentre = []
        found=[]
        chg = False
        found = qtL.queryRadius(self.pos,self.radius,found)
        if found:
            for f in found:
                barycentre.append([f.x, f.y])
            bary = (np.mean(barycentre,axis=0) - self.pos) *0.3
            chg=True
            canvas.create_oval(bary[0]-3,bary[1]-3,bary[0]+3,bary[1]+3,fill='black')
        return bary,chg

    def Show(self,qtP,qtF):
        s= 4
        x = self.pos[0]
        y = self.pos[1]
        canvas.create_oval(x-s,y-s,x+s,y+s,fill=self.couleur)
        # if (self.pheromone):
        #     s=2
        #     for p in self.path:
        #         x = p[0]
        #         y = p[1]
        #         canvas.create_oval(x-s,y-s,x+s,y+s,fill='blue')
        # if (self.fourmiliere):
        #     s=2
        #     for p in self.path:
        #         x = p[0]
        #         y = p[1]
        #         canvas.create_oval(x-s,y-s,x+s,y+s,fill='red')
        if self.debug:
            self.Debug(qtP,qtF)

fenetre = Tk()
# fenetre.config(bg='#D9D8D7')
canvas = Canvas(fenetre, width=canvas_width, height=canvas_height, background='white')
canvas.pack()

fourmis = []
for n in range(N):
    fourmis.append(ant(np.array([650,500]),np.zeros((2))))
# fourmis[50].debug=True

#----------
# test quadtree
def test_quadtree():
    canvas.delete('all')
    qt_P[0] = qt.Quadtree(boundary,4)

    for n in range(1000):
        p = qt.Point(np.random.randint(0,800),np.random.randint(0,600))
        qt_P[0].insert(p)
    all = []
    rec = []
    all = qt_P[0].getPoint(all)
    rec = qt_P[0].getRect(rec)
    for a in all:
        canvas.create_oval(a[0]-1,a[1]-1,a[0]+1,a[1]+1,fill='black')
    for r in rec:
        canvas.create_rectangle(r[0],r[1],r[2],r[3])
    found = []
    source = qt.Rectangle(100,100,50,50)
    found = qt_P[0].query(source,found)
    canvas.create_rectangle(source.x-source.w,source.y-source.h,source.x+source.w,source.y+source.h,fill='yellow')
    for f in found:
        canvas.create_oval(f.x-2,f.y-2,f.x+2,f.y+2,fill='blue')
    found = []
    centre = [400,400]
    radius = 50
    canvas.create_oval(centre[0]-radius,centre[1]-radius,centre[0]+radius,centre[1]+radius,fill='yellow')
    found = qt_P[0].queryRadius(centre,radius,found)
    for f in found:
        canvas.create_oval(f.x-3,f.y-3,f.x+3,f.y+3,fill='red')
    barycentre = []
    for f in found:
        barycentre.append([f.x, f.y])
    bary = np.mean(barycentre,axis=0)
    canvas.create_oval(bary[0]-3,bary[1]-3,bary[0]+3,bary[1]+3,fill='black')

#-----------
def loop():
    global qt_P, qt_F, it
    it_c = it
    it = (it + 1) % 2
    qt_P[it] = qt.Quadtree(boundary,4)
    qt_F[it] = qt.Quadtree(boundary,4)

    canvas.delete('all')
    canvas.create_oval(Nourriture[0]-Nou_dist,Nourriture[1]-Nou_dist,Nourriture[0]+Nou_dist,Nourriture[1]+Nou_dist,fill='yellow')
    canvas.create_oval(Foumiliere[0]-Fou_dist,Foumiliere[1]-Fou_dist,Foumiliere[0]+Fou_dist,Foumiliere[1]+Fou_dist,fill='grey')
    for f in fourmis:
        force = (np.random.rand(2) - 0.5)* 2 * max(1,f.Magnitude())
        # print(f,f.magnitude())
        f.Force(force,qt_P[it_c],qt_F[it_c])
    # fourmis.print()
        f.Update(qt_P[it],qt_F[it])
        f.Edge()
        f.Nourriture()
        f.Fourmiliere()
        f.Show(qt_P[it_c],qt_F[it_c])
    # print(fourmis[50])

    if not HOLD:
        fenetre.after(100,loop)

def hold():
    global HOLD
    HOLD = not HOLD
    if not HOLD:
        fenetre.after(100,loop)
        pause['text'] = 'Pause'
    else:
        pause['text'] = 'Resume'

# ligne1 = canvas.create_line(75, 0, 75, 120)
# ligne2 = canvas.create_line(0, 60, 150, 60)
# txt = canvas.create_text(75, 60, text="Cible", font="Arial 16 italic", fill="blue")

# bouton de sortie
Button(fenetre, text="FIN", command=fenetre.quit).pack(side='left',padx=5,pady=5)
# Button(fenetre, text="Run", command=loop, bg='#4a7abc', fg='blue',activebackground='blue', activeforeground='black').pack(expand=True,side='right',padx=5,pady=5)
Button(fenetre, text="Run", command=loop).pack(side='right',padx=5,pady=5)
pause = Button(fenetre, text="Pause", command=hold)
pause.pack(side='right',padx=5,pady=5)
Button(fenetre, text="Test", command=test_quadtree).pack(side='left',padx=5,pady=5)

fenetre.mainloop()
