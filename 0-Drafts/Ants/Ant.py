import pygame as ga
import numpy as np
from math import *
import pygame.gfxdraw as gfx
from random import *

class Ant():
    # visualisation pygame
    surface = None
    phe_food = None
    phe_base = None
    grid = None
    width = 100
    height = 100
    # caracteristique des fourmis
    size = 6
    wanderTheta = pi/2
    wanderRadius = 2
    wanderProjection = 14
    maxSpeed = 3.5
    maxSpeed_squared = 100
    maxForce = 0.35
    # detections nourriture/base
    dArrive_sq = 2500
    foodDetection = 70
    foodDetection_sq = 4900
    food = []
    PheroAngle = [-45,0,45]
    PheroSq = 10
    PheroDist = 16

    LIFE = 2048
    PHERO = 255
    # animation
    debug = True
    anim = 0

    def __init__(self,x,y,color=(255,0,255)) -> None:
        self.pos = ga.math.Vector2(x,y)
        self.base = ga.math.Vector2(self.pos.x,self.pos.y)
        self.dir = random()*360
        self.vel = ga.math.Vector2()
        self.vel.from_polar((randint(2,10),self.dir))
        self.acc = ga.math.Vector2(0,0)
        self.desired = ga.math.Vector2()
        self.forFood = True
        self.forBase = False
        self.withFood = False
        self.target = None
        self.life = self.LIFE
        self.getFood = 0
        self.pid = None

        self.color = color
        self.PheroCt = ga.math.Vector2((self.PheroSq//2,self.PheroSq//2))
        self.look = []
        self.lookChoix = 1
        self.pheroCpt = self.PHERO

    def Behavior(self):
        if self.target == None:
            self.Force(self.Wander())
            if self.withFood:
                if not self.Look4Base():
                    self.HandleBase()
            else:
                if not self.Look4Food():
                    self.HandleFood()
        else:
            self.Force(self.Arrive(self.target))

    def Look4Food(self):
        food = []
        for f in self.food:
            self.desired = (f.pos-self.pos)
            d,a = self.desired.as_polar()
            if d < self.foodDetection:
                food.append(f)
        if len(food)>0:
            self.target = food[0].pos
            self.pid = food[0].pid
            return True
        return False

    def Look4Base(self):
        self.desired = (self.base-self.pos)
        d,a = self.desired.as_polar()
        if d < self.foodDetection:
            self.target=self.base
            return True
        return False

    def HandleFood(self):
        vlook =[]
        self.look = []
        p = ga.math.Vector2(0,0)
        for i in range(3):
            p.from_polar((self.PheroDist,self.dir+ self.PheroAngle[i]))
            p = p + self.pos - self.PheroCt
            if self.debug:
                self.look.append(p.copy())
                r=ga.Rect(int(p.x),int(p.y),self.PheroSq,self.PheroSq)
                gfx.rectangle(self.surface,r,(0,255,0))
            vlook.append(np.sum(self.phe_food[int(p.x):int(p.x+self.PheroSq), int(p.y):int(p.y+self.PheroSq)]))
        self.lookChoix = np.argmax(vlook)
        if (vlook[self.lookChoix]>0 and self.lookChoix !=1 ):
            self.vel = self.vel.rotate(self.PheroAngle[self.lookChoix])
            self.acc.xy = (0,0)
        else:
            self.lookChoix=1
        if self.debug:
            p = self.look[self.lookChoix]
            r=ga.Rect(int(p.x),int(p.y),self.PheroSq,self.PheroSq)
            gfx.box(self.surface,r,(0,255,0))

    def HandleBase(self):
        vlook=[]
        self.look = []
        p = ga.math.Vector2(0,0)
        for i in range(3):
            p.from_polar((self.PheroDist,self.dir+ self.PheroAngle[i]))
            p = p +  self.pos - self.PheroCt
            if self.debug:
                self.look.append(p.copy())
                r=ga.Rect(int(p.x),int(p.y),self.PheroSq,self.PheroSq)
                gfx.rectangle(self.surface,r,(255,0,0))
            vlook.append(np.sum(self.phe_base[int(p.x):int(p.x+self.PheroSq), int(p.y):int(p.y+self.PheroSq)]))
        self.lookChoix = np.argmax(vlook)
        if (vlook[self.lookChoix]>0 and self.lookChoix !=1) :
            self.vel = self.vel.rotate(self.PheroAngle[self.lookChoix])
            self.acc.xy = (0,0)
        else:
            self.lookChoix=1
        if self.debug:
            p = self.look[self.lookChoix]
            r=ga.Rect(int(p.x),int(p.y),self.PheroSq,self.PheroSq)
            gfx.box(self.surface,r,(255,0,0))

    def Wander(self):
        wPoint = (self.pos + self.vel*self.wanderProjection)
        if self.debug:
            gfx.pixel(self.surface,int(wPoint.x),int(wPoint.y),(255,255,255))
            gfx.circle(self.surface,int(wPoint.x),int(wPoint.y),self.wanderRadius,(00,255,00))
        theta = self.wanderTheta + self.vel.as_polar()[1]
        adV = ga.math.Vector2()
        adV.from_polar((self.wanderRadius,theta))
        wPoint += adV
        self.wanderTheta += uniform(-45,45)
        if self.debug:
            gfx.filled_circle(self.surface,int(wPoint.x),int(wPoint.y),2,(0,255,255))
            gfx.line(self.surface,int(self.pos.x),int(self.pos.y),int(wPoint.x),int(wPoint.y),(255,255,255))
        return (self.Seek(wPoint))

    def Edge(self):
        if self.pos.x < 0 :
            self.vel.x = -self.vel.x
            self.pos.x = 1
        if self.pos.x >= self.width:
            self.vel.x = -self.vel.x
            self.pos.x = self.width-1
        if self.pos.y < 0 :
            self.vel.y = -self.vel.y
            self.pos.y = 1
        if self.pos.y >= self.height:
            self.vel.y = -self.vel.y
            self.pos.y = self.height-1

    def aLive(self):
        self.life -= 1
        return self.life<=0

    def Update(self):
        if self.pheroCpt > 1:
            self.pheroCpt -= 1
        self.vel += self.acc
        m = self.vel.magnitude_squared()
        if m > (self.maxSpeed_squared):
            self.vel = self.vel * self.maxSpeed_squared/m
        self.pos += self.vel
        self.acc *= 0      
        # maj et edge
        _, self.dir = self.vel.as_polar()
        # self.pos.x = (self.pos.x%self.width)
        # self.pos.y = (self.pos.y%self.height)
        self.Edge()
        if self.withFood:
            self.phe_food[int(self.pos.x),int(self.pos.y)] += self.pheroCpt
        else:
            self.phe_base[int(self.pos.x),int(self.pos.y)] += self.pheroCpt

    def Force(self,f):
        self.acc = f

    def Seek(self,target):
        self.desired = (target-self.pos)
        self.desired = self.desired.normalize() * self.maxSpeed
        steer = (self.desired-self.vel)
        steer = steer.normalize() * self.maxForce
        return steer

    def Arrive(self,target):
        self.desired = (target-self.pos)
        d = self.desired.magnitude_squared()
        self.desired = self.desired.normalize()
        if d<self.dArrive_sq:
            if (d < 100):
                self.target=None
                self.pheroCpt = self.PHERO
                self.life = self.LIFE
                if self.withFood:
                    self.getFood = 1
                    self.withFood = False
                else:
                    try:
                        self.pid.decQte()
                        self.withFood = True
                    except:
                        self.target=None
                        self.pid = None
                        return ga.math.Vector2(-3,-3)
                self.vel = self.vel.reflect(self.vel).normalize()*self.maxSpeed
            c = np.interp(d,[0,self.dArrive_sq],[1,self.maxSpeed])
            self.desired *= c
        else:
            self.desired *= self.maxSpeed
        steer = (self.desired-self.vel)
        steer = steer.normalize() * self.maxForce
        if self.debug:
            ga.draw.line(self.surface,(255,0,0),self.pos,self.pos+self.desired*10,1)
            ga.draw.line(self.surface,(255,255,0),self.pos,self.pos+self.vel*10,1)
            ga.draw.line(self.surface,(0,255,255),self.pos,self.pos+steer*10,1)
        return steer

    def Show(self):
        if self.debug: # avec les épures de construction : switch d/D
            _, self.dir = self.vel.as_polar()
            # print(self.dir,self.vel)
            p1 = ga.math.Vector2()
            p2 = ga.math.Vector2()
            p1.from_polar((self.foodDetection,self.dir-45))
            p2.from_polar((self.foodDetection,self.dir+45))
            p1 += self.pos
            p2 += self.pos
        #  gfx.line(self.surface,int(self.pos.x),int(self.pos.y),int(self.vel.x),int(self.pos.y),(255,255,255))
            gfx.pie(self.surface,int(self.pos.x),int(self.pos.y),self.foodDetection,int(self.dir-45), int(self.dir+45),(120,120,120,120))
            # gfx.filled_trigon(self.surface,int(self.pos.x),int(self.pos.y),int(p1.x),int(p1.y),int(p2.x),int(p2.y),(120,120,120,130))
            if self.withFood:
                # gfx.filled_circle(self.surface,int(self.pos.x),int(self.pos.y),self.size,(0,0,255))
                gfx.pixel(self.surface,int(self.pos.x),int(self.pos.y),(0,0,255))
            else:
                # gfx.filled_circle(self.surface,int(self.pos.x),int(self.pos.y),self.size,(255,0,255))
                gfx.pixel(self.surface,int(self.pos.x),int(self.pos.y),(255,0,255))
            ga.draw.line(self.surface,(255,255,255),self.pos,self.pos+self.vel*10,1)
        else:
            if self.anim==0: # un pixel est c'est tout
                gfx.pixel(self.surface,int(self.pos.x),int(self.pos.y),(255,255,255))
            elif self.anim == 1: # une jolie petite fourmi : switch a/A
                oeil1 = ga.math.Vector2()
                oeil2 = ga.math.Vector2()
                iris1 = ga.math.Vector2()
                iris2 = ga.math.Vector2()
                s1 = self.size/2
                s2 = self.size/5
                _, angle = self.vel.as_polar()
                _, a_iris = self.desired.as_polar()
                # print(angle,tmp,s1,self.pos)
                oeil1.from_polar((s1,angle-40))
                oeil1 += self.pos
                oeil2.from_polar((s1,angle+40))
                oeil2 += self.pos
                iris1.from_polar((s2,a_iris))
                iris1 += oeil1
                iris2.from_polar((s2,a_iris))
                iris2 += oeil2
                # gfx.filled_ellipse(self.surface,int(self.pos.x),int(self.pos.y),int(self.size*1.5),self.size,(255,0,255,100))
                if self.withFood:
                    gfx.filled_circle(self.surface,int(self.pos.x),int(self.pos.y),self.size,(255,0,255,150))
                else:
                    gfx.filled_circle(self.surface,int(self.pos.x),int(self.pos.y),self.size,(0,255,120,150))
                gfx.filled_circle(self.surface,int(oeil1.x),int(oeil1.y),int(s1),(255,255,255))
                gfx.filled_circle(self.surface,int(oeil2.x),int(oeil2.y),int(s1),(255,255,255))
                gfx.filled_circle(self.surface,int(iris1.x),int(iris1.y),2,(0,0,0))
                gfx.filled_circle(self.surface,int(iris2.x),int(iris2.y),2,(0,0,0))
            elif self.anim == 2:
                tete = self.pos - ga.math.Vector2(0,self.size)
                inter = self.pos + ga.math.Vector2(0,3/4*self.size)
                patte = self.pos  + ga.math.Vector2(0,7/4*self.size)
                corps = self.pos + ga.math.Vector2(0,3*self.size)
                # if self.withFood:
                #     gfx.filled_circle(self.surface,int(self.pos.x),int(self.pos.y),self.size,(255,0,255,150))
                # else:
                #     gfx.filled_circle(self.surface,int(self.pos.x),int(self.pos.y),self.size,(0,255,120,150))
                gfx.filled_circle(self.surface,int(tete.x),int(tete.y),int(self.size),(255,255,255))
                gfx.filled_circle(self.surface,int(patte.x),int(patte.y),int(self.size//5),(255,255,255))
                gfx.filled_ellipse(self.surface,int(inter.x),int(inter.y),int(self.size//2),int(self.size),(255,255,255))
                gfx.filled_ellipse(self.surface,int(corps.x),int(corps.y),int(3*self.size//2),int(2*self.size),(255,255,255))
    
    def newSet(self,target):
        self.forFood = True
        self.withFood = False
        self.target = target


    @classmethod
    def setMaxSpeed(cls,s):
        cls.maxSpeed += s
        cls.maxSpeed_squared = cls.maxSpeed * cls.maxSpeed
    


class Colony():
    def __init__(self) -> None:
        pass


class Food():
    surface = None
    
    def __init__(self,x,y) -> None:
        self.pos = ga.math.Vector2(x,y)
        self.pheromone = None
        self.qte = 1025
        self.fin = False
        self.pid = None

    def decQte(self):
        if self.qte>1:
            self.qte -= 1
        else:
            self.fin = True
    
    def Show(self):
        gfx.filled_circle(self.surface,int(self.pos.x),int(self.pos.y),floor(self.qte//256),(0,255,0))
