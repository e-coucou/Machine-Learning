import numpy as np
from math import *
# from msilib.schema import Class
# 

class PVector():
    def __init__(self,x,y):
        self.v = np.array([x,y])

    def setHeading(self,a,mag=1):
        return (np.array([mag*cos(a),-mag*sin(a)]) + self.v).astype(int)

    def heading(self):
        return (np.arctan2(-self.v[1],self.v[0]))

    def setMag(self,mag):
        coef = np.linalg.norm(self.v)
        self.v = self.v/coef*mag

    def norm(self):
        coef = np.linalg.norm(self.v)
        self.v = self.v/coef

    def dist(self,o):
        d = np.linalg.norm(o-self.v)
        return d

    def limit(self,lim):
        return self.norm(self.v)*lim

    def findProjection(self,pos,a ,b):
        v1 = pos -a
        v2 = b - a
        v2 = self.norm(v2)
        sp = np.dot(v1,v2)
        return (a+v2*sp).astype(int),sp

    def add(self,other):
        self.v += other

    def sub(self,other):
        self.v -= other
    
    def mult(self,n):
        self.v *=  n

    @staticmethod
    def sub(v1,v2):
        return v2-v1

    def add(v1,v2):
        return v1+v2

    def mult(v,n):
        return v * n
    
    def setMag(v,mag):
        coef = np.linalg.norm(v)
        return (v/coef*mag)
    