import pygame as ga
import pygame.gfxdraw as gfx
from pygame import Vector2 as Vc
import numpy as np
import math
import sys
import time as tm
from matplotlib import pyplot as plt

MAX_ITER = 30000
width = 600
height = 200
def dist(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

class Fluid():

    def __init__(self):
        self.nx = width
        self.ny = height
        self.tau = 0.53
        # t = 3000

        # lattice speed and weight
        self.nl = 9
        self.cxs = np.array([0,0,1,1,1,0,-1,-1,-1])
        self.cys = np.array([0,1,1,0,-1,-1,-1,0,1])
        self.weight = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

        # conditions initiales
        self.F = np.ones((self.ny,self.nx,self.nl)) + 0.01 * np.random.randn(self.ny,self.nx,self.nl)
        self.F[:,:,3] = 2.3

        self.cylindre = np.full((self.ny,self.nx),False)
        for y in range(0,self.ny):
            for x in range(0,self.nx):
                if dist(self.nx//4,self.ny//2,x,y) < 12: #self.ny//3:
                    self.cylindre[y][x] = True
        # self.cylindre2 = np.full((self.ny,self.nx),False)
        # for y in range(0,self.ny):
        #     for x in range(0,self.nx):
        #         if dist(self.nx//4,self.ny//3*2,x,y) < 10: #self.ny//3:
        #             self.cylindre2[y][x] = True


    def update(self):
        # efface les bords en x
        self.F[:,-1,[6,7,8]] = self.F[:,-2,[6,7,8]]
        self.F[:,0,[2,3,4]] = self.F[:,1,[2,3,4]]

        # loop
        for i,cx,cy in zip(range(self.nl),self.cxs,self.cys):
            self.F[:,:,i] = np.roll(self.F[:,:,i],cx,axis=1)
            self.F[:,:,i] = np.roll(self.F[:,:,i],cy,axis=0)

        bndry = self.F[self.cylindre,:]
        bndry = bndry[:, [0,5,6,7,8,1,2,3,4]]
        # bndry2 = self.F[self.cylindre2,:]
        # bndry2 = bndry2[:, [0,5,6,7,8,1,2,3,4]]

        # fluide variable
        density = np.sum(self.F, 2)
        ux = np.sum(self.F * self.cxs, 2) / density
        uy = np.sum(self.F * self.cys, 2) / density

        self.F[self.cylindre,:] = bndry
        ux[self.cylindre] = 0
        uy[self.cylindre] = 0
        # self.F[self.cylindre2,:] = bndry2
        # ux[self.cylindre2] = 0
        # uy[self.cylindre2] = 0

        # collisions
        Feq = np.zeros(self.F.shape)
        for i,cx,cy,w in zip(range(self.nl),self.cxs,self.cys,self.weight):
            Feq[:,:,i] = density * w * (
                1 + 3*(cx*ux + cy*uy) + 9*(cx*ux + cy*uy)**2 / 2 - 3*(ux**2 + uy**2)/2 
            )
        self.F -= (self.F - Feq)/self.tau
        return ux,uy

def main()->int:
    # init pygame avec la fenetre win
    ga.init()
    win = ga.display.set_mode([width,height])
    run = True
    iter = 0

    fluide = Fluid()
    surface = ga.Surface([width,height]) #,ga.SRCALPHA)

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
        iter +=1
        if (iter>MAX_ITER): run=False
        ux, uy = fluide.update()
        d = np.sqrt(ux**2 + uy**2)
        dfy_dx = ux[2:, 1:-1 ] - ux[0:-2, 1:-1]
        dfx_dy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
        curl = dfy_dx - dfx_dy

        curl = d

        # affiche en pygame
        if (iter%20 == 0):
            curl = np.int32(curl * 1000.0)
            # M = np.max(curl)
            # m = np.min(curl)
            # image = (np.int16((curl-m) / (M-m) * 0x100) & 0xFF)
            image = curl.T * 0x100 #+ curl*0x10000
            ga.surfarray.blit_array(surface,image) # type: ignore
            # surface = ga.pixelcopy.make_surface(image.T)
            # arr = ga.surfarray.array2d(surface)
            # ga.pixelcopy.surface_to_array(arr,surface)
            # print(arr.shape, arr[0,0])
            # print(image.T.shape, np.max(image.T))
            win.blit(surface,(0,0))
            # ga.time.wait(10)
        
        # affiche en matplot
        # if (iter%100 == 0):
        #     # plt.imshow(d)
        #     plt.imshow(curl, cmap='bwr')
        #     plt.pause(0.001)
        #     plt.cla()
        # print(img.shape)
        # print(img[:10,:10])
        # display('(c) eCoucou',win,font,width-60,height-7,fontsize=10)
        # win.blit(graphe_s,(15,15))
        ga.display.flip()

    return 0
if __name__ == '__main__':
    sys.exit(main())
    # sys.exit(main(True,A_file='a00_Actor.h5',C_file='a00_Critic.h5',J_file='a00_Game.json'))
    # sys.exit(run(A_file='00_Actor.h5',C_file='00_Critic.h5'))