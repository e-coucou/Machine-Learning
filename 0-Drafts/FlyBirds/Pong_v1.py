import gym
import pygame as ga
import pygame.gfxdraw as gfx
from pygame import Vector2 as Vc
import numpy as np
import math
import sys
import time as tm
import random
from json import JSONEncoder
import json
# from collections import deque
import matplotlib.pyplot as plt
import NN as nn
import NN_keras as nnK
import NN_rky as nnRky

# import ga

width = 1000
height = 520
SCALE = 10
G_w = 700
G_h = 10*21*2
D_w = width - G_w - 20 - 15
D_h = D_w
N = 255
LIMIT = 10
INC = 3

FILE = 'v2_pong'

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class DQNAgent():
    def __init__(self,_dim,gamma_=0.99) -> None:
        self.gamma = gamma_
        self.dim = _dim
        self.rem = 4
        self.size =80
        self.EPOCHS = 10
        # self.LR = 0.000025
        self.LR = 0.000025

        self.obs_size = (self.rem,self.size,self.size) 
        self.memory = np.zeros(self.obs_size)
        self.A_file, self.C_file, self.J_file = '', '', ''

        # self.Actor, self.Critic = nnK.myModel(input_shape=self.obs_size, action_space = self.dim, lr=self.LR)
        self.Actor, self.Critic = nnRky.myModel(input_shape=self.obs_size, action_space = self.dim, lr=self.LR)

        self.obs, self.rewards, self.actions, self.predictions = [], [], [], []
    
    def record(self,o_,r_,a_,i_,p_):
        self.obs.append(o_)
        self.rewards.append(r_)
        a_r = np.zeros([self.dim])
        a_r[a_] = 1
        self.actions.append(a_r)
        self.predictions.append(p_)

    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        # From the last reward to the first...
        for t in reversed(range(0, len(r))):
            # ...reset the reward sum
            if r[t] != 0:
                running_add = 0
            # ...compute the discounted reward
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        # on normalise : xm = (x-m)/std
        discounted_r -= np.mean(discounted_r)
        discounted_r *= np.std(discounted_r)
        return discounted_r

    def act(self, state): # On utilise le réseau de Neurone pour déterminer l'action du joueur
        prediction = self.Actor.predict(state,verbose='0')[0]
        action = np.random.choice(self.dim, p=prediction)
        return action, prediction

    def replay(self): # après chaque partie on rejoue chaque action en entrainant le modele (actor vs critic)
        start=tm.time()
        obs = np.vstack(self.obs)
        actions = np.vstack(self.actions)
        predictions = np.vstack(self.predictions)
        delta = []
        start = cpt_elapse(start,delta)
        rewards = np.vstack(self.discount_rewards(self.rewards))
        start = cpt_elapse(start,delta)
        values = self.Critic.predict(obs,verbose='0')  #[:,0]
        avantages = rewards - values
        # y_true = np.hstack([avantages, predictions, actions])
        y_true = np.hstack([actions])
        start = cpt_elapse(start,delta)
        # h_A = self.Actor.fit(obs,actions,epochs=1,batch_size=128,verbose=0,suffle=False)
        h_A = self.Actor.fit(obs,actions,sample_weight= avantages,epochs=1,batch_size=32,verbose=0,suffle=False)
        # print(y_true.shape,avantages.shape,predictions.shape,actions.shape)
        # h_A = self.Actor.fit(obs,y_true,epochs=self.EPOCHS,batch_size=len(self.rewards),verbose=0,shuffle=True)
        start = cpt_elapse(start,delta)
        h_C = self.Critic.fit(obs,rewards,epochs=1, verbose=0,batch_size=32,shuffle=False)
        # h_C = h_A
        # h_C = self.Critic.fit(obs,rewards,epochs=self.EPOCHS, verbose=0,batch_size=len(self.rewards),shuffle=True)
        start = cpt_elapse(start,delta)
        print(f'prep/rew/cpt/actor/critic : {delta}')
        # reset des tableaux
        self.obs, self.rewards, self.actions, self.predictions = [], [], [], []
        # self.pred = []
        return h_A,h_C

    def getImg(self, img_):
        frame = np.array(img_[::3,::3]).astype(np.float32) / 255.0

        self.memory = np.roll(self.memory,1,axis=0)
        self.memory[0,:,:] = frame

        return np.expand_dims(self.memory,axis=0)

    def _Obs(self, obs):
        frame = obs[35:195:2,::2,:]
        # print(frame.max(),np.unique(frame,return_counts=True))
        frame = 0.299*frame[:,:,0] + 0.587*frame[:,:,1] + 0.114*frame[:,:,2]
        # convert everything to black and white (agent will train faster)
        frame[frame < 100] = 0
        frame[frame >= 100] = 255
        #option ep
        # frame[frame==130]=0
        # frame[frame==72]=0
        # frame[frame==186]=255
        # frame[frame==236]=255
        cpt_frame = np.array(frame).astype(np.float32) / 255.0
        # print(frame.max(),np.unique(frame,return_counts=True))
        # plt.imshow(frame,cmap='gray')
        # plt.show()

        self.memory = np.roll(self.memory,1,axis=0)
        self.memory[0,:,:] = cpt_frame
        return np.expand_dims(self.memory,axis=0)

    def reset(self, state_):
        for i in range(self.rem):
            obs = self._Obs(state_)
            return obs

def convolve2D(image, kernel, strides=1):
    # Cross Correlation
    # kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xK = kernel.shape[0]
    yK = kernel.shape[1]
    xI = image.shape[0]
    yI = image.shape[1]

    # Shape of Output Convolution
    xO = int(((xI - xK + 2 * 0) / strides) + 1)
    yO = int(((yI - yK + 2 * 0) / strides) + 1)
    output = np.zeros((xO, yO),np.uint32)

    # Apply Equal Padding to All Sides
    # if padding != 0:
    #     imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
    #     imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    #     print(imagePadded)
    # else:
    img = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yK:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xK:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * img[x: x + xK, y: y + yK]).sum()
                        # print(output[x,y])
                except:
                    break
    output = (output / output.max() * 255).astype(np.uint8)
    return output

def cpt_elapse(start,delta):
    end = tm.time()
    delta.append((round((end-start)*1_000)))
    return end

def load_agent(agent):
    print("load Actor/Critic")
    agent.Actor = nnK.load_model(agent.A_file,compile=False)
    agent.Critic = nnK.load_model(agent.C_file,compile=True)
    agent.Actor = nnK.comp(agent.Actor,agent.LR)
def save_agent(agent):
    print("Save Actor/Critic")
    agent.Actor.save(agent.A_file)
    agent.Critic.save(agent.C_file)

class epPlot():
    def __init__(self,width_,height_,**kwargs) -> None:
        self.w = width_
        self.h = height_
        self.s =ga.Surface([width_,height_],ga.SRCALPHA)
        self.c = ga.Rect(0,0,width_,height_)
        self.color = kwargs.get('color',(255,255,255,255))
        self.bkcolor = kwargs.get('bkcolor',(0,0,30,255))
    
    def plot_ax(self):
        centre = self.h //2
        for x in range(self.w//10):
            if x%10==0:
                gfx.vline(self.s,int(x*10),centre-7,centre+7,(130,130,0,255))
            else:
                gfx.vline(self.s,int(x*10),centre-5,centre+5,(100,100,0,255))
        b, h = 0, self.h
        for c in range(6):
            b += centre // pow(2,c)
            h -= centre // pow(2,c)
            # print(b,h)
            gfx.hline(self.s,0,self.w,b,(120,120,120,150))
            gfx.hline(self.s,0,self.w,h,(120,120,120,150))

    def plot(self, data, *points, **kwargs):
        SCALE = 10
        old = self.h
        AVG=50
        gfx.box(self.s,self.c,self.bkcolor)
        for v in range(len(data)-1):
            average = int( self.h//2 - SCALE * sum(data[max(v-AVG,0):v+1])/min(AVG,v+1))
            gfx.vline(self.s,int(v),int(self.h/2-SCALE*data[v]),int(self.h/2-SCALE*data[v+1]),self.color)
            gfx.line(self.s,int(v),int(old),int(v+1),int(average),(255,255,0,255))
            old = average
        self.plot_ax()
        echSize = kwargs.get('echSize',10)
        avg = kwargs.get('moyenne',None)
        if avg:
            y = int(self.h/2-SCALE*avg)
            gfx.hline(self.s,len(data)+1,self.w,y,(255,255,100,255))
        for _, point in enumerate(points):
            gfx.filled_circle(self.s,self.w-5,int(self.h/2-SCALE*point),echSize//2,(0,255,255,255))
        echHI = kwargs.get('echHI',None)
        echLO = kwargs.get('echLO',None)
        if echHI:
            y = int(self.h/2-SCALE*echHI - echSize/2 +1)
            gfx.filled_trigon(self.s,self.w-echSize,y,self.w,y,self.w-echSize//2,y+echSize//2,(0,255,0,255))
        if echLO:
            y = int(self.h/2-SCALE*echLO + echSize/2 -1)
            gfx.filled_trigon(self.s,self.w-echSize,y,self.w,y,self.w-echSize//2,y-echSize//2,(255,0,0,255))
        return self.s
    
    def bar(self,data,**kwargs):
        gfx.box(self.s,self.c,self.bkcolor)
        lo = kwargs.get('lo',min(data))
        hi = kwargs.get('hi',max(data)*1.05+0.01)
        color = kwargs.get('color',self.color)
        SCALE = self.h/(hi-lo)
        l = kwargs.get('largeur',1)
        for x in range(len(data)):
            y0=int(self.h - lo*SCALE)
            y1=int(self.h - data[x]*SCALE)
            if l>1:
                r = ga.Rect(x*l+1,y1,l-2,y0-y1)
                gfx.box(self.s,r,color)
            else:
                gfx.vline(self.s,x,y0,y1,color)
        return self.s

    def etoile(self,data_):
        gfx.box(self.s,self.c,self.bkcolor)
        data = np.array(data_)
        sum = np.sum(data,axis=1)
        m = np.average(sum)*2.0
        n = data.shape[0]
        a = 2*math.pi / n
        x0 = self.w // 2
        y0 = self.h // 2
        for i in range(n):
            x1 = int(x0 + data[i][0]/m*x0*math.cos(a*i))
            y1 = int(y0 + data[i][0]/m*x0*math.sin(a*i))
            x2 = int(x1 + data[i][1]/m*x0*math.cos(a*i))
            y2 = int(y1 + data[i][1]/m*x0*math.sin(a*i))
            if i==n-1:
                gfx.line(self.s,x0,y0,x1,y1,(0,0,255,255))
                gfx.line(self.s,x1,y1,x2,y2,(255,255,0,255))
            else:
                gfx.line(self.s,x0,y0,x1,y1,self.color)
                gfx.line(self.s,x1,y1,x2,y2,(255,255,0,255))
        return self.s
    
def load_game(file):
    with open(file, "r") as read_file:
        return  json.load(read_file)

def save_game(data,file):
    print('Save info')
    with open(file, "w") as write_file:
        json.dump(data, write_file, cls=NumpyArrayEncoder)

def display(message,win,font, x, y, **kwargs):
    fontSize = kwargs.get('fontsize',14)
    font = ga.font.Font('freesansbold.ttf', fontSize)
    text = font.render(message, True, (255,255,255), (0,0,0))
    textRect = text.get_rect()
    textRect.center = (x, y)
    win.blit(text, textRect)

#-----------------------------------------------------
def train():
    pass
#--------------------------
def run(**kwargs):
    A_file = kwargs.get('A_file','pong_Adef.h5')
    C_file = kwargs.get('C_file','pong_Cdef.h5')
    blobP = gym.make('PongDeterministic-v4',render_mode = 'human') # 'human'
    # blobP = gym.make('ALE/Pong-v5',render_mode = 'human') # 'human'
    blobP.seed(0) # print(blob.action_space)
    N_ACTIONS = blobP.action_space.n
    agent  = DQNAgent(N_ACTIONS)
    agent.A_file = A_file
    agent.C_file = C_file
    load_agent(agent)
    ga.init()
    state, _ = blobP.reset()
    obs = agent.reset(state)
    live = True
    while live:
        for event in ga.event.get():
            if event.type == ga.QUIT:
                live = False
            if event.type == ga.KEYDOWN:
                if event.key == ga.K_ESCAPE:
                    live = False
        action, _ = agent.act(obs)
        state, _, done, _, _ = blobP.step(action)
        obs = agent._Obs(state)
        if done: live=False

#---------------------------------------------------------------------------------------------------
def main(load=False,**kwargs) -> int:
    #load param
    A_file = kwargs.get('A_file','pong_Adef.h5')
    C_file = kwargs.get('C_file','pong_Cdef.h5')
    J_file = kwargs.get('J_file','pong_Jdef.json')
    # init pygame avec la fenetre win
    ga.init()
    win = ga.display.set_mode([width,height])
    # surface= ga.Surface([width,height],ga.SRCALPHA)
    Graph = epPlot(G_w,G_h,color=(210,210,210,255), bkcolor=(0,0,60,255))
    Graph2 = epPlot(G_w,60,color=(170,170,170,255), bkcolor=(0,0,60,255))
    Graph3 = epPlot(D_w,70,color=(170,170,170,255), bkcolor=(0,0,60,255))
    Graph4 = epPlot(D_w,D_w,color=(170,170,170,255), bkcolor=(0,0,60,255))
    h_A, h_C = None, None

    message = ga.freetype.SysFont("freesansbold.ttf", 14, bold=False, italic=False) # couriernew
    font = ga.font.Font('freesansbold.ttf', 14)
        
    win.fill(0)
    run = True
    cycle = 0
    Infos = np.zeros((3))
    #v1
    # info_score = [-21.0, -20.0, -20.0, -19.0, -19.0, -21.0, -21.0, -19.0, -20.0, -21.0, -21.0, -20.0, -19.0, -18.0, -21.0, -20.0, -20.0, -18.0, -21.0, -18.0, -20.0, -19.0, -20.0, -21.0, -20.0, -20.0, -17.0, -19.0, -20.0, -20.0, -20.0, -18.0, -19.0, -21.0, -21.0, -19.0, -20.0, -18.0, -20.0, -20.0, -20.0, -19.0, -15.0, -20.0, -19.0, -20.0, -18.0, -20.0, -16.0, -19.0, -19.0, -18.0, -17.0, -20.0, -18.0, -21.0, -20.0, -19.0, -16.0, -14.0, -19.0, -20.0, -20.0, -21.0, -18.0, -20.0, -19.0, -20.0, -17.0, -15.0, -18.0, -19.0, -19.0, -19.0, -18.0, -20.0, -19.0, -20.0, -15.0, -16.0, -17.0, -17.0, -19.0, -19.0, -18.0, -17.0, -18.0, -17.0, -17.0, -19.0, -16.0, -18.0, -17.0, -15.0, -15.0, -19.0, -16.0, -16.0, -17.0, -19.0, -17.0, -12.0, -17.0, -15.0, -16.0, -16.0, -18.0, -15.0, -15.0, -17.0, -12.0, -12.0, -12.0, -16.0, -13.0, -12.0, -15.0, -11.0, -10.0, -14.0, -10.0, -15.0, -10.0, -9.0, -11.0, -8.0, -11.0, -8.0, -8.0, -12.0, -9.0, -13.0, -9.0, -5.0, -7.0, -11.0, -9.0, -6.0, -9.0, -2.0, -8.0, -14.0, -10.0, -8.0, -5.0, -3.0, -6.0, -7.0, -5.0]
    #v0
    # info_score = [-21.0, -21.0, -21.0, -19.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -21.0, -20.0, -21.0, -19.0, -21.0, -19.0, -21.0, -20.0, -18.0, -19.0, -19.0, -20.0, -19.0, -21.0, -19.0, -19.0, -21.0, -21.0, -21.0, -20.0, -20.0, -20.0, -19.0, -18.0, -20.0, -20.0, -17.0, -18.0, -20.0, -20.0, -21.0, -21.0, -18.0, -18.0, -19.0, -20.0, -18.0, -19.0, -20.0, -15.0, -20.0, -21.0, -21.0, -20.0, -20.0, -19.0, -19.0, -20.0, -17.0, -18.0, -18.0, -17.0, -18.0, -19.0, -17.0, -15.0, -18.0, -19.0, -20.0, -18.0, -18.0, -18.0, -19.0, -20.0, -19.0, -17.0, -16.0, -20.0, -19.0, -17.0, -18.0, -15.0, -17.0, -16.0, -17.0, -13.0, -13.0, -15.0, -14.0, -16.0, -16.0, -17.0, -16.0, -15.0, -17.0, -16.0, -15.0, -17.0, -15.0, -13.0, -12.0, -16.0, -15.0, -16.0, -13.0, -13.0, -18.0, -17.0, -14.0, -14.0, -11.0, -11.0, -16.0, -15.0, -8.0, -11.0, -12.0, -8.0, -11.0, -13.0, -17.0, -12.0, -19.0, -9.0, -8.0, -12.0, -15.0, -7.0, -1.0, -7.0, -12.0, -7.0, -8.0, -11.0, -12.0, -5.0, -9.0, -9.0, -6.0, -4.0, -9.0, 5.0, -2.0, 5.0, 8.0, 7.0, 4.0, -3.0, 8.0, 6.0, 5.0, 6.0, 18.0, 14.0, 10.0, 13.0, 7.0, 6.0, 5.0, 11.0, 14.0, 13.0, 17.0, 18.0, 9.0, 14.0, 20.0, -12.0, -3.0, -11.0, 9.0, 16.0, 15.0, 11.0, 13.0, 14.0, 13.0, 13.0, 14.0, 20.0, 4.0, -5.0, 9.0, -19.0, -10.0, 9.0, -6.0, 10.0, 13.0, 5.0, 17.0, 7.0, 12.0, 4.0, 12.0, 18.0, 20.0, 19.0, 8.0, 7.0, 12.0, 13.0, 20.0, 5.0,20.0]
    info_score=[]
    game = len(info_score)
    info_game = {}
    hist_A, hist_C, cycles, elapse = [], [], [], []

    blob = gym.make('PongDeterministic-v4',render_mode = 'rgb_array') # 'human'
    # blob = gym.make('ALE/Pong-v5',render_mode = 'rgb_array') # 'human'
    blob.seed(0) # print(blob.action_space)
    N_ACTIONS = blob.action_space.n
    agent  = DQNAgent(N_ACTIONS)
    agent.A_file = A_file
    agent.C_file = C_file
    agent.J_file = J_file
    if load:
        load_agent(agent)
        info_game = load_game(agent.J_file)
        info_score = info_game['score']
    game = len(info_score)
    for key in info_game:
        # print(key)
        if key == 'lossActor':
            hist_A = info_game['lossActor']
        if key == 'lossCritic':
            hist_C = info_game['lossCritic']
        if key == 'cycles':
            cycles = info_game['cycles']
        if key == 'elapse':
            elapse = info_game['elapse']

    graphe_s = Graph.plot(info_score)
    graphe_s2 = Graph2.bar([1],lo=0)
    graphe_s3 = Graph3.bar([1])
    graphe_s4 = Graph4.etoile([[10,10]])

    #init round 0
    state, info = blob.reset()
    obs = agent.reset(state)
    score, score_P, score_C =0, 0, 0
    actions_record =[]

    # run = False
    #---------------------------------------------------------------------
    # Boucle principale
    start = tm.time()
    startCycle = tm.time()
    delta=[]
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
                if event.unicode == 'b':
                    save_agent(agent)
                if event.unicode == 'l':
                    load_agent(agent)
                if event.unicode == 'p':
                    # print(agent.pred)
                    print(np.unique(actions_record,return_counts=True),len(agent.actions))
                    print(f'Scores: {info_score}')
                if event.unicode == 'i':
                    print('Save info')
                    with open("pong.json", "w") as write_file:
                        json.dump(Infos, write_file, cls=NumpyArrayEncoder)
                if event.unicode == 'g':
                    graphe_s = Graph.plot(info_score)

        cycle += 1
        action, prediction = agent.act(obs)
        # - apply action
        state, r, done, _, info = blob.step(action)
        score += r
        if r == -1: score_C+=1
        if r ==  1: score_P+=1
        new_obs = agent._Obs(state)
        # - record
        agent.record(obs,r,action,info,prediction)
        actions_record.append(action)
        obs = new_obs
        # Print Info
        if (r==1 or r==-1):
            win.fill(0)
            dT = round((tm.time() - startCycle))
            moyenne = sum(info_score[-50:]) / max(len(info_score[-50:]),1)
            mess_nour = 'Partie #' + str(int(game)) + '   -> ' +str(dT)+' s'
            display(mess_nour, win, font, 850, 10)
            # message.render_to(win, (width-250, 30), mess_nour, (255,255,255,255))
            mess_nour = 'Cycles : [' + str(cycle)+']'
            display(mess_nour, win, font, 850, 33)
            mess_nour = 'Score : ' + str(int(score)) + ' § ['+str(round(moyenne*100)/100)+']'
            display(mess_nour, win, font, 850, 56)
            mess_nour = '/ ' + str(int(score_C)) + '  vs  ' + str(int(score_P)) + ' \\'
            display(mess_nour, win, font, 850, 88, fontsize=32)
            graphe_s = Graph.plot(info_score,score,echLO=score_P-21,echHI=21-score_C,moyenne=moyenne)
            # graphe_s3 = Graph3.bar([ 79,454, 250, 167, 311, 411],lo=0,largeur=(D_w//6)) ## TEST
            if h_A != None:
                mess_nour = 'loss Actor ' + str(round(h_A.history['loss'][0]*100000)/100000.0)
                display(mess_nour, win, font, 850, 121)
            if h_C != None:
                mess_nour = 'loss Critic ' + str(round(h_C.history['loss'][0]*100000)/100000.0)
                display(mess_nour, win, font, 850, 141)
            # win.blit(graphe_s,(25,(height-G_h)//2))
            if len(cycles)>0:
                graphe_s2 = Graph2.bar(cycles,lo=0)
                graphe_s4 = Graph4.etoile(elapse)
            display('(c) eCoucou',win,font,width-60,height-7,fontsize=10)
            win.blit(graphe_s,(15,15))
            win.blit(graphe_s2,(15,(height-75)))
            win.blit(graphe_s3,(15+10+G_w,158))
            win.blit(graphe_s4,(15+10+G_w,168+70+1))
            ga.display.flip()
        if done:
            info_score.append(score)
            moyenne = sum(info_score[-50:]) / len(info_score[-50:])
            actions = np.unique(actions_record,return_counts=True)[1]
            print(f'#Game: {game} -- {score} / {cycle}, moyenne= {moyenne}')
            print(actions,len(agent.actions))
            graphe_s = Graph.plot(info_score)
            graphe_s3 = Graph3.bar(actions,lo=0,largeur=(D_w//N_ACTIONS))
            start = cpt_elapse(start,delta)
            h_A, h_C = agent.replay()
            state, info = blob.reset()
            obs = agent.reset(state)
            start = cpt_elapse(start,delta)
            startCycle = start
            hist_A.append(h_A.history['loss'][0])
            hist_C.append(h_C.history['loss'][0])
            cycles.append(cycle)
            elapse.append(delta)
            game += 1
            # save_agent(agent)
            info_game= {'score':info_score, 'game': game, 'lossActor':hist_A, 'lossCritic': hist_C, 'elapse': elapse, 'cycles': cycles}
            save_game(info_game,agent.J_file)
            delta, actions_record, score, cycle ,score_C, score_P = [], [], 0, 0, 0, 0
        # ga.time.wait(20)
    return 0
######
if __name__ == '__main__':
    sys.exit(main(False,A_file='epNN000_Actor.h5',C_file='epNN000.h5',J_file='aep0_Game.json'))
    # sys.exit(main(True,A_file='a00_Actor.h5',C_file='a00_Critic.h5',J_file='a00_Game.json'))
    # sys.exit(run(A_file='00_Actor.h5',C_file='00_Critic.h5'))