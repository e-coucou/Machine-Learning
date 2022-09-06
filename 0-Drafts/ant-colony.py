from ast import Del, Global
from tkinter import *
from typing import List 
import numpy as np
from scipy.spatial import distance
from scipy.linalg import fractional_matrix_power
from sklearn.metrics import d2_tweedie_score, fbeta_score

canvas_width = 600
canvas_height = 400
V_size = 25
Villes=np.empty(0)
nAnts = 400
nIter = 100
Liste = []
alpha = 1 # à 0 uniquement la visibilité
beta = 2 # à 0 uniquement les phéromones
rho = 0.3 # coeff d'évaporation
Q = 1
fBest = 1.3
T = np.array((V_size,V_size))

def init_all():
    global T
    val = 1/V_size
    T = np.full((V_size,V_size),val)

def dist(v0,v1):
    return distance.euclidean(v0, v1)

def random_villes(nb=5,xmax=600,ymax=400):
    villes=[]
    edge=50
    for v in range(nb):
        x=np.random.randint(edge,xmax-edge)
        y=np.random.randint(edge,ymax-edge)
        villes.append([x,y])
    return villes

def draw_villes(villes,couleur='black',size=3):
    s=size
    for v in villes:
        x = v[0]
        y = v[1]
        # print('x=',x,'y=',y)
        canvas.create_oval(x-s,y-s,x+s,y+s,fill=couleur)

def update_villes():
    canvas.delete('all')
    villes = random_villes(V_size,canvas_width,canvas_height)
    draw_villes(villes)
    return(villes)

def update():
    global Villes, Liste, T
    Villes = update_villes()
    Liste = np.arange(V_size)
    init_all()
    print(Villes,Liste)

def ville_choix(villes,liste):
    # n=np.random.randint(0,V_size)
    # s = Villes[n]
    s = np.random.choice(liste)
    liste = np.setdiff1d(liste,s)
    sel=[]
    sel.append([villes[s][0],villes[s][1]])
    return sel, liste,s

def calcul_P(origine,visibilite):
    p = np.zeros(visibilite.size)
    sum=0
    for j in range(visibilite.size):
        sum = sum + pow(T[origine][j],alpha)*pow(visibilite[j],beta)
    for l in range(visibilite.size):
        p[l] = pow(T[origine,l],alpha)*pow(visibilite[l],beta)/sum
    # print(p)
    return p

def test():
    for i in range(1000):
        choix()

def choix():
    global Liste
    canvas.delete('all')
    draw_villes(Villes)
    Liste = np.arange(V_size)
    J = np.empty(0,dtype=int)
    o,Liste,next = ville_choix(Villes,Liste)
    J = np.append(J,next)
    # print(Liste,next)
    draw_villes(o,couleur='Yellow',size=10)
    v=o
    while (Liste.size > 0):
        next = choix_update(v,next)
        J = np.append(J,next)
        Liste = np.setdiff1d(Liste,next)
        v=[]
        v.append([Villes[next][0],Villes[next][1]])
    J = np.append(J,J[0])
    # print(J)
    Long=0
    for i in range(J.size-1):
        # print('v0',Villes[J[i]],J[i+1])
        Long=Long+dist(Villes[J[i]],Villes[J[i+1]])
    Delta = Q/Long
    for x in range(J.size-1):
        i=J[x]
        j=J[x+1]
        T[i][j] = (1-rho)*T[i][j] + Delta
        T[j][i] = (1-rho)*T[j][i] + Delta

    for x in range(J.size-1):
        i=J[x]
        j=J[x+1]
        x0 = Villes[i][0]
        y0 = Villes[i][1]
        x1 = Villes[j][0]
        y1 = Villes[j][1]
        canvas.create_line(x0,y0,x1,y1,fill='blue',width=5)

    Long = round(Long/1000,1)
    canvas.create_text(50, 15, text=Long, font="Arial 16 italic", fill="black")
    fenetre.update()
    fenetre.update_idletasks()

def choix_update(v,id):
    global Liste
    if (Liste.size > 0):
        v_dist = np.zeros(V_size)
        draw_villes(v,couleur='red',size=3)
        for l in Liste:
            d = dist(v,Villes[l])
            v_dist[l]=1/d
        P_liste = calcul_P(id,v_dist)
        v_dist_sort = np.argsort(v_dist)[::-1]
        P_liste_sort = np.argsort(P_liste)[::-1]
        # print(Liste,v_dist_sort)
        x0 = v[0][0]
        y0 = v[0][1]
        for p in P_liste_sort[:min(1,Liste.size)]:
        # for p in v_dist_sort[:min(4,Liste.size)]:
            x = Villes[p][0]
            y = Villes[p][1]
            canvas.create_line(x0,y0,x,y,fill='red')
        return P_liste_sort[0]


def new_choix_ville(origine,liste):
    v_dist = np.zeros(V_size)
    for v in liste:
        d = dist(Villes[v],Villes[origine])
        v_dist[v] = 1/ d
    p_dist = calcul_P(origine,v_dist)
    ret = np.argsort(p_dist)[::-1][0]
    return ret,p_dist

def new_choix_ville_M(origine,liste,mat):
    # print(origine,liste)
    p = np.zeros(V_size)
    mul = np.zeros(V_size)
    for n in liste:
        mul[n] = 1
    # p_den = np.dot(mat,mul)[origine]
    p_dist = np.multiply(mat,mul.T)[origine]
    # print(origine,liste,mat[origine],p_den[origine])
    ret = np.argsort(p_dist)[::-1][0]
    return ret,p_dist

def print_T():
    max = np.max(T)
    print(max,T)
    for i in range(V_size):
        for j in range(V_size):
            x0 = Villes[i][0]
            y0 = Villes[i][1]
            x1 = Villes[j][0]
            y1 = Villes[j][1]
            wT = T[i][j] / max * 10
            canvas.create_line(x0,y0,x1,y1,fill='black',width=wT)
    canvas.create_text(50, 35, text=round(max,2), font="Arial 16 italic", fill="black")

def compute():
    global Villes
    print(Villes)
    canvas.delete('all')
    draw_villes(Villes)
    n=np.random.randint(0,V_size)
    s = Villes[n]
    # s = np.random.choice(Villes)
    sel=[]
    sel.append([s[0],s[1]])
    print(sel)
    draw_villes(sel,couleur='red',size=6)
    x0 = s[0]
    y0 = s[1]
    for v in Villes:
        x = v[0]
        y = v[1]
        canvas.create_line(x0,y0,x,y,fill='grey')

def draw_line(id0,id1,couleur='red',size=1):
    x0 = Villes[id0][0]
    y0 = Villes[id0][1]
    x1 = Villes[id1][0]
    y1 = Villes[id1][1]
    canvas.create_line(x0,y0,x1,y1,fill=couleur,width=size)

def compute_dist_inverse(villes):
    s=V_size
    ret = np.zeros((s,s))
    for i in range(s):
        for j in range(s):
            if (i != j):
                d = pow(1/ dist(villes[i],villes[j]),alpha)
                ret[i][j]=d
    return ret

def matrix_power(mat,e):
    d1=mat.shape[0]
    d2=mat.shape[0]
    ret = np.zeros((d1,d2))
    # print(d1,d2)
    for i in range(d1):
        for j in range(d2):
            if (i != j):
                ret[i][j]=pow(mat[i][j],e)
            else:
                ret[i][j]=0
    return ret

def new_algo():
    global T,Q
    Lbest = 10000
    init_all()
    m_visibility = compute_dist_inverse(Villes)
    # print(Villes)
    # print(T)
    # print(m_visibility)
    # print(m_Tau)
    # print(m_Delta)
    canvas.delete('all')
    draw_villes(Villes)
    # pour chaque tour
    for t in range(nIter):
        m_Tau = matrix_power(T,beta)
        m_Delta = np.multiply(m_visibility,m_Tau)
        # print(m_Delta)
        # pour chaque fourmi
        Tdelta = np.zeros((V_size,V_size))
        for k in range(nAnts):
            kLong=10000
            non_visitee = np.arange(V_size)
            J = np.empty(0,dtype=int)
            a = V_size
            # choisir une ville au hasard
            first = np.random.choice(non_visitee)
            J = np.append(J,first)
            non_visitee = np.setdiff1d(non_visitee,first)
            next = first
            # pour chaque ville non visitée
            while (non_visitee.size > 0):
                # o=next
                # choisir la meilleure option
                next, liste = new_choix_ville_M(next,non_visitee,m_Delta)
                # nextOld, listeOld = new_choix_ville(o,non_visitee)
                # print('Compare',next,nextOld, liste, listeOld)
                J = np.append(J,next)
                non_visitee = np.setdiff1d(non_visitee,next)
                # draw_line(o,next,'green',a)
                # a = a-1
            J = np.append(J,first)
            #-----
            # déposer les phéromones
            Long=0
            for i in range(J.size-1):
                # print('v0',Villes[J[i]],J[i+1])
                Long=Long+dist(Villes[J[i]],Villes[J[i+1]])
            if (Long<=Lbest):
                Lbest = Long
                Q=Lbest*fBest
                Best = np.copy(J)
            else:
                Q=Lbest
            if (Long < kLong):
                kBest =np.copy(J)
            Delta = Q/Long
            for x in range(J.size-1):
                i=J[x]
                j=J[x+1]
                Tdelta[i][j] = Delta #* m_visibility[i][j]
                Tdelta[j][i] = Delta #* m_visibility[j][i]
            # max = np.max(T)

        # évaporer les pistes ...
        T = T*(1-rho) + Tdelta
        # on affiche à chaque itération
        max = np.max(T)
        canvas.delete('all')
        for x in range(J.size-1):
            a = T[kBest[x]][kBest[x+1]] / max *12
            draw_line(kBest[x],kBest[x+1],'grey',a)
        # for x in range(J.size-1):
        #     a = T[J[x]][J[x+1]] / max *12
        #     draw_line(J[x],J[x+1],'grey',a)
        for x in range(Best.size-1):
            draw_line(Best[x],Best[x+1],'blue',2)
        draw_villes(Villes,couleur='blue',size=4)
        draw_villes([Villes[J[0]]],couleur='red',size=6)
        Long = round(Long,1)
        canvas.create_text(50, 15, text=Long, font="Arial 14", fill="black")
        canvas.create_text(150, 15, text=round(Lbest,1), font="Arial 14", fill="black")
        canvas.create_text(canvas_width - 30, 15, text=round(t,0), font="Arial 16", fill="black")
        canvas.create_text(canvas_width-30, 30, text=round(max,2), font="Arial 12", fill="black")
        # canvas.create_text(canvas_width-60, 30, text=round(np.min(T),2), font="Arial 12", fill="black")
        canvas.create_text(canvas_width-60, 30, text=round(Delta,2), font="Arial 12", fill="black")
        fenetre.update()
        # print(T)
    # on affiche les résultats ...




init_all()
fenetre = Tk()
canvas = Canvas(fenetre, width=canvas_width, height=canvas_height, background='white')


# ligne1 = canvas.create_line(75, 0, 75, 120)
# ligne2 = canvas.create_line(0, 60, 150, 60)
# txt = canvas.create_text(75, 60, text="Cible", font="Arial 16 italic", fill="blue")

canvas.pack()
Villes=update_villes()
# bouton de sortie
Button(fenetre, text="FIN", command=fenetre.quit).pack(side='left',padx=5,pady=5)
Button(fenetre, text="Update", command=update).pack(side='right',padx=5,pady=5)
Button(fenetre, text="Compute", command=compute).pack(side='right',padx=5,pady=5)
Button(fenetre, text="Test", command=test).pack(side='right',padx=5,pady=5)
Button(fenetre, text="NEW", command=new_algo).pack(side='right',padx=5,pady=5)
Button(fenetre, text="Pheromone", command=print_T).pack(side='right',padx=5,pady=5)

fenetre.mainloop()
