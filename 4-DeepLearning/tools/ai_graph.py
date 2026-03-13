import json
import math
import os
from datetime import datetime, timedelta
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

VERSION = 'v3.3.2'

MONITOR_FILE = 'model/monitor.log'
TRAIN_INFO = 'model/train_info.json'
TARGET_BLOCK_DS1 = 2525599 #3367466 #5051199 # old wiki_raw 5963390
TARGET_BLOCK_DS2 = 1203058 #1619593 #2429390
TARGET_BLOCK_DS3 = 85440 #116091  #174137
MIXED = 0.25
LITT = 0.05
BATCH_SIZE = 4
GRAD_ACCUM = 96

xActual = 0
line_dict = {}

def calcul_epoch(step, target, mix, batch, grad, t_step):
    epoch = 0
    if mix > 0:
        epoch = int((target - step * batch) / (batch * mix * grad) + t_step)
    return epoch
    
def calcul_ema(data_loss, data_steps, compare=None):
    if len(data_loss) == 0:
        return 0, 0
    ema_loss = data_loss[0]
    # Trouve l'index le plus proche du point de comparaison
    if compare is not None:
        try:
            end = np.where(data_steps >= compare)[0][0]
        except IndexError:
            end = len(data_steps) - 1
    else:
        end = len(data_steps) - 1

    # Calcul de la moyenne mobile exponentielle
    for l in data_loss[:end+1]:
        ema_loss = 0.1 * l + (1 - 0.1) * ema_loss
    return ema_loss, end

def calcul_lr(it):
    # Utilisation des paramètres passés à l'init
    warmup = 1500
    max_iters = 30000
    lr_max = 0.00035
    lr_min = lr_max * 0.1
    # 1) Phase de warmup
    if it < warmup:
        return lr_max * it / warmup
    # 2) Phase de plateau bas
    if it > max_iters:
        return lr_min
    # 3) Phase de Cosine Decay
    decay_ratio = (it - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr_min + coeff * (lr_max - lr_min)


def predict_loss_projection(steps, losses, target_step, start_idx=-1):
    """
    Projection robuste de la Loss pour LLM.
    On bloque les coefficients dans une zone 'réelle'
    """
    # Modèle : L(s) = a * s^-b + c
    def power_law(s, a, b, c):
        return a * np.power(s, -b) + c

    # On garde une fenêtre glissante si start_idx n'est pas défini
    if start_idx == -1:
        start_idx = len(steps) // 10 
    
    fit_steps = np.array(steps[start_idx:])
    fit_losses = np.array(losses[start_idx:])

    try:
        # --- ASTUCE 1 : Les Bornes (Bounds) ---
        # a > 0
        # b (vitesse) : entre 0.01 (lent) et 0.8 (rapide). Un LLM dépasse rarement 0.5.
        # c (asymptote) : la loss finale théorique. Pour ton modèle, elle sera entre 1.5 et 3.5.
        lower_bounds = [0.1, 0.01, 1.5]
        upper_bounds = [1000, 0.8, 4.0] 
        
        # --- ASTUCE 2 : Poids dégressifs (Sigma) ---
        # On donne plus d'importance aux points RÉCENTS pour capturer la pente actuelle,
        # tout en gardant les anciens pour la courbure globale.
        sigmas = np.ones(len(fit_steps))
        # On divise l'erreur autorisée par 2 sur les 10% derniers points
        sigmas[-max(1, len(sigmas)//10):] = 0.5 

        # Estimation initiale réaliste
        p0 = [10, 0.2, 2.8] 
        
        popt, _ = curve_fit(
            power_law, fit_steps, fit_losses, 
            p0=p0, 
            bounds=(lower_bounds, upper_bounds),
            sigma=sigmas,
            maxfev=20000
        )
        
        # Génération de la projection
        future_steps = np.arange(steps[-1], target_step, 100)
        projection = power_law(future_steps, *popt)
        
        return future_steps, projection, popt
    except Exception as e:
        print(f"Erreur d'extrapolation : {e}")
        return None, None, None

def load_train_info(path="model/train_info.json"):
    with open(path, 'r') as f:
        return json.load(f)

def step_to_tokens(step, model_name, config):
    """Convertit un step spécifique d'un modèle en nombre de tokens total."""
    if model_name not in config:
        raise ValueError(f"Modèle {model_name} non trouvé dans le JSON")
    
    phases = config[model_name]
    total_tokens = 0
    
    for phase in phases:
        start = phase['start_step']
        end = phase['end_step']
        
        if step > start:
            # On calcule combien de steps ont été faits dans cette phase
            # Si le step est au-delà de la phase, on prend toute la phase, sinon le reste
            steps_in_phase = min(step, end) - start
            
            # Formule : Steps * EBS * Block_Size
            total_tokens += steps_in_phase * phase['ebs'] * phase['block_size']
            
    return total_tokens

def convert_steps_list(steps_list, model_name, config):
    """Macro fonction pour convertir une liste de steps en liste de millions de tokens."""
    # On divise par 1e6 pour avoir des chiffres lisibles sur le graphique (ex: 450M)
    return [step_to_tokens(s, model_name, config) / 1_000_000 for s in steps_list]

def predict_loss_projection_old(steps, losses, target_step, start_idx = -1):
    """
    ATTENTION : version un peu instable
    Projette la perte (Loss) vers un step futur.
    steps : liste des steps actuels (ex: [100, 200, ... 4100])
    losses : liste des valeurs de loss correspondantes
    target_step : le step jusqu'auquel on veut voir le futur (ex: 15000)
    """
    # Modèle de loi de puissance : L(s) = a * s^-b + c
    def power_law(s, a, b, c):
        return a * np.power(s, -b) + c

    # On ignore les tout premiers steps (warmup) pour ne pas fausser la courbe
    if start_idx == -1:
        start_idx = len(steps) // 5 
    fit_steps = np.array(steps[start_idx:])
    fit_losses = np.array(losses[start_idx:])

    try:
        # Estimation initiale pour aider l'algorithme
        p0 = [10, 0.5, 2.0] 
        popt, _ = curve_fit(power_law, fit_steps, fit_losses, p0=p0, 
                    bounds=([0, 0.01, 1.0], [np.inf, 1.0, 4.0]))
#        popt, _ = curve_fit(power_law, fit_steps, fit_losses, p0=p0, maxfev=10000)
        
        # Génération de la projection
        future_steps = np.arange(steps[-1], target_step, 100)
        projection = power_law(future_steps, *popt)
        
        return future_steps, projection, popt
    except Exception as e:
        print(f"Erreur d'extrapolation : {e}")
        return None, None, None

def check_efficiency(popt, current_step, target_step=60000):
    """
    Analyse l'efficacité de l'apprentissage à un horizon donné.
    """
    a, b, c = popt
    
    # 1. Calcul de la perte projetée à l'horizon (ex: 60k)
    # Formule : L(s) = a * s^-b + c
    loss_at_horizon = a * np.power(target_step, -b) + c
    
    # 2. Calcul de la pente (dérivée) au step actuel
    # La dérivée de a*s^-b + c est : -a * b * s^-(b+1)
    # Elle représente de combien la loss descend par step.
    current_slope = -a * b * np.power(current_step, -(b + 1))
    
    # 3. Calcul de la pente à l'horizon
    # Pour savoir si le modèle sera "mort" (plateau) ou s'il descendra encore
    future_slope = -a * b * np.power(target_step, -(b + 1))
    
    return {
        "loss_horizon": round(loss_at_horizon, 4),
        "current_slope_per_1000": round(current_slope * 1000, 5), # Gain pour 1000 steps
        "future_slope_per_1000": round(future_slope * 1000, 5),
        "is_plateau": abs(future_slope * 1000) < 0.005 # Seuil arbitraire de stagnation
    }

def add_label(ax, label, value, offset, left=1., color="gray", textAlign='left'):
    ax.annotate(
        label,
        xy=(left, value),
        xytext=(offset,0),
        xycoords=('axes fraction','data'),
        textcoords='offset points',
        fontweight='bold',
        fontsize=8,
        color=color,
        bbox=dict (facecolor='white',alpha=0.8, edgecolor='none',boxstyle='square,pad=0'),
        va='center',
        ha=textAlign
    )

def add_cursor_x(ax, label, value, ymin, offset, color="gray", textAlign='up'):
    ax.annotate(
        label,
        xy=(value, 0),
        xytext=(0, -offset),
        xycoords=('data', 'axes fraction'),
        textcoords='offset points',
        fontweight='bold',
        fontsize=8,
        color=color,
        bbox=dict (facecolor='white',alpha=0.8, edgecolor='none',boxstyle='square,pad=0'),
        ha='center',
        va=textAlign
    )

def on_mouse_move(event):
    global xActual, spdActual
    if event.inaxes:
        v_line = event.canvas.figure.v_line
        h_line = event.canvas.figure.h_line
        t_line = event.canvas.figure.t_line
        v_line.set_xdata([event.xdata, event.xdata])
        v_line.set_visible(True)
        ax = h_line.axes
        x, y = ax.transData.inverted().transform((event.x, event.y))
        h_line.set_ydata([y,y])
        h_line.set_visible(True)

        if x > xActual:
            lr = calcul_lr(x)
            elapse = ((x - xActual)*spdActual)
            eta = datetime.now() + timedelta(seconds=elapse)
            ETA = eta.strftime("%d-%B %H:%M")
            t_line.set_text(f"{ETA}\nlr = {lr:10.8f}\n[{round(x):5d} - {y:4.2f}]")
            t_line.xy = (x, y+0.05)
            t_line.set_visible(True)
        else:
            t_line.set_visible(False)
            
#    else:
#        v_line.set_visible(False)
        event.canvas.draw_idle()

def on_pick(event):
    global line_dict
    leg_item = event.artist
    if leg_item in  line_dict:
        orig_line = line_dict[leg_item][0]
        vis = not orig_line.get_visible()
        orig_line.set_visible(vis)
        leg_item.set_alpha(1. if vis else 0.2)
        event.canvas.draw_idle()

def ajouter_phase_train(ax, start, end, phases):
    for debut, fin, label, color in phases:
        if (debut < end) :
            ax.axvspan(debut, end, ymin=0, ymax=0.02, facecolor=color, alpha=1, edgecolor='gray', linewidth=0.1)
            xText = (max(debut,start) + min(fin,end))//2
            ax.text( xText, 0.01, label, transform = ax.get_xaxis_transform() , ha ='center', va = 'center', fontsize = 8, fontweight = 'bold', color = 'white')

def copyright(ax, ymin):
    label = f'eCoucou {VERSION}'
    print(f"-"*100,"\n",label,"-")
    ax.annotate(
        label,
        xy=(1, 0),
        xytext=(50,-15),
        xycoords=('axes fraction','axes fraction'),
        textcoords='offset points',
#        fontweight='bold',
        fontsize=6,
        color='black',
        bbox=dict (facecolor='white',alpha=0.8, edgecolor='none',boxstyle='square,pad=0'),
        va='top',
        ha='center'
    )
    

def plot_poussin_gap(
        y_min=2.6,
        y_max=3.3,
        x_max=70000,
        x_min=0,
        smooth=5,
        log_path="model/my_wiky_history.json",
        second="save/model_3/my_wiky_history.json",
        third=None,
        compare=None,
        annot_event=None,
        target=None,
        speed=None,
        raw=False,
        titre = "Training GPT Mac-M1",
        horizon = 30000,
        proj = 10,
        phase_train = [(0,1500,'WarmUp','#993311') , (1500,2000,'Stabilisation','#51719f'), (2000, 5000, 'Cristallisation','#619f71'), (5000, 10000, 'Grammaire', '#9c0056'), (10000, 20000, 'Polissage', '#101099') ],
        ds = False
    ):

    # data
    global xActual, spdActual, line_dict
    x_sens = (x_max - x_min) * 0.013

    info = load_train_info(TRAIN_INFO)

    # --- CHARGEMENT DATA 1 ---
    if not os.path.exists(log_path):
        print(f"❌ Fichier principal introuvable : {log_path}")
        return
    with open(log_path, "r") as f:
        data = json.load(f)

    # chargement monitor_file
    data_monitor = []
    with open(MONITOR_FILE, "r") as f:
        for line in f:
            if line.strip():
                data_monitor.append(json.loads(line))
    monitor_loss, monitor_ds1, monitor_ds2, monitor_ds3, monitor_mixed,  monitor_step_ds, monitor_grad, monitor_lr = [], [], [], [], [], [], [], []

    for d in data_monitor:
        monitor_loss.append(d['loss'])
        monitor_step_ds.append(d['step'])
        monitor_lr.append(float(d['lr'])*1.e4)
        monitor_grad.append(np.mean(d['grad_norm']))
        monitor_ds1.append( (d.get('dataset1',0)/TARGET_BLOCK_DS1*BATCH_SIZE) % 1 )
        monitor_ds2.append( (d.get('dataset2',0)/TARGET_BLOCK_DS2*BATCH_SIZE) % 1 )
        monitor_ds3.append( (d.get('dataset3',0)/TARGET_BLOCK_DS3*BATCH_SIZE) % 1 )
        monitor_mixed.append(d.get('dataset1',0)/(d.get('dataset1',0)+d.get('dataset2',0.1)+d.get('dataset3',0.1)))
    monitor_lr.insert(0,0)
    monitor_ds1.insert(0,0)
    monitor_ds2.insert(0,0)
    monitor_ds3.insert(0,0)
    monitor_mixed.insert(0,0)
    monitor_step_ds.insert(0,0)
    monitor_grad.insert(0,0)
    monitor_tokens = convert_steps_list(monitor_step_ds, "model_5", info)
 
    if compare == -1:
        compare = data["steps"][-1]
        # echéances epoch
    step_ds1 = data_monitor[-1]['dataset1']
    epoch_ds1 = calcul_epoch(step=step_ds1, target=TARGET_BLOCK_DS1, mix=(1-MIXED-LITT), batch=BATCH_SIZE, grad=GRAD_ACCUM , t_step=compare)
    step_ds2 = data_monitor[-1]['dataset2']
    epoch_ds2 = calcul_epoch(step=step_ds2, target=TARGET_BLOCK_DS2, mix=MIXED, batch=BATCH_SIZE, grad=GRAD_ACCUM , t_step=compare)
    step_ds3 = data_monitor[-1]['dataset3']
    epoch_ds3 = calcul_epoch(step=step_ds3, target=TARGET_BLOCK_DS3, mix=LITT, batch=BATCH_SIZE, grad=GRAD_ACCUM , t_step=compare)

    print("Epochs : ",epoch_ds1, epoch_ds2, epoch_ds3)
    steps = np.array(data["steps"])
    train_loss = np.array(data["train_loss"])
    val_loss = np.array(data["val_loss"])
    val_loss_wiki = np.array(data["val_loss_wiki"])
    val_loss_cult = np.array(data["val_loss_cult"])
    val_loss_litt = np.array(data["val_loss_litt"])
    gap = val_loss - train_loss
    tokens = convert_steps_list(steps, "model_5", info)

    # --- CHARGEMENT DATA 2 (COMPARAISON) ---

    if second and os.path.exists(second):
        with open(second, "r") as f2:
            data_second = json.load(f2)
        val_loss_second = np.array(data_second["val_loss"])
        steps_second = np.array(data_second["steps"])
        tokens_second = convert_steps_list(steps_second, "model_5", info)
        ema_loss_second, end_second = calcul_ema(val_loss_second, steps_second, compare=compare)
    else:
        val_loss_second, steps_second, tokes_second  = None, None, None
        ema_loss_second, end_second = 0, 0
        print(f"⚠️  Note : Second fichier de log non chargé ({second})")

    if third and os.path.exists(third):
        with open(third, "r") as f2:
            data_third = json.load(f2)
        val_loss_third = np.array(data_third["val_loss"])
        steps_third = np.array(data_third["steps"])
        ema_loss_third, end_third = calcul_ema(val_loss_third, steps_third, compare=compare)
    else:
        val_loss_third, steps_third = None, None
        ema_loss_third, end_third = 0, 0
        print(f"⚠️  Note : Troisieme fichier de log non chargé ({third})")

    # --- LISSAGE ET EMA ---
    def moving_average(x, w):
        if w <= 1:
            return x
        return np.convolve(x, np.ones(w), "valid") / w

    ema_loss, end = calcul_ema(val_loss, steps, compare=compare)
    ema_last, end_last = calcul_ema(val_loss, steps, compare=None)


    # --- STYLE ET FIGURE ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(15, 7.5))  #, layout="constrained")
    # Ajuste les marges manuellement (0.1 = 10% de marge)
    plt.subplots_adjust(left=0.03, right=0.88, top=0.95, bottom=0.05)
    # On créé la barre verticale avec le curseur ETA
    xActual = data_monitor[-1]['step']
    spdActual = data_monitor[-1]['elapse'] / (xActual - data_monitor[-2]['step'])    
    
    fig.v_line = ax.axvline( color='#aaaaaa',linestyle='-.',linewidth=0.7, visible=False, zorder=10)
    fig.h_line = ax.axhline(0, color='#aaaaaa',linestyle='-.',linewidth=0.7, visible=False, zorder=100)
    fig.t_line = ax.annotate("", xy=(0,1), xycoords=('data'), xytext = (10,10), textcoords = 'offset points', rotation=0, color="black",
            fontsize=7, verticalalignment="bottom",
            bbox=dict (facecolor='#ccddff',alpha=1., edgecolor='#aabbdd',boxstyle='square,pad=0.4'),
            ha = 'center',
            va = 'center'
        )
    
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    # Définit un repère principal tous les 2500
    if (x_max-x_min) < 5000:
        x_ticks = 100
    else:
        x_ticks = 200
    epsilon = 1e-5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(x_ticks))
    # Optionnel : Ajoute des petits traits (minuscules) tous les 500 pour plus de précision
#    ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    # Rotation des labels pour éviter qu'ils ne se chevauchent s'il y en a trop
    ax.set_yticks(np.arange(y_min, y_max + epsilon, 0.1)) #(y_max-y_min)/10))
    ax.tick_params(axis='y', colors='orange',labelsize=8, length = 2, width=0.7)
    ax.tick_params(axis='x', which='major', length=5, width=1, colors='black',labelsize=8)
    ax.tick_params(axis='x', which='minor', length=2, width=0.5, colors='gray')
    plt.xticks(rotation=45)

    label = f"\u25b2"
    add_cursor_x(ax, label, data_monitor[-1]['step'], y_min, 2, color="black", textAlign='top')
    
    #----------------------------------------------------------------------
    # Zones de couleurs pour le GAP (Overfitting check)
    for i in range(len(steps) - 1):
        g = gap[i]
        c = "#acffaa" if g <= 0.05 else ("#ffff00" if g <= 0.12 else "red")
        ax.axvspan(steps[i], steps[i + 1], facecolor=c, alpha=0.1)

    # Bandeau en bas du graph
    ajouter_phase_train(ax, x_min, x_max, phase_train)
    
    # Calcul de la couverture du dataset
    prc = []
    prc_train = []
    prc_ds1, prc_ds2 = [], []
    ema_smooth, ema_smooth_raw = [], []
    ema_smooth_2 = []
    ema_smooth_3 = []

    for i in range(len(steps)):
        ema_smooth.append(calcul_ema(val_loss[: i + 1], steps[: i + 1])[0])

    if val_loss_second is not None:
        for i in range(len(steps_second)):
            ema_smooth_2.append(calcul_ema(val_loss_second[: i + 1], steps_second[: i + 1])[0])

    if val_loss_third is not None:
        for i in range(len(steps_third)):
            ema_smooth_3.append(calcul_ema(val_loss_third[: i + 1], steps_third[: i + 1])[0])

    if raw:
        for i in range(len(monitor_step_ds)):
            ema_smooth_raw.append(calcul_ema(monitor_loss[: i + 1], monitor_step_ds[: i + 1])[0])

    # --- Calcul des stats ---
    steps_proj, ema_proj, popt = predict_loss_projection(steps, val_loss, x_max, proj)
    stats = check_efficiency(popt, steps[-1], horizon)
    status_plateau = f"\u2714 OUI" if stats["is_plateau"] else f"\u2715 NON"
#    print('projection  :',popt)
#    print(stats)
    # --- Préparation du texte ---
    info_text = (
        f"PREDICTION (Target: {horizon/1000:.0f}k)\n"
        f"---------------------------\n"
        f"Loss Horizon : {stats['loss_horizon']}\n"
        f"Slope/1k (now) : {stats['current_slope_per_1000']:.4f}\n"
        f"Slope/1k ({horizon/1000:.0f}k) : {stats['future_slope_per_1000']:.4f}\n"
        f'Plateau : {status_plateau}'
    )

    lines = []

    # --- TRACÉ DES COURBES ---
    if smooth > 1:
        s_smooth = steps[smooth - 1 :]
        l_smooth = moving_average(val_loss, smooth)
        line_ax_1, = ax.plot(s_smooth, moving_average(train_loss, smooth), label=f"Train (Smooth {smooth})", color="#3498db", lw=1.5)
        lines.append(ax.plot(s_smooth, l_smooth, label=f"Val (Smooth {smooth})", color="#e67e22", lw=1.5))
        ax.text(monitor_step_ds[-1], train_loss[-1], s=f"\u25c4 {train_loss[-1]:.3f}", color="#3498db", rotation=0, fontsize=9, horizontalalignment="left", va='center')
        ax.text(monitor_step_ds[-1], val_loss[-1], s=f"\u25c4 {val_loss[-1]:.3f}", color="#e67e22", rotation=0, fontsize=9, horizontalalignment="left", va='center')
        # Tracé des courbes de comparaison si disponibles - second
        if val_loss_second is not None:
            s_sm_sec = steps_second[smooth - 1 :]
            lines.append(ax.plot( s_sm_sec,moving_average(val_loss_second, smooth), label="Model 2 (Comparison)",color="#fb59b6",lw=0.9, linestyle="--"))
            lines.append(ax.plot(steps_second, ema_smooth_2, label="EMA lissée second", color="#cf35b8", linestyle="-.", lw=1.))

        # Tracé des courbes de comparaison si disponibles - third
        if val_loss_third is not None:
            s_sm_sec = steps_third[smooth - 1 :]
            lines.append(ax.plot( s_sm_sec, moving_average(val_loss_third, smooth), label="Model 1 (Comparison)",color="#5bf9b6",lw=0.9,linestyle="--"))
            lines.append(ax.plot(steps_third, ema_smooth_3, label="EMA lissée third", color="#70eef3", linestyle="-.", lw=1.))

        lines.append(ax.plot(steps, train_loss, color="#3498db", alpha=0.4, lw=1))
        lines.append(ax.plot(steps, val_loss, color="#fa8118", alpha=0.4, lw=1))
        lines.append(ax.plot(steps, ema_smooth, label="EMA lissée", color="#ff0000", linestyle="-.", lw=1.5))
        lines.append(ax.plot(steps_proj, ema_proj, label="EMA Projectionf", color="#e67e22", linestyle=":", lw=1.1))
#        ax.text(steps[-1]+x_sens/10, ema_smooth[-1], s=f"{ema_smooth[-1]:.3f}", color="red", rotation=0, fontsize=9, horizontalalignment="left")
        if raw:
            lines.append(ax.plot(monitor_step_ds[5:], moving_average(monitor_loss,5)*1., label="train: raw data", color="#000000", linestyle=":", lw=0.5))
 #           ax.text(monitor_step_ds[-1]+x_sens/10, ema_smooth_raw[-1], s=f"{ema_smooth_raw[-1]:.3f}", color="black", rotation=0, fontsize=9, horizontalalignment="left")
    else:
        lines.append(ax.plot(steps, train_loss, label="Train Loss", color="#3498db", lw=1.5))
        lines.append(ax.plot(steps, val_loss, label="Val Loss", color="#fa831b", lw=2))

        if ds:
            lines.append(ax.plot(steps, val_loss_wiki, label="Val Loss Wikipédia", color="Blue", linestyle="--", lw=0.9, alpha=0.7))
            lines.append(ax.plot(steps, val_loss_cult, label="Val Loss CulturaX", color="Orange", linestyle=":", lw=0.9, alpha=0.7))
            lines.append(ax.plot(steps, val_loss_litt, label="Val Loss Littéraire", color="Green", linestyle=":", lw=0.9, alpha=0.7))
        
        ax.text(monitor_step_ds[-1], train_loss[-1], s=f"\u25c4 {train_loss[-1]:.3f}", color="#3498db", rotation=0, fontsize=9, horizontalalignment="left", va='center')
        ax.text(monitor_step_ds[-1], val_loss[-1], s=f"\u25c4 {val_loss[-1]:.3f}", color="#e67e22", rotation=0, fontsize=9, horizontalalignment="left", va='center')
        lines.append( ax.plot(steps, ema_smooth, label="EMA lissée", color="#ff0000", linestyle="-.", lw=1.5))
#        ax.text(steps[-1]+x_sens, ema_smooth[-1], s=f"{ema_smooth[-1]:.3f}", color="red", rotation=0, fontsize=9, horizontalalignment="left")
        lines.append(ax.plot(steps_proj, ema_proj, label="EMA Projectionf", color="#e67e22", linestyle=":", lw=1.1))
        if val_loss_second is not None:
            lines.append(ax.plot(steps_second, val_loss_second, label="Model v4", color="#fb59b6", linestyle="--", lw=0.9))
        if raw:
            lines.append(ax.plot(monitor_step_ds[6:], moving_average(monitor_loss,6)*1, label="train: raw data", color="#000000", linestyle=":", lw=0.5))
#            ax.text(monitor_step_ds[-1]+x_sens/10, monitor_loss[-1]*1.01, s=f"{monitor_loss[-1]:.3f}", color="black", rotation=0, fontsize=9, horizontalalignment="left")

    # --- TARGET --- (Ligne Horizontale) -------------
    if (target is not None) & (target>y_min):
        ax.axhline( y=target, color="#aa55bb", alpha=0.7, linestyle="-.", lw=1.5)
#        ax.text( 1000, (target+0.01), s="Cible à " ,color="#aa55bb", rotation=0, fontsize=9, horizontalalignment="left")
        ax.text( max(500, x_min+200), (target+0.01), s=("Loss Cible à "+f"{target:.2f}") ,color="#aa55bb", rotation=0, fontsize=9, horizontalalignment="left")
    # --- ANNOTATIONS (Lignes verticales et texte) ---
    if annot_event:
        for ev in annot_event:
            if ev["step"] >= x_min:
                if ((speed is not None) & (ev['step']>steps[-1])):
                    elapse = ((ev['step']-steps[-1])*speed)
                    eta = datetime.now() + timedelta(seconds=elapse)
                    ETA = " ("+eta.strftime("%d %H:%M")+")"
                else:
                    ETA = ""
                ax.axvline( x=ev["step"], color=ev.get("color", "black"), linestyle="--", alpha=0.6, lw=ev.get("lw", 1))
                ax.text( ev["step"], y_min + 0.06, f"[{ev['step']:5d}] - " + ev["label"] + ETA, rotation=90, color=ev.get("color", "black"), fontsize=9, ha= 'right' , verticalalignment="bottom")

    if compare:
        ax.axvline( x=compare, color="black", linestyle="--", alpha=0.7, lw=0.8)
        ax.text( compare+x_sens/10, y_max, "Compare " , rotation=90, color="black", fontsize=9, verticalalignment="top")
        label = f"[{epoch_ds1}] Epoch 1 - Dataset 1 (Wiki)"
        if epoch_ds1<x_max:
            ax.axvline( x=epoch_ds1, color="magenta", linestyle="--", alpha=0.7, lw=0.8)
            ax.text( epoch_ds1, y_min + 0.06, label , rotation=90, color="magenta", fontsize=9, ha='right', verticalalignment="bottom")
        label = f"[{epoch_ds2}] Epoch 1 - Dataset 2 (CulturaX)"
        if epoch_ds2<x_max :
            ax.axvline( x=epoch_ds2, color="magenta", linestyle="--", alpha=0.7, lw=0.8)
            ax.text( epoch_ds2, y_min + 0.06, label , rotation=90, color="magenta", fontsize=9, ha='right', verticalalignment="bottom")
        label = f"[{epoch_ds3}] Epoch 1 - Dataset 3 (Littéraire)"
        if epoch_ds3<x_max :
            ax.axvline( x=epoch_ds3, color="magenta", linestyle="--", alpha=0.7, lw=0.8)
            ax.text( epoch_ds3, y_min + 0.06, label , rotation=90, color="magenta", fontsize=9, ha='right', verticalalignment="bottom")

    # --- RÉGLAGES FINAUX ---
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.set_title(
        f"{titre} | Step: {steps[-1]} | Gap: {gap[-1]:.4f}", fontsize=14
    )
   
    ax.grid(True, alpha=0.3)
    add_label(ax,  f"\u25c4 {monitor_loss[-1]:.3f}", monitor_loss[-1], offset=0, left=0.)
    add_label(ax,  f"\u25ba", train_loss[-1], offset=0, left=0., color="#3498db", textAlign="right")
    add_label(ax,  f"\u25ba", val_loss[-1], offset=0, left=0., color="#e67e22", textAlign="right")
    add_label(ax,  f"\u25c4 {ema_smooth[-1]:.3f}", ema_smooth[-1], offset=0, left=0., color="#ff0000", textAlign="left")
    # On affiche les dataset sur l'axe des ordonnées
    add_label(ax,  f"\u25ac", val_loss_wiki[-1], offset=0, left=0., color = "Blue", textAlign="center")
    add_label(ax,  f"\u25ac", val_loss_cult[-1], offset=0, left=0., color = "Orange", textAlign="center")
    add_label(ax,  f"\u25ac", val_loss_litt[-1], offset=0, left=0., color = "Green", textAlign="center")

    # Après avoir tracé la loss sur ax
    ax_droite = ax.twinx()
    ax_droite.grid(False)
    y_1_max = math.ceil(20 * min(1.0, 1.25*max(max(monitor_ds1),max(monitor_ds2),max(monitor_ds3))) ) / 20.
    y_1_tick = ( y_1_max / 20.)
    ax_droite.set_ylim(0.0, y_1_max)
    ax_droite.set_yticks(np.arange(0., y_1_max + epsilon, y_1_tick)) #(y_max-y_min)/10))
    ax_droite.tick_params(axis='y', colors='gray',labelsize=8, length = 2, width=0.7)
    # 2 options pour afficher des %
#    ax_droite.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax_droite.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{(100*x):.1f}%'))
    #ax_droite.plot(steps, prc, color="#7BFFB0", alpha=0.7, label="%", lw=0.7)
    #ax_droite.plot(steps, prc_train, color="#f6881b", alpha=0.7, label="%", lw=0.7)
    lines.append(ax_droite.plot(monitor_step_ds, monitor_ds1, color="#7B0080", alpha=0.7, label="Wiki %", lw=0.8))
    ax_droite.text( monitor_step_ds[-1]+x_sens/10,monitor_ds1[-1] , "wikipédia" , color="blue", fontsize=9, va='center', horizontalalignment="left")
    lines.append(ax_droite.plot(monitor_step_ds, monitor_ds2, color="#7B0080", alpha=0.7, label="Cult %", lw=0.8))
    ax_droite.text( monitor_step_ds[-1]+x_sens/10,monitor_ds2[-1] , "culturaX" , color="orange", fontsize=9, va='center', horizontalalignment="left")
    lines.append(ax_droite.plot(monitor_step_ds, monitor_ds3, color="#7B0080", alpha=0.7, label="Litt %", lw=0.8))
    ax_droite.text( monitor_step_ds[-1]+x_sens/10,monitor_ds3[-1] , "littéraire" , color="green", fontsize=9, va='center', horizontalalignment="left")
#    ax_droite.plot(monitor_step_ds, monitor_mixed, color="#7B3F80", alpha=0.7, label="%", lw=0.8)
#    ax_droite.plot(steps, prc_ds1, color="#7BFFB0", alpha=0.7, label="%", lw=0.7)
#    ax_droite.plot(steps, prc_ds2, color="#7BFFB0", alpha=0.7, label="%", lw=0.7)
    # ax_droite.set_ylabel("Token vus", color="magenta")
    add_label(ax_droite,  f"\u25c4 {monitor_ds1[-1]*100:.1f}%", monitor_ds1[-1], 0, color = 'blue')
    add_label(ax_droite,  f"\u25c4 {monitor_ds2[-1]*100:.1f}%", monitor_ds2[-1], 0, color = 'orange')
    add_label(ax_droite,  f"\u25c4 {monitor_ds3[-1]*100:.1f}%", monitor_ds3[-1], 0, color = 'green')
    
    ax_droite_2 = ax.twinx()
    ax_droite_2.grid(False)
    y_2_max = max(1.0, max(monitor_grad[-15:]))
    ax_droite_2.set_ylim(0. , y_2_max)
    ax_droite_2.spines['right'].set_position(("outward",34))
    ax_droite_2.set_yticks(np.arange(0., y_2_max+epsilon, 0.1)) #(y_max-y_min)/10))
    ax_droite_2.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))
    ax_droite_2.tick_params(axis='y', which='major',colors='gray',labelsize=8, length = 3.5, width=0.8)
    ax_droite_2.tick_params(axis='y', which='minor',colors='gray', length = 2, width=0.5)
    lines.append(ax_droite_2.plot(monitor_step_ds, monitor_grad, color="#7B8c80", alpha=0.7, label="Gradient norm", lw=0.6))

    add_label(ax_droite_2,  f"\u25c4 {monitor_grad[-1]:.3f}", monitor_grad[-1], offset=34)

    ax_droite_3 = ax.twinx()
    ax_droite_3.grid(False)
    y_3_max = 3.5 #max(monitor_lr) #0.00002
    ax_droite_3.set_ylim(0.0, y_3_max)
    ax_droite_3.spines['right'].set_position(("outward",65))
    ax_droite_3.set_yticks(np.arange(0., y_3_max+epsilon, 0.1)) #(y_max-y_min)/10))
    ax_droite_3.tick_params(axis='y', colors='gray',labelsize=8)
    ax_droite_3.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
    ax_droite_3.tick_params(axis='y', which='major',colors='gray',labelsize=8, length = 3.5, width=0.8)
    ax_droite_3.tick_params(axis='y', which='minor',colors='gray', length = 2, width=0.5)
    lines.append(ax_droite_3.plot(monitor_step_ds, monitor_lr, color="#5b6c60", alpha=0.8, label="lr", linestyle=':', lw=.9))

    add_label(ax_droite_3,  f"\u25c4 {monitor_lr[-1]:.2f}e-4", monitor_lr[-1], offset=65)

    # --- Affichage de l'encart ---
    plt.text(0.5, 0.80, info_text, transform=plt.gca().transAxes, 
         fontsize=9, verticalalignment='bottom', family='monospace',
         bbox=dict(boxstyle='round', facecolor='#ccddff', edgecolor="#aabbdd", alpha=0.5))   # facecolor='wheat'

    # Gestion de la légence clickable
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_droite.get_legend_handles_labels()
    h3, l3 = ax_droite_2.get_legend_handles_labels()
    h4, l4 = ax_droite_3.get_legend_handles_labels()
    handles = h1 + h2 + h3 + h4
    labels = l1 + l2 + l3 + l4
    
    leg = ax_droite_3.legend(handles, labels, loc="upper right", frameon= True)
    line_dict = {}
    for leg_line, leg_text, orig_line in zip(leg.get_lines(),leg.get_texts(), lines):
        leg_line.set_picker(True)
        leg_text.set_picker(True)
        line_dict[leg_text] = orig_line
        line_dict[leg_line] = orig_line

    fig.canvas.mpl_connect('pick_event', on_pick)

    copyright(ax, y_min)
    
    # plt.tight_layout() # modifié avec le constraint dans figsplot
    plt.savefig("model/screenshot/analyse_poussin_results.png")
    # print(f"✅ Analyse terminée. Image : analyse_poussin_results.png")
    plt.show()

    # Affichage des stats dans le terminal ()
    print("-" * 30)
    print(f"Dernier Step: {steps[-1]} | Gap actuel: {gap[-1]:.4f} | EMA loss : {ema_last:.3f}")
    if val_loss_second is not None:
        print(f"COMPARE POINT ANALYSIS: {steps[end]} / {steps_second[end_second]}") #" / {steps_third[end_third]}")
        print(f"  - Val Loss: {val_loss[-1]:.4f} vs {val_loss_second[end_second]:.4f}") #" ➥ {val_loss_third[end_third]:.4f}")
        print(f"  - EMA Loss: {ema_loss:.3f} vs {ema_loss_second:.3f}") #" ➥ {ema_loss_third:.3f}")


    tokens_m5 = convert_steps_list(steps, "model_5", info)
#    tokens_m5 = convert_steps_list(steps_m5, "model_5", info)
#    print(tokens_m5[:100])

