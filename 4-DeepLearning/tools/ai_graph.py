import json
import math
import os
from datetime import datetime, timedelta
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

MONITOR_FILE = 'model/monitor.log'
TRAIN_INFO = 'model/train_info.json'
TARGET_BLOCK_DS1 = 3367466 #5051199 # old wiki_raw 5963390
TARGET_BLOCK_DS2 = 1619593 #2429390
TARGET_BLOCK_DS3 = 116091  #174137
MIXED = 0.35
LITT = 0.1
BATCH_SIZE = 8
GRAD_ACCUM = 48

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
        proj = 10
    ):

    # data
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
    monitor_loss, monitor_ds1, monitor_ds2, monitor_ds3, monitor_mixed,  monitor_step_ds, monitor_grad = [], [], [], [], [], [], []
    for d in data_monitor:
        monitor_loss.append(d['loss'])
        monitor_step_ds.append(d['step'])
        monitor_grad.append(np.mean(d['grad_norm']))
        monitor_ds1.append( (d.get('dataset1',0)/TARGET_BLOCK_DS1*BATCH_SIZE) % 1 )
        monitor_ds2.append( (d.get('dataset2',0)/TARGET_BLOCK_DS2*BATCH_SIZE) % 1 )
        monitor_ds3.append( (d.get('dataset3',0)/TARGET_BLOCK_DS3*BATCH_SIZE) % 1 )
        monitor_mixed.append(d.get('dataset1',0)/(d.get('dataset1',0)+d.get('dataset2',0.1)+d.get('dataset3',0.1)))
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
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # Définit un repère principal tous les 2500
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
    # Optionnel : Ajoute des petits traits (minuscules) tous les 500 pour plus de précision
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(500))
    # Rotation des labels pour éviter qu'ils ne se chevauchent s'il y en a trop
    ax.set_yticks(np.arange(y_min, y_max, 0.1)) #(y_max-y_min)/10))
    plt.xticks(rotation=45)
    

    # Zones de couleurs pour le GAP (Overfitting check)
    for i in range(len(steps) - 1):
        g = gap[i]
        c = "#acffaa" if g <= 0.05 else ("#ffff00" if g <= 0.12 else "red")
        ax.axvspan(steps[i], steps[i + 1], facecolor=c, alpha=0.1)

    # Calcul de la couverture du dataset
    prc = []
    prc_train = []
    prc_ds1, prc_ds2 = [], []
    ema_smooth, ema_smooth_raw = [], []
    ema_smooth_2 = []
    ema_smooth_3 = []
    for i in range(len(steps)):
#        token_vu = steps[i] * 32_768
#        dataset = 1_526_627_840
#        prob = 1 - math.exp(-token_vu / dataset)
#        prc.append(prob)
#        prc_train.append(token_vu / dataset)
#        prc_ds1.append(steps[i]*(1-MIXED)*GRAD_ACCUM*BATCH_SIZE/TARGET_BLOCK_DS1)
#        prc_ds2.append(steps[i]*MIXED*GRAD_ACCUM*BATCH_SIZE/TARGET_BLOCK_DS2)
        # EMA lissée
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
    print(stats)
    # --- Préparation du texte ---
    info_text = (
        f"PREDICTION (Target: {horizon/1000:.0f}k)\n"
        f"---------------------------\n"
        f"Loss Horizon : {stats['loss_horizon']}\n"
        f"Slope/1k (now) : {stats['current_slope_per_1000']:.4f}\n"
        f"Slope/1k ({horizon/1000:.0f}k) : {stats['future_slope_per_1000']:.4f}\n"
        f'Plateau : {status_plateau}'
    )

    # --- TRACÉ DES COURBES ---
    if smooth > 1:
        s_smooth = steps[smooth - 1 :]
        l_smooth = moving_average(val_loss, smooth)
        ax.plot(s_smooth, moving_average(train_loss, smooth), label=f"Train (Smooth {smooth})", color="#3498db", lw=1.5)
        ax.plot(s_smooth, l_smooth, label=f"Val (Smooth {smooth})", color="#e67e22", lw=1.5)
        ax.text(steps[-1]+x_sens/10, l_smooth[-1], s=f"{l_smooth[-1]:.3f}", color="#e67e22", rotation=0, fontsize=9, horizontalalignment="left")
        # Tracé des courbes de comparaison si disponibles - second
        if val_loss_second is not None:
            s_sm_sec = steps_second[smooth - 1 :]
            ax.plot( s_sm_sec,moving_average(val_loss_second, smooth), label="Model 2 (Comparison)",color="#fb59b6",lw=0.9, linestyle="--")
            ax.plot(steps_second, ema_smooth_2, label="EMA lissée second", color="#cf35b8", linestyle="-.", lw=1.)

        # Tracé des courbes de comparaison si disponibles - third
        if val_loss_third is not None:
            s_sm_sec = steps_third[smooth - 1 :]
            ax.plot( s_sm_sec, moving_average(val_loss_third, smooth), label="Model 1 (Comparison)",color="#5bf9b6",lw=0.9,linestyle="--")
            ax.plot(steps_third, ema_smooth_3, label="EMA lissée third", color="#70eef3", linestyle="-.", lw=1.)

        ax.plot(steps, train_loss, color="#3498db", alpha=0.4, lw=1)
        ax.plot(steps, val_loss, color="#fa8118", alpha=0.4, lw=1)
        ax.plot(steps, ema_smooth, label="EMA lissée", color="#ff0000", linestyle="-.", lw=1.5)
        ax.plot(steps_proj, ema_proj, label="EMA Projectionf", color="#e67e22", linestyle=":", lw=1.1)
        ax.text(steps[-1]+x_sens/10, ema_smooth[-1], s=f"{ema_smooth[-1]:.3f}", color="red", rotation=0, fontsize=9, horizontalalignment="left")
        if raw:
            ax.plot(monitor_step_ds[20:], moving_average(monitor_loss,20)*1., label="train: raw data", color="#000000", linestyle=":", lw=0.5)
            ax.text(monitor_step_ds[-1]+x_sens/10, ema_smooth_raw[-1], s=f"{ema_smooth_raw[-1]:.3f}", color="black", rotation=0, fontsize=9, horizontalalignment="left")
    else:
        ax.plot(steps, train_loss, label="Train Loss", color="#3498db", lw=1.5)
        ax.plot(steps, val_loss, label="Val Loss", color="#fa831b", lw=2)
        ax.plot(steps, ema_smooth, label="EMA lissée", color="#ff0000", linestyle="-.", lw=1.5)
        ax.text(steps[-1]+x_sens, ema_smooth[-1], s=f"{ema_smooth[-1]:.3f}", color="red", rotation=0, fontsize=9, horizontalalignment="left")
        if raw:
            ax.plot(monitor_step_ds[6:], moving_average(monitor_loss,6)*1, label="train: raw data", color="#000000", linestyle=":", lw=0.5)
            ax.text(monitor_step_ds[-1]+x_sens/10, ema_smooth_raw[-1], s=f"{ema_smooth_raw[-1]:.3f}", color="black", rotation=0, fontsize=9, horizontalalignment="left")

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
                ax.text( ev["step"] - x_sens, y_min + 0.02, f"[{ev['step']:5d}] - " + ev["label"] + ETA, rotation=90, color=ev.get("color", "black"), fontsize=9, verticalalignment="bottom")

    if compare:
        ax.axvline( x=compare, color="black", linestyle="--", alpha=0.7, lw=0.8)
        ax.text( compare+x_sens/10, y_max, "Compare " , rotation=90, color="black", fontsize=9, verticalalignment="top")
        label = f"[{epoch_ds1}] Epoch 1 - Dataset 1 (Wiki)"
        if epoch_ds1<x_max:
            ax.axvline( x=epoch_ds1, color="magenta", linestyle="--", alpha=0.7, lw=0.8)
            ax.text( epoch_ds1+x_sens/10, y_min + 0.02, label , rotation=90, color="magenta", fontsize=9, verticalalignment="bottom")
        label = f"[{epoch_ds2}] Epoch 1 - Dataset 2 (CulturaX)"
        if epoch_ds2<x_max :
            ax.axvline( x=epoch_ds2, color="magenta", linestyle="--", alpha=0.7, lw=0.8)
            ax.text( epoch_ds2+x_sens/10, y_min + 0.02, label , rotation=90, color="magenta", fontsize=9, verticalalignment="bottom")
        label = f"[{epoch_ds3}] Epoch 1 - Dataset 3 (Littéraire)"
        if epoch_ds3<x_max :
            ax.axvline( x=epoch_ds3, color="magenta", linestyle="--", alpha=0.7, lw=0.8)
            ax.text( epoch_ds3+x_sens/10, y_min + 0.02, label , rotation=90, color="magenta", fontsize=9, verticalalignment="bottom")

    # --- RÉGLAGES FINAUX ---
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.set_title(
        f"{titre} | Step: {steps[-1]} | Gap: {gap[-1]:.4f}", fontsize=14
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Après avoir tracé la loss sur ax
    ax_droite = ax.twinx()
    ax_droite.grid(False)
    ax_droite.set_ylim(0.0, 1.0)
    #ax_droite.plot(steps, prc, color="#7BFFB0", alpha=0.7, label="%", lw=0.7)
    #ax_droite.plot(steps, prc_train, color="#f6881b", alpha=0.7, label="%", lw=0.7)
    ax_droite.plot(monitor_step_ds, monitor_ds1, color="#7B0080", alpha=0.7, label="%", lw=0.8)
    ax_droite.text( monitor_step_ds[-1]+x_sens/10,monitor_ds1[-1] , "wikipédia" , color="black", fontsize=9, horizontalalignment="left")
    ax_droite.plot(monitor_step_ds, monitor_ds2, color="#7B0080", alpha=0.7, label="%", lw=0.8)
    ax_droite.text( monitor_step_ds[-1]+x_sens/10,monitor_ds2[-1] , "culturaX" , color="black", fontsize=9, horizontalalignment="left")
    ax_droite.plot(monitor_step_ds, monitor_ds3, color="#7B0080", alpha=0.7, label="%", lw=0.8)
    ax_droite.text( monitor_step_ds[-1]+x_sens/10,monitor_ds3[-1] , "littéraire" , color="black", fontsize=9, horizontalalignment="left")
    ax_droite.plot(monitor_step_ds, monitor_grad, color="#7B8c80", alpha=0.7, label="%", lw=0.6)
#    ax_droite.plot(monitor_step_ds, monitor_mixed, color="#7B3F80", alpha=0.7, label="%", lw=0.8)
#    ax_droite.plot(steps, prc_ds1, color="#7BFFB0", alpha=0.7, label="%", lw=0.7)
#    ax_droite.plot(steps, prc_ds2, color="#7BFFB0", alpha=0.7, label="%", lw=0.7)
    # ax_droite.set_ylabel("Token vus", color="magenta")

    # --- Affichage de l'encart ---
    plt.text(0.5, 0.80, info_text, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='bottom', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))    

    
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

