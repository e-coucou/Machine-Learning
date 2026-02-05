import json
import math
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

MONITOR_FILE = 'model/monitor.log'
TARGET_BLOCK_DS1 = 5963390
TARGET_BLOCK_DS2 = 2429390
MIXED = 0.35
BATCH_SIZE = 8
GRAD_ACCUM = 16

def calcul_epoch(step, target, mix, batch, grad, t_step):
    return int((target - step * batch) / (batch * mix * grad) + t_step)
    
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


def plot_poussin_gap(
    y_min=2.6,
    y_max=3.3,
    x_max=70000,
    smooth=5,
    log_path="model/my_wiky_history.json",
    second="save/model_2/my_wiky_history.json",
    third=None,
    compare=None,
    annot_event=None,
    target=None,
    speed=None,
    raw=False,
    titre = "Training GPT Mac-M1"
):
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

    monitor_step, monitor_loss = [], []
    for d in data_monitor:
        monitor_step.append(d['step'])
        monitor_loss.append(d['loss'])
 
    if compare == -1:
        compare = data["steps"][-1]
        # echéances epoch
    step_ds1 = data_monitor[-1]['dataset1']
    epoch_ds1 = calcul_epoch(step=step_ds1, target=TARGET_BLOCK_DS1, mix=(1-MIXED), batch=BATCH_SIZE, grad=GRAD_ACCUM , t_step=compare)
    step_ds2 = data_monitor[-1]['dataset2']
    epoch_ds2 = calcul_epoch(step=step_ds2, target=TARGET_BLOCK_DS2, mix=MIXED, batch=BATCH_SIZE, grad=GRAD_ACCUM , t_step=compare)

    steps = np.array(data["steps"])
    train_loss = np.array(data["train_loss"])
    val_loss = np.array(data["val_loss"])
    gap = val_loss - train_loss

    # --- CHARGEMENT DATA 2 (COMPARAISON) ---

    if second and os.path.exists(second):
        with open(second, "r") as f2:
            data_second = json.load(f2)
        val_loss_second = np.array(data_second["val_loss"])
        steps_second = np.array(data_second["steps"])
        ema_loss_second, end_second = calcul_ema(val_loss_second, steps_second, compare=compare)
    else:
        val_loss_second, steps_second = None, None
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
    fig, ax = plt.subplots(figsize=(14, 7))
    # Définit un repère principal tous les 2500
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2500))
    # Optionnel : Ajoute des petits traits (minuscules) tous les 500 pour plus de précision
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(500))
    # Rotation des labels pour éviter qu'ils ne se chevauchent s'il y en a trop
    ax.set_yticks(np.arange(y_min, y_max, (y_max-y_min)/10))
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
    ema_smooth = []
    ema_smooth_2 = []
    ema_smooth_3 = []
    for i in range(len(steps)):
        token_vu = steps[i] * 32_768
        dataset = 1_526_627_840
        prob = 1 - math.exp(-token_vu / dataset)
        prc.append(prob)
        prc_train.append(token_vu / dataset)
        prc_ds1.append(steps[i]*(1-MIXED)*GRAD_ACCUM*BATCH_SIZE/TARGET_BLOCK_DS1)
        prc_ds2.append(steps[i]*MIXED*GRAD_ACCUM*BATCH_SIZE/TARGET_BLOCK_DS2)
        # EMA lissée
        ema_smooth.append(calcul_ema(val_loss[: i + 1], steps[: i + 1])[0])
        if val_loss_second is not None:
            ema_smooth_2.append(calcul_ema(val_loss_second[: i + 1], steps_second[: i + 1])[0])
        if val_loss_third is not None:
            ema_smooth_3.append(calcul_ema(val_loss_third[: i + 1], steps_third[: i + 1])[0])

    # --- TRACÉ DES COURBES ---
    if smooth > 1:
        s_smooth = steps[smooth - 1 :]
        ax.plot(
            s_smooth,
            moving_average(train_loss, smooth),
            label=f"Train (Smooth {smooth})",
            color="#3498db",
            lw=1.5,
        )
        ax.plot(
            s_smooth,
            moving_average(val_loss, smooth),
            label=f"Val (Smooth {smooth})",
            color="#e67e22",
            lw=1.5,
        )
        # Tracé des courbes de comparaison si disponibles - second
        if val_loss_second is not None:
            s_sm_sec = steps_second[smooth - 1 :]
            ax.plot(
                s_sm_sec,
                moving_average(val_loss_second, smooth),
                label="Model 2 (Comparison)",
                color="#fb59b6",
                lw=0.9,
                linestyle="--",
            )
            ax.plot(steps, ema_smooth_2, label="EMA lissée second", color="#cf35b8", linestyle="-.", lw=1.)

        # Tracé des courbes de comparaison si disponibles - third
        if val_loss_third is not None:
            s_sm_sec = steps_third[smooth - 1 :]
            ax.plot(
                s_sm_sec,
                moving_average(val_loss_third, smooth),
                label="Model 1 (Comparison)",
                color="#5bf9b6",
                lw=0.9,
                linestyle="--",
            )
            ax.plot(steps, ema_smooth_3, label="EMA lissée third", color="#70eef3", linestyle="-.", lw=1.)

        ax.plot(steps, train_loss, color="#3498db", alpha=0.2, lw=1)
        ax.plot(steps, val_loss, color="#fa8118", alpha=0.2, lw=1)
        ax.plot(steps, ema_smooth, label="EMA lissée", color="#ff0000", linestyle="-.", lw=1.5)
        if raw:
            ax.plot(monitor_step[20:], moving_average(monitor_loss,21)*1., label="train: raw data", color="#000000", linestyle=":", lw=0.5)
    else:
        ax.plot(steps, train_loss, label="Train Loss", color="#3498db", lw=1.5)
        ax.plot(steps, val_loss, label="Val Loss", color="#fa831b", lw=2)
        ax.plot(steps, ema_smooth, label="EMA lissée", color="#ff0000", linestyle="-.", lw=1.5)
        if raw:
            ax.plot(monitor_step[20:], moving_average(monitor_loss,21)*1, label="train: raw data", color="#000000", linestyle=":", lw=0.5)

    # --- TARGET --- (Ligne Horizontale) -------------
    if target is not None:
        ax.axhline( y=target, color="#aa55bb", alpha=0.7, linestyle="-.", lw=1.5)
#        ax.text( 1000, (target+0.01), s="Cible à " ,color="#aa55bb", rotation=0, fontsize=9, horizontalalignment="left")
        ax.text( 500, (target+0.01), s=("Loss Cible à "+f"{target:.2}") ,color="#aa55bb", rotation=0, fontsize=9, horizontalalignment="left")
    # --- ANNOTATIONS (Lignes verticales et texte) ---
    if annot_event:
        for ev in annot_event:
            if ((speed is not None) & (ev['step']>steps[-1])):
                elapse = ((ev['step']-steps[-1])*speed)
                eta = datetime.now() + timedelta(seconds=elapse)
                ETA = " ("+eta.strftime("%d %H:%M")+")"
            else:
                ETA = ""
            ax.axvline( x=ev["step"], color=ev.get("color", "black"), linestyle="--", alpha=0.6, lw=ev.get("lw", 1))
            ax.text( ev["step"] - 800, y_min + 0.02, f"[{ev['step']:5d}] - " + ev["label"] + ETA, rotation=90, color=ev.get("color", "black"), fontsize=9, verticalalignment="bottom")

    if compare:
        ax.axvline( x=compare, color="black", linestyle="--", alpha=0.7, lw=0.8)
        ax.text( compare+200, y_min + 0.02, "Point de comparaison des Loss/EMA" , rotation=90, color=ev.get("color", "black"), fontsize=9, verticalalignment="bottom")
        label = f"[{epoch_ds1}] Epoch 1 - Dataset 1 (Wiki)"
        if epoch_ds1<x_max:
            ax.axvline( x=epoch_ds1, color="magenta", linestyle="--", alpha=0.7, lw=0.8)
            ax.text( epoch_ds1-800, y_min + 0.02, label , rotation=90, color=ev.get("color", "magenta"), fontsize=9, verticalalignment="bottom")
        label = f"[{epoch_ds2}] Epoch 1 - Dataset 2 (CulturaX)"
        if epoch_ds2<x_max :
            ax.axvline( x=epoch_ds2, color="magenta", linestyle="--", alpha=0.7, lw=0.8)
            ax.text( epoch_ds2-800, y_min + 0.02, label , rotation=90, color=ev.get("color", "magenta"), fontsize=9, verticalalignment="bottom")

    # --- RÉGLAGES FINAUX ---
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, x_max)
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
    ax_droite.plot(steps, prc_ds1, color="#7BFFB0", alpha=0.7, label="%", lw=0.7)
    ax_droite.plot(steps, prc_ds2, color="#7BFFB0", alpha=0.7, label="%", lw=0.7)
    # ax_droite.set_ylabel("Token vus", color="magenta")

    plt.tight_layout()
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
