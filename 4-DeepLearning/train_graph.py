import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


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
    for l in data_loss[:end]:
        ema_loss = 0.1 * l + (1 - 0.1) * ema_loss
    return ema_loss, end


def plot_poussin_gap(
    y_min=2.6,
    y_max=3.3,
    smooth=5,
    log_path="model/my_wiky_history.json",
    second="save/model_2/my_wiky_history.json",
    third=None,
    compare=12900,
    annot_event=None,
):
    # --- CHARGEMENT DATA 1 ---
    if not os.path.exists(log_path):
        print(f"❌ Fichier principal introuvable : {log_path}")
        return
    with open(log_path, "r") as f:
        data = json.load(f)

    steps = np.array(data["steps"])
    train_loss = np.array(data["train_loss"])
    val_loss = np.array(data["val_loss"])
    gap = val_loss - train_loss

    # --- CHARGEMENT DATA 2 (COMPARAISON) ---
    val_loss_second, steps_second = None, None
    ema_loss_second, end_second = 0, 0

    if second and os.path.exists(second):
        with open(second, "r") as f2:
            data_second = json.load(f2)
        val_loss_second = np.array(data_second["val_loss"])
        steps_second = np.array(data_second["steps"])
        ema_loss_second, end_second = calcul_ema(
            val_loss_second, steps_second, compare=compare
        )
    else:
        print(f"⚠️  Note : Second fichier de log non chargé ({second})")

    if third and os.path.exists(third):
        with open(third, "r") as f2:
            data_third = json.load(f2)
        val_loss_third = np.array(data_third["val_loss"])
        steps_third = np.array(data_third["steps"])
    else:
        print(f"⚠️  Note : Troisieme fichier de log non chargé ({third})")

    # --- LISSAGE ET EMA ---
    def moving_average(x, w):
        if w <= 1:
            return x
        return np.convolve(x, np.ones(w), "valid") / w

    ema_loss, end = calcul_ema(val_loss, steps, compare=None)

    # --- STYLE ET FIGURE ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))
    # Définit un repère principal tous les 2500
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2500))
    # Optionnel : Ajoute des petits traits (minuscules) tous les 500 pour plus de précision
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(500))
    # Rotation des labels pour éviter qu'ils ne se chevauchent s'il y en a trop
    plt.xticks(rotation=45)

    # Zones de couleurs pour le GAP (Overfitting check)
    for i in range(len(steps) - 1):
        g = gap[i]
        c = "#aaffbb" if g <= 0.05 else ("yellow" if g <= 0.12 else "red")
        ax.axvspan(steps[i], steps[i + 1], facecolor=c, alpha=0.1)

    # Calcul de la couverture du dataset
    prc = []
    prc_train = []
    for i in range(len(steps)):
        token_vu = steps[i] * 32_768
        dataset = 1_526_627_840
        prob = 1 - math.exp(-token_vu / dataset)
        prc.append(prob)
        prc_train.append(token_vu / dataset)

    # --- TRACÉ DES COURBES ---
    if smooth > 1:
        s_smooth = steps[smooth - 1 :]
        ax.plot(
            s_smooth,
            moving_average(train_loss, smooth),
            label=f"Train (Smooth {smooth})",
            color="#3498db",
            lw=2,
        )
        ax.plot(
            s_smooth,
            moving_average(val_loss, smooth),
            label=f"Val (Smooth {smooth})",
            color="#e67e22",
            lw=2,
        )

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

        ax.plot(steps, train_loss, color="#3498db", alpha=0.2, lw=1)
        ax.plot(steps, val_loss, color="#e67e22", alpha=0.2, lw=1)
    else:
        ax.plot(steps, train_loss, label="Train Loss", color="#3498db", lw=1.5)
        ax.plot(steps, val_loss, label="Val Loss", color="#e67e22", lw=2)

    # --- ANNOTATIONS (Lignes verticales et texte) ---
    if annot_event:
        for ev in annot_event:
            ax.axvline(
                x=ev["step"],
                color=ev.get("color", "black"),
                linestyle="--",
                alpha=0.6,
                lw=ev.get("lw", 1),
            )
            ax.text(
                ev["step"] + 100,
                y_min + 0.02,
                f"[{ev['step']:5d}] - " + ev["label"],
                rotation=90,
                color=ev.get("color", "black"),
                fontsize=9,
                verticalalignment="bottom",
            )

    if compare:
        ax.axvline(
            x=compare,
            color="black",
            linestyle="--",
            alpha=0.7,
            label="Point de comparaison",
            lw=0.8,
        )

    # --- RÉGLAGES FINAUX ---
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, 70000)
    ax.set_title(
        f"Analyse Mac-GPT | Step: {steps[-1]} | Gap: {gap[-1]:.4f}", fontsize=14
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Après avoir tracé la loss sur ax
    ax_droite = ax.twinx()
    ax_droite.set_ylim(0.0, 1.0)
    ax_droite.plot(steps, prc, color="#5bf9b6", alpha=0.5, label="%", lw=0.7)
    ax_droite.plot(steps, prc_train, color="#e67e22", alpha=0.5, label="%", lw=0.7)
    # ax_droite.set_ylabel("Token vus", color="magenta")

    plt.tight_layout()
    plt.savefig("model/screenshot/analyse_poussin_results.png")
    print(f"✅ Analyse terminée. Image : analyse_poussin_results.png")
    plt.show()

    # Affichage des stats dans le terminal (ton stdout)
    print("-" * 30)
    print(f"Dernier Step: {steps[-1]} | Gap actuel: {gap[-1]:.4f}")
    if val_loss_second is not None:
        print("COMPARE POINT ANALYSIS:")
        print(f"  - Val Loss: {val_loss[-1]:.4f} vs {val_loss_second[end_second]:.4f}")
        print(f"  - EMA Loss: {ema_loss:.3f} vs {ema_loss_second:.3f}")


# --- LANCEMENT ---
if __name__ == "__main__":
    # 1. Configuration des arguments
    parser = argparse.ArgumentParser(description="Analyseur de Loss Mac-GPT")

    parser.add_argument(
        "--compare", type=int, default=12900, help="Step de comparaison (ex: 12900)"
    )
    parser.add_argument(
        "--ymin", type=float, default=2.6, help="Limite basse de l'axe Y"
    )
    parser.add_argument(
        "--ymax", type=float, default=3.3, help="Limite haute de l'axe Y"
    )
    parser.add_argument(
        "--smooth", type=int, default=5, help="Fenêtre de lissage (smooth)"
    )

    args = parser.parse_args()

    # 2. Définition de tes événements (tu peux les laisser en dur ou les charger)
    mes_evenements = [
        {"step": 825, "label": "°°°", "color": "gray", "lw": 0.7},
        {"step": 3300, "label": "First run", "color": "purple"},
        {"step": 4425, "label": "°°°", "color": "gray", "lw": 0.7},
        {"step": 6525, "label": "Start Divergence", "color": "red"},
        {"step": 7002, "label": "°°°", "color": "gray", "lw": 0.7},
        {"step": 9300, "label": "°°°", "color": "gray", "lw": 0.7},
        {"step": 10275, "label": "°°°", "color": "gray", "lw": 0.7},
        {"step": 12325, "label": "End Divergence", "color": "red"},
        {"step": 15225, "label": "°°°", "color": "gray", "lw": 0.7},
        {"step": 15600, "label": "New data ...", "color": "blue"},
        {"step": 46589, "label": "epoch 2", "color": "red", "lw": 1.1},
        {"step": 39000, "label": "LR_Decay 90000", "color": "gray", "lw": 0.7},
        {
            "step": 32600,
            "label": "Dropout 0.1 / 70000 lr 3e-4",
            "color": "gray",
            "lw": 0.7,
        },
        {"step": 19500, "label": "Dropout 0.15", "color": "gray", "lw": 0.7},
        # {"step": 3300, label='LR_decay 50000')\n",
    ]

    # 3. Lancement avec les arguments saisis
    plot_poussin_gap(
        y_min=args.ymin,
        y_max=args.ymax,
        smooth=args.smooth,
        log_path="model/my_wiky_history.json",
        second="save/model_2/my_wiky_history.json",
        third="save/model_1/my_wiky_history.json",
        compare=args.compare,
        annot_event=mes_evenements,
    )
