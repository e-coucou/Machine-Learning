import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

NB = 100

def plot_training_performance(json_path):
    if not os.path.exists(json_path):
        print(f"❌ Fichier {json_path} introuvable.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    steps = data["steps"]

    # Conversion en tableaux Numpy pour les calculs d'empilement
    t_init = np.array(data["time_init"])
    t_batch = np.array(data["time_batch"])
    t_model = np.array(data["time_model"])
    t_backw = np.array(data["time_bacwd"])
    t_optim = np.array(data["time_optim"])
    s = len(t_init)
    t_eval = np.array(data["time_eval"])[:s]

    print(len(t_init))
    print(len(t_batch))
    print(len(t_model))
    print(len(t_backw))
    print(len(t_optim))
    print(len(t_eval))
    # Style Dark pour le Mac Pro
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [
        "Data (CPU)",
        "Forward (GPU)",
        "Backward (GPU)",
        "Optim (RAM)",
        "Init",
        "Eval Loss",
    ]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f1c40f", "#ad00f0", "#dfafaf"]

    # Création du graphique en aires empilées
    ax.stackplot(
        steps[-NB:],
        t_batch[-NB:],
        t_model[-NB:],
        t_backw[-NB:],
        t_optim[-NB:],
        t_init[-NB:],
        t_eval[-NB:],
        labels=labels,
        colors=colors,
        alpha=0.8,
    )

    ax.set_title(
        f"Profil de Performance Poussin-v2 (Moyenne par step)", fontsize=14, pad=20
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Temps par Step (secondes)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)

    # Ajustement dynamique de la hauteur
    total_time = t_batch + t_model + t_backw + t_optim + t_init + t_eval
    ax.set_ylim(0, max(total_time) * 1.2)

    plt.tight_layout()
    output_file = "model/screenshot/performance_clean.png"
    plt.savefig(output_file)
    print(f"✅ Profil de performance sauvegardé : {output_file}")

    # Ouverture automatique sur macOS
#    subprocess.run(["open", output_file])
    plt.show()


def plot_training_performance_eval(json_path):
    if not os.path.exists(json_path):
        print(f"❌ Fichier {json_path} introuvable.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    steps = data["steps"]

    # Conversion en tableaux Numpy pour les calculs d'empilement
    t_init = np.array(data["time_init"])
    s = len(t_init)
    t_eval = np.array(data["time_eval"])[:s]

    # Style Dark pour le Mac Pro
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [
        "Init",
        "Eval Loss",
    ]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f1c40f", "#ad00f0", "#dfafaf"]

    # Création du graphique en aires empilées
    ax.stackplot(
        steps[-50:],
        t_init[-50:],
        t_eval[-50:],
        labels=labels,
        colors=colors,
        alpha=0.8,
    )

    ax.set_title(
        f"Profil de Performance Poussin-v2 (Moyenne par step)", fontsize=14, pad=20
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Temps par Step (secondes)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)

    # Ajustement dynamique de la hauteur
    total_time = t_init + t_eval
    ax.set_ylim(0, max(total_time) * 0.2)

    plt.tight_layout()
    output_file = "model/screenshot/performance_clean_eval.png"
    plt.savefig(output_file)
    print(f"✅ Profil de performance sauvegardé : {output_file}")

    # Ouverture automatique sur macOS
#    subprocess.run(["open", output_file])
    plt.show()


def plot_training_swap(json_path):
    if not os.path.exists(json_path):
        print(f"❌ Fichier {json_path} introuvable.")
        return
    with open(json_path, 'r') as f:
        # On lit tout d'un coup, c'est plus rapide que de boucler sur le fichier
        lines = f.readlines()
        
    # Utilisation d'une "list comprehension" : c'est 2x plus rapide qu'une boucle .append()
    data = [json.loads(line) for line in lines if line.strip()]

    steps, t_swap = [], []
    for d in data:
        steps.append(d["step"])
        t_swap.append(d["swap"])

    # Style Dark pour le Mac Pro
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 6))

    labels = [
        "Swap",
    ]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f1c40f", "#ad00f0", "#dfafaf"]

    # Création du graphique en aires empilées
    ax.stackplot(
        steps[-NB:],
        t_swap[-NB:],
        labels=labels,
        colors=colors,
        alpha=0.8,
    )

    ax.set_title(
        f"Profil de Performance Poussin-v2 (Moyenne par step)", fontsize=14, pad=20
    )
    ax.set_xlabel("Steps")
    ax.set_ylabel("Mémoire en Go")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.2)

    # Ajustement dynamique de la hauteur
#    ax.set_ylim(0, max(t_swap))
    ax.set_ylim(0, 2.1)

    plt.tight_layout()
    output_file = "model/screenshot/performance_swap.png"
    plt.savefig(output_file)
    print(f"✅ Profil de performance sauvegardé : {output_file}")

    # Ouverture automatique sur macOS
#    subprocess.run(["open", output_file])
    plt.show()

if __name__ == "__main__":
#    plot_training_performance("model/my_wiky_history.json")
    plot_training_swap("model/monitor.log")
#    plot_training_performance_eval("model/my_wiky_history.json")
