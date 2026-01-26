import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np


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
    t_eval = np.array(data["time_eval"])

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
        steps,
        t_batch,
        t_model,
        t_backw,
        t_optim,
        t_init,
        t_eval,
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
    subprocess.run(["open", output_file])


if __name__ == "__main__":
    plot_training_performance("model/my_wiky_history.json")
