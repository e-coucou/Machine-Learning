import argparse
import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch


# --- 1. INSPECTION TECHNIQUE ---
def run_inspection(inf_path, train_path):
    print("\n--- 🧐 INSPECTION DES CHECKPOINTS ---")
    if os.path.exists(inf_path):
        sd = torch.load(inf_path, map_location="cpu", weights_only=True)
        model = sd["model"] if "model" in sd else sd
        n_params = sum(t.numel() for t in model.values() if isinstance(t, torch.Tensor))
        print(f"✅ Taille du modèle : {n_params / 1e6:.2f}M paramètres")

    if os.path.exists(train_path):
        ckpt = torch.load(train_path, map_location="cpu", weights_only=True)
        print(f"✅ Clés d'entraînement : {list(ckpt.keys())}")


# --- 2. PROFIL DE PERFORMANCE ---
def plot_perf(json_path):
    print("--- ⚡ GÉNÉRATION PROFIL PERFORMANCE ---")
    with open(json_path, "r") as f:
        data = json.load(f)

    steps = data["steps"]
    metrics = [
        "time_batch",
        "time_model",
        "time_bacwd",
        "time_optim",
        "time_init",
        "time_eval",
    ]
    labels = [
        "Data (CPU)",
        "Forward (GPU)",
        "Backward (GPU)",
        "Optim (RAM)",
        "Init",
        "Eval",
    ]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f1c40f", "#ad00f0", "#dfafaf"]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 5))

    # Empilement des données temporelles
    stacks = [np.array(data[m]) for m in metrics]
    ax.stackplot(steps, *stacks, labels=labels, colors=colors, alpha=0.8)

    ax.set_title("Répartition de la charge (Moyenne par Step)")
    ax.set_ylabel("Secondes")
    ax.legend(loc="upper left")

    out = "model/screenshot/master_perf.png"
    plt.savefig(out)
    return out


# --- 3. ANALYSE LOSS & GAP ---
def plot_loss(json_path, compare_step, ymin, ymax):
    print("--- 📉 GÉNÉRATION COURBE DE LOSS ---")
    with open(json_path, "r") as f:
        data = json.load(f)

    steps = np.array(data["steps"])
    train_l = np.array(data["train_loss"])
    val_l = np.array(data["val_loss"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, train_l, label="Train", color="#3498db", alpha=0.3)
    ax.plot(steps, val_l, label="Val", color="#e67e22", lw=2)

    # Ajout du repère de comparaison
    ax.axvline(
        x=compare_step, color="red", linestyle="--", label=f"Ref: {compare_step}"
    )
    ax.set_ylim(ymin, ymax)
    ax.legend()
    ax.set_title(f"Suivi Loss | Dernier Step: {steps[-1]}")

    out = "model/screenshot/master_loss.png"
    plt.savefig(out)
    return out


# --- EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", type=int, default=12900)
    parser.add_argument("--ymin", type=float, default=2.6)
    parser.add_argument("--ymax", type=float, default=3.5)
    args = parser.parse_args()

    JSON_FILE = "model/my_wiky_history.json"
    INF_FILE = "model/my_wiki_inference.pth"
    TRAIN_FILE = "model/my_wiki.pth"

    run_inspection(INF_FILE, TRAIN_FILE)
    img_perf = plot_perf(JSON_FILE)
    img_loss = plot_loss(JSON_FILE, args.compare, args.ymin, args.ymax)
    img_train = "model/screenshot/analyse_poussin_results.png"

    # Ouverture groupée sur macOS
    subprocess.run(["open", img_perf, img_loss, img_train])
    print("\n🚀 Dashboard mis à jour. Images ouvertes dans Aperçu.")
