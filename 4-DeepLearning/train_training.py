import argparse
import glob
import os
import re
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


def plot_poussin_dashboard(log_folder="model/"):
    data_points = []
    # Récupération de tous les fichiers training*.log
    files = glob.glob(os.path.join(log_folder, "training*.log"))

    if not files:
        print(f"❌ Aucun fichier log trouvé dans {log_folder}")
        return

    # Regex pour extraire les données des logs
    pattern = re.compile(
        r"step (\d+): train loss ([\d\.]+), val loss ([\d\.]+), lr [\d\.e-]+ \(([\d\.]+)\s*s\)"
    )

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    data_points.append(
                        {
                            "step": int(match.group(1)),
                            "train": float(match.group(2)),
                            "val": float(match.group(3)),
                            "time_block": float(match.group(4)),
                        }
                    )

    if not data_points:
        print("❌ Aucune donnée extraite. Vérifie le format de tes logs.")
        return

    # 1. Préparation des données avec Pandas
    df = pd.DataFrame(data_points).drop_duplicates(subset="step").sort_values("step")

    # Calcul du temps par step réel (différentiel entre deux logs)
    df["step_diff"] = df["step"].diff()
    df["time_per_step"] = df["time_block"] / df["step_diff"]

    # Lissage des courbes
    df["val_smooth"] = df["val"].rolling(window=10, min_periods=1).mean()
    df["train_smooth"] = df["train"].rolling(window=10, min_periods=1).mean()

    # Nettoyage des valeurs aberrantes (pics de démarrage ou pauses système)
    df.loc[df["time_per_step"] > 15, "time_per_step"] = None

    # --- Configuration du Dashboard ---
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # --- GRAPHIQUE 1 : LOSS ---
    ax1.plot(df["step"], df["val"], color="red", alpha=0.1, linewidth=0.5)
    ax1.plot(
        df["step"],
        df["val_smooth"],
        color="#ff4757",
        linewidth=2.5,
        label="Val Loss (Lissée)",
    )
    ax1.plot(
        df["step"],
        df["train_smooth"],
        color="#bec2cc",
        linewidth=1.5,
        label="Train Loss (Lissée)",
        alpha=0.8,
    )

    # Ligne de record (Ton objectif 2.613)
    record_val = 2.613
    ax1.axhline(
        y=record_val,
        color="#eccc68",
        linestyle="--",
        linewidth=2,
        label=f"Record Nuit ({record_val})",
    )

    # Zoom intelligent
    current_min = df["val_smooth"].min()
    ax1.set_ylim(current_min * 0.96, current_min * 1.10)

    ax1.set_title(
        f"Poussin-v2 Dashboard | Dernier Step: {df['step'].iloc[-1]}",
        fontsize=16,
        color="white",
        pad=20,
    )
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
    ax1.legend(loc="upper right", frameon=True, facecolor="#2f3542")
    ax1.grid(True, alpha=0.1)

    # --- GRAPHIQUE 2 : PERFORMANCE MATÉRIELLE ---
    ax2.scatter(
        df["step"],
        df["time_per_step"],
        s=8,
        color="#7bed9f",
        alpha=0.3,
        label="Temps/Step brut",
    )
    ax2.plot(
        df["step"],
        df["time_per_step"].rolling(window=15).mean(),
        color="#2ed573",
        linewidth=2,
        label="Stabilité Mac Pro",
    )

    ax2.set_ylabel("Secondes / 1 Step", fontsize=12)
    ax2.set_xlabel("Total Training Steps", fontsize=12)

    # Formatage de l'axe X en milliers (k)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x / 1000)}k"))

    if not df["time_per_step"].dropna().empty:
        ax2.set_ylim(0, df["time_per_step"].quantile(0.98))

    ax2.legend(loc="upper right", frameon=True, facecolor="#2f3542")
    ax2.grid(True, alpha=0.1)

    # Stats finales
    last_val = df["val"].iloc[-1]
    ax1.annotate(
        f"Current Val Loss: {last_val:.4f}",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        color="white",
        weight="bold",
    )

    # --- Sauvegarde et Ouverture ---
    output_file = "model/screenshot/poussin_dashboard.png"
    plt.savefig(output_file, dpi=150)
    print(f"📊 Dashboard mis à jour : {output_file}")

    # Commande spécifique macOS pour ouvrir l'image sans bloquer le script
    subprocess.run(["open", output_file])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder", type=str, default="model/", help="Dossier contenant les .log"
    )
    args = parser.parse_args()

    plot_poussin_dashboard(args.folder)
