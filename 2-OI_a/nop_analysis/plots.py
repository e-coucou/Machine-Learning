"""
Toutes les visualisations de l'analyse NOP.

Aucune logique métier ici : chaque fonction prend en entrée des DataFrames
ou des dicts déjà calculés par nop_analysis.py, et se contente de les
afficher.

Convention : chaque fonction accepte un paramètre optionnel `ax`. Si `ax`
n'est pas fourni, la fonction crée sa propre figure et l'affiche. Si `ax`
est fourni (cas du dashboard composite), la fonction dessine dessus sans
créer de figure ni appeler `show()`.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Style global (appliqué une fois à l'import du module)
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.0)
PRIMARY = "#2C6E9E"      # bleu : mesures / valeurs normales
ACCENT = "#D9534F"       # rouge : points d'attention / seuils
GOOD = "#3E9B4F"         # vert : performance positive
NEUTRAL_GREY = "#8C8C8C"

plt.rcParams.update({
    "axes.edgecolor": "#4A4A4A",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "figure.facecolor": "white",
})


def _finalize(ax: plt.Axes, created_fig: bool) -> None:
    if created_fig:
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Distribution des durées
# ---------------------------------------------------------------------------
def plot_duration_histogram(ops: pd.DataFrame, clip_std: float = 2.5) -> None:
    """
    Histogramme des durées avec panneau de statistiques.

    L'axe X est recentré sur mean ± clip_std*std pour ne pas écraser la
    masse principale à cause de quelques valeurs extrêmes (rien n'est
    supprimé des données, seul l'affichage est zoomé).
    """
    d = ops["duration_min"]
    mean, median, std = d.mean(), d.median(), d.std()
    lo, hi = max(0, mean - clip_std * std), mean + clip_std * std
    n_clipped = int(((d < lo) | (d > hi)).sum())

    fig = plt.figure(figsize=(13, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 2, 1])
    ax_hist, ax_zoom, ax_stats = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    sns.histplot(d, bins=60, color=PRIMARY, ax=ax_hist)
    ax_hist.set_title("Vue complète")
    ax_hist.set_xlabel("Durée (min)")

    zoom_data = d[(d >= lo) & (d <= hi)]
    sns.histplot(zoom_data, bins=40, color=PRIMARY, kde=True, ax=ax_zoom)
    ax_zoom.axvline(mean, color=ACCENT, linestyle="--", label=f"Moyenne ({mean:.0f} min)")
    ax_zoom.axvline(median, color=GOOD, linestyle="--", label=f"Médiane ({median:.0f} min)")
    ax_zoom.set_title(f"Zoom ±{clip_std:g}σ ({n_clipped} valeurs hors champ)")
    ax_zoom.set_xlabel("Durée (min)")
    ax_zoom.legend(fontsize=9)

    ax_stats.axis("off")
    stats_text = (
        f"n opérations : {len(d):,}\n\n"
        f"Moyenne  : {mean:.1f} min\n"
        f"Médiane  : {median:.1f} min\n"
        f"Écart-type : {std:.1f} min\n"
        f"CV       : {std / mean:.2f}\n\n"
        f"Min      : {d.min():.1f} min\n"
        f"Max      : {d.max():.1f} min\n\n"
        f"P10 / P90 : {d.quantile(.1):.0f} / {d.quantile(.9):.0f} min"
    )
    ax_stats.text(0.0, 0.5, stats_text, fontsize=11, va="center", family="monospace")

    fig.suptitle("Distribution des durées d'opération", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_duration_by_weekday(ops: pd.DataFrame, ax: Optional[plt.Axes] = None) -> None:
    """
    Durée par jour de semaine.

    Remplace l'ancien boxplot par valeur (non pertinent : chaque valeur
    d'opération n'apparaît qu'une fois, un boxplot à 1 point ne montre
    aucune dispersion). Ici chaque jour regroupe plusieurs opérations :
    utile pour repérer un jour systématiquement plus lent (équipe,
    maintenance, approvisionnement...).
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(10, 5))

    data = ops.copy()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    data["weekday"] = pd.Categorical(data["start"].dt.day_name(), categories=order, ordered=True)

    sns.boxplot(data=data, x="weekday", y="duration_min", color=PRIMARY, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Durée (min)")
    ax.set_title("Durée par jour de la semaine")
    ax.tick_params(axis="x", rotation=30)
    _finalize(ax, created_fig)


# ---------------------------------------------------------------------------
# Meilleures séries : capacité selon la taille de fenêtre
# ---------------------------------------------------------------------------
def plot_capacity_vs_window(results: Dict[int, dict], ops: pd.DataFrame, ax: Optional[plt.Axes] = None) -> None:
    """
    Capacité journalière équivalente en fonction de la taille de fenêtre.

    Remplace la heatmap comparative : montre directement l'effet de
    "régression vers la moyenne" (plus la fenêtre est longue, plus la
    capacité de pointe se rapproche de la capacité moyenne réelle), ce qui
    aide à choisir une taille de fenêtre représentative d'une capacité
    *soutenue* plutôt que d'un sprint ponctuel.
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(10, 5))

    windows = sorted(results.keys())
    capacities = [results[w]["capacity_per_day"] for w in windows]
    cvs = [results[w]["cv"] for w in windows]

    sustained_capacity = 1440 / ops["duration_min"].mean()

    ax.plot(windows, capacities, "-o", color=PRIMARY, lw=2, ms=8, label="Capacité du meilleur run")
    ax.axhline(sustained_capacity, color=NEUTRAL_GREY, linestyle="--",
               label=f"Capacité moyenne soutenue ({sustained_capacity:.2f} ops/j)")

    for w, cap, cv in zip(windows, capacities, cvs):
        ax.annotate(f"CV={cv:.2f}", (w, cap), textcoords="offset points", xytext=(0, 8),
                    fontsize=8, ha="center", color=NEUTRAL_GREY)

    ax.set_xlabel("Taille de fenêtre (nb opérations consécutives)")
    ax.set_ylabel("Capacité équivalente (ops/jour)")
    ax.set_title("Capacité de pointe vs taille de fenêtre")
    ax.legend()
    _finalize(ax, created_fig)


def plot_rolling_capacity_over_time(ops: pd.DataFrame, window: int, ax: Optional[plt.Axes] = None) -> None:
    """
    Capacité glissante dans le temps pour une taille de fenêtre donnée.

    Remplace la heatmap globale : contrairement à elle, cette vue a un
    véritable axe temporel exploitable pour repérer une tendance.
    """
    from .nop_analysis import build_sliding_windows

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(13, 5))

    w = build_sliding_windows(ops, window)
    capacity = window / w["duration"] * 1440

    ax.plot(w["start_time"], capacity, color=PRIMARY, lw=1, alpha=0.5, label="Capacité par fenêtre")
    ax.plot(w["start_time"], capacity.rolling(30, min_periods=1, center=True).mean(),
            color=ACCENT, lw=2, label="Moyenne mobile (30 fenêtres)")

    ax.set_xlabel("Date")
    ax.set_ylabel("Capacité équivalente (ops/jour)")
    ax.set_title(f"Capacité glissante dans le temps (fenêtre = {window} opérations)")
    ax.legend()
    _finalize(ax, created_fig)


# ---------------------------------------------------------------------------
# Évolution mensuelle
# ---------------------------------------------------------------------------
def plot_monthly_zscore(monthly_z: pd.DataFrame, window: int, ax: Optional[plt.Axes] = None) -> None:
    """
    Z-score mensuel en barres colorées (vert = au-dessus de la moyenne
    historique, rouge = en-dessous). Vue principale pour juger de la
    tendance : contrairement au score (comparé au record absolu), le
    z-score compare chaque mois à la distribution de tous les mois, ce qui
    rend une amélioration ou une dégradation progressive plus lisible.
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(12, 5))

    months = monthly_z["month"].astype(str)
    colors = [GOOD if v >= 0 else ACCENT for v in monthly_z["mean_z"]]

    ax.bar(months, monthly_z["mean_z"], color=colors)
    ax.axhline(0, color="black", lw=1)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Performance (z-score)")
    ax.set_title(f"Tendance mensuelle — runs de {window} opérations")
    _finalize(ax, created_fig)


def plot_monthly_score(monthly: pd.DataFrame, ax: Optional[plt.Axes] = None) -> None:
    """
    Score mensuel ("écart au record absolu") en barres, vue secondaire.
    À interpréter avec prudence : la référence est un extrême (la
    meilleure séquence jamais observée), donc ce graphe répond à "à quel
    point suis-je loin du record ?", pas à "est-ce que je progresse ?"
    (préférer plot_monthly_zscore pour la tendance).
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(12, 4))

    months = monthly["month"].astype(str)
    ax.bar(months, monthly["mean_score"], color=PRIMARY, alpha=0.85)
    ax.axhline(1, color=ACCENT, linestyle="--", label="Record absolu")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Score (1 = record)")
    ax.set_title("Écart mensuel au record absolu")
    ax.legend()
    _finalize(ax, created_fig)


def plot_monthly_cv_vs_zscore(monthly_zscore: pd.DataFrame, monthly_cv: pd.DataFrame, window: int) -> None:
    """
    Z-score (niveau) et CV (dispersion) mensuels, empilés sur un axe temps
    partagé. Un CV qui grimpe pendant que le z-score reste stable est un
    signal précoce à surveiller : la moyenne tient encore, mais le procédé
    devient moins régulier.
    """
    fig, (ax_z, ax_cv) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    months_z = monthly_zscore["month"].astype(str)
    colors = [GOOD if v >= 0 else ACCENT for v in monthly_zscore["mean_z"]]
    ax_z.bar(months_z, monthly_zscore["mean_z"], color=colors)
    ax_z.axhline(0, color="black", lw=1)
    ax_z.set_ylabel("Z-score (niveau)")
    ax_z.set_title(f"Niveau vs dispersion — runs de {window} opérations")

    months_cv = monthly_cv["month"].astype(str)
    ax_cv.plot(months_cv, monthly_cv["cv"], "-o", color=PRIMARY, lw=2)
    ax_cv.set_ylabel("CV (dispersion)")
    ax_cv.tick_params(axis="x", rotation=45)
    ax_cv.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Carte de contrôle SPC (I-MR)
# ---------------------------------------------------------------------------
def plot_control_chart(ops: pd.DataFrame, limits: dict, value_col: str = "duration_min",
                        ax: Optional[plt.Axes] = None) -> None:
    """
    Carte de contrôle individus (I-MR) : durée de chaque opération dans le
    temps, avec ligne centrale et limites de contrôle ±3σ (σ estimé par
    étendue mobile). Les points hors limites (variation "cause spéciale")
    sont surlignés en rouge — c'est un moyen objectif de fixer le seuil
    haut du mask plutôt qu'à l'œil sur l'histogramme.
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(15, 5))

    from .nop_analysis import flag_out_of_control
    flagged = flag_out_of_control(ops, limits, value_col=value_col)

    ax.plot(flagged["start"], flagged[value_col], color=PRIMARY, lw=1, alpha=0.7, zorder=1)
    ax.scatter(flagged["start"], flagged[value_col], color=PRIMARY, s=12, zorder=2)

    ooc = flagged[flagged["out_of_control"]]
    ax.scatter(ooc["start"], ooc[value_col], color=ACCENT, s=50, zorder=3, label=f"Hors contrôle ({len(ooc)})")

    ax.axhline(limits["center"], color="black", lw=1.2, label=f"Moyenne ({limits['center']:.0f} min)")
    ax.axhline(limits["ucl"], color=ACCENT, linestyle="--", label=f"UCL ({limits['ucl']:.0f} min)")
    ax.axhline(limits["lcl"], color=ACCENT, linestyle="--", label=f"LCL ({limits['lcl']:.0f} min)")

    ax.set_xlabel("Date")
    ax.set_ylabel("Durée (min)")
    ax.set_title("Carte de contrôle SPC (I-MR) des durées d'opération")
    ax.legend(fontsize=9, loc="upper right")
    _finalize(ax, created_fig)


def plot_monthly_sequence_count(monthly: pd.DataFrame, ax: Optional[plt.Axes] = None) -> None:
    """Nombre de séquences analysées par mois (contrôle de robustesse statistique)."""
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(12, 3))

    ax.bar(monthly["month"].astype(str), monthly["n_sequences"], color=NEUTRAL_GREY)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Nombre de séquences analysées par mois")
    ax.set_ylabel("n")
    _finalize(ax, created_fig)


# ---------------------------------------------------------------------------
# Best run : vue d'ensemble et détail
# ---------------------------------------------------------------------------
def plot_best_run_overlay(ops: pd.DataFrame, result: dict, ax: Optional[plt.Axes] = None) -> None:
    """Timeline complète des durées avec la meilleure séquence surlignée."""
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(16, 4.5))

    ax.plot(ops["start"], ops["duration_min"], lw=1, color=NEUTRAL_GREY, alpha=0.6, label="Toutes les opérations")
    ax.axvspan(result["start_time"], result["end_time"], color=ACCENT, alpha=0.15, label="Meilleur run")

    mask = (ops["start"] >= result["start_time"]) & (ops["end"] <= result["end_time"])
    ax.plot(ops.loc[mask, "start"], ops.loc[mask, "duration_min"], color=ACCENT, lw=2, zorder=3)
    ax.scatter(ops.loc[mask, "start"], ops.loc[mask, "duration_min"], color=ACCENT, s=35, zorder=4)

    ax.set_xlabel("Date")
    ax.set_ylabel("Durée (min)")
    ax.set_title(f"Meilleure séquence de {result['window']} opérations sur la période")
    ax.legend(loc="upper right")
    _finalize(ax, created_fig)


def _best_run_slice(ops: pd.DataFrame, result: dict) -> pd.DataFrame:
    return ops[(ops["start"] >= result["start_time"]) & (ops["end"] <= result["end_time"])].copy()


def plot_best_sequence_stats(ops: pd.DataFrame, result: dict, ax: Optional[plt.Axes] = None) -> None:
    """Détail de la meilleure séquence avec bande moyenne ± écart-type."""
    seq = _best_run_slice(ops, result)
    mean, std = seq["duration_min"].mean(), seq["duration_min"].std()

    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.plot(range(len(seq)), seq["duration_min"], "-o", color=ACCENT, lw=2, label="Durée")
    ax.axhline(mean, color=PRIMARY, label=f"Moyenne ({mean:.0f} min)")
    ax.fill_between(range(len(seq)), mean - std, mean + std, alpha=0.15, color=PRIMARY, label="±1σ")
    ax.set_xlabel("Position dans la séquence")
    ax.set_ylabel("Durée (min)")
    ax.set_title(f"Stabilité du run ({result['window']} opérations)")
    ax.legend(fontsize=9)
    _finalize(ax, created_fig)


# ---------------------------------------------------------------------------
# Dashboard composite (golden run)
# ---------------------------------------------------------------------------
def plot_golden_run_dashboard(ops: pd.DataFrame, golden: dict, all_results: Dict[int, dict]) -> plt.Figure:
    """
    Tableau de bord synthétique du golden run :
        - timeline complète en pleine largeur (le plus important : où et
          quand se situe le run, dans le contexte de toute la période)
        - stabilité du run (durée, ±1σ) et capacité vs taille de fenêtre
        - synthèse chiffrée
    """
    result = golden["result"]

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])

    ax_overlay = fig.add_subplot(gs[0, :])
    plot_best_run_overlay(ops, result, ax=ax_overlay)

    ax_stats = fig.add_subplot(gs[1, 0])
    plot_best_sequence_stats(ops, result, ax=ax_stats)

    ax_capacity = fig.add_subplot(gs[1, 1])
    plot_capacity_vs_window(all_results, ops, ax=ax_capacity)

    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis("off")
    days_covered = (result["end_time"] - result["start_time"]).total_seconds() / 86400
    summary = (
        f"GOLDEN RUN — fenêtre : {golden['window']} opérations   |   "
        f"Période : {result['start_time']:%Y-%m-%d} → {result['end_time']:%Y-%m-%d} "
        f"({days_covered:.1f} jours)\n\n"
        f"Durée totale : {result['total_duration_min']:.0f} min   |   "
        f"Moyenne/op : {result['mean_duration_min']:.1f} min   |   "
        f"Médiane/op : {result['median_duration_min']:.1f} min   |   "
        f"Écart-type : {result['std_duration_min']:.1f} min   |   "
        f"CV : {result['cv']:.2f}\n\n"
        f"Capacité équivalente : {result['capacity_per_day']:.2f} ops/jour"
    )
    ax_summary.text(0.02, 0.5, summary, fontsize=12, va="center", family="monospace")

    fig.suptitle("Golden Run Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return fig
