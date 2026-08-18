"""
Visualisations pour l'exploration des repères Att (fréquence, transitions).

Mêmes conventions que `nop_analysis.plots` : aucune logique métier ici,
chaque fonction prend en entrée des DataFrames déjà calculés par
`att_analysis.py`, et accepte un paramètre optionnel `ax` (dashboard
composite) ; sans `ax`, la fonction crée sa propre figure et l'affiche.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns

from nop_analysis.plots import ACCENT, GOOD, PRIMARY

from . import config


def _finalize(ax: plt.Axes, created_fig: bool) -> None:
    if created_fig:
        plt.tight_layout()
        plt.show()


def plot_repere_frequency(
    freq_table,
    top_n: int = config.TOP_N_REPERES_DISPLAY,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Pareto du temps total par repère : barres triées par durée totale
    décroissante (les `top_n` premiers) + courbe cumulée en %, pour repérer
    d'un coup d'œil les quelques repères qui concentrent l'essentiel du
    temps de cycle.
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(13, 5))

    data = freq_table.head(top_n)
    labels = data["repere"].astype(str)

    ax.bar(labels, data["total_duration_min"], color=PRIMARY)
    ax.set_xlabel("Repère")
    ax.set_ylabel("Durée totale (min)")
    ax.set_title(f"Pareto du temps total par repère (top {len(data)} / {len(freq_table)})")
    ax.tick_params(axis="x", rotation=90)

    ax2 = ax.twinx()
    ax2.plot(labels, data["cum_pct_of_total_time"], color=ACCENT, marker="o", ms=3, lw=1.5)
    ax2.set_ylabel("% cumulé du temps total", color=ACCENT)
    ax2.set_ylim(0, 105)
    ax2.axhline(80, color=ACCENT, linestyle="--", lw=1, alpha=0.5)
    ax2.grid(False)

    _finalize(ax, created_fig)


def plot_transition_heatmap(
    transitions,
    top_n: int = config.TOP_N_REPERES_DISPLAY,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Heatmap de la matrice de transition (probabilité repère(t) -> repère(t+1)),
    restreinte aux `top_n` repères qui totalisent le plus de transitions
    (une matrice sur les ~3300 valeurs possibles serait illisible et
    quasi-vide). Les repères affichés sont triés par code croissant, ce qui
    fait apparaître la diagonale de la séquence type si l'ordre des codes
    suit globalement l'ordre chronologique des pas.
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(11, 9))

    top_reperes = sorted(
        transitions.groupby("from_repere")["n"].sum().sort_values(ascending=False).head(top_n).index
    )
    subset = transitions[
        transitions["from_repere"].isin(top_reperes) & transitions["to_repere"].isin(top_reperes)
    ]
    matrix = subset.pivot(index="from_repere", columns="to_repere", values="probability").reindex(
        index=top_reperes, columns=top_reperes
    )

    sns.heatmap(matrix, cmap="Blues", ax=ax, cbar_kws={"label": "P(suivant | repère)"}, linewidths=0.3)
    ax.set_xlabel("Repère suivant")
    ax.set_ylabel("Repère")
    ax.set_title(f"Matrice de transition (top {len(top_reperes)} repères par volume de transitions)")

    _finalize(ax, created_fig)


def plot_defaut_pareto(defaut_freq, pas_reference, ax: Optional[plt.Axes] = None) -> None:
    """
    Barres du nombre de défauts détectés par pas (triées décroissant), avec
    le libellé du pas plutôt que son seul numéro — pour repérer d'un coup
    d'œil les pas qui génèrent le plus de retries (cf.
    `att_analysis.defaut_frequency_table`).
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(11, 5))

    merged = defaut_freq.merge(pas_reference[["pas_num", "code_court"]], on="pas_num", how="left")
    labels = merged["pas_num"].astype(str) + " " + merged["code_court"].fillna("?")

    ax.bar(labels, merged["n_defauts"], color=ACCENT)
    ax.set_xlabel("Pas")
    ax.set_ylabel("Nombre de défauts détectés")
    ax.set_title("Défauts par pas (régressions vers un pas antérieur)")
    ax.tick_params(axis="x", rotation=45)

    _finalize(ax, created_fig)


def plot_attente_lancement_histogram(
    cycle_times_df,
    max_minutes: int = 60,
    ax: Optional[plt.Axes] = None,
) -> None:
    """
    Histogramme de l'attente au lancement (`attente_lancement_min`, cf.
    `att_analysis.cycle_times`), borné à `[0, max_minutes]`.

    Cette distribution a une queue extrêmement lourde (arrêts prolongés :
    week-ends, absence de commande...) qui, avec un zoom ±Nσ classique
    (`nop_analysis.plots.plot_duration_histogram`), reste dominée par cette
    queue et écrase la lecture du chaînage normal, qui se joue en pratique
    sur quelques dizaines de minutes. On fixe donc une borne absolue en
    minutes plutôt qu'une borne statistique — rien n'est supprimé des
    données, seules les valeurs au-delà de `max_minutes` sortent du champ
    (comptées et annoncées dans le titre).
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(11, 5))

    d = cycle_times_df["attente_lancement_min"]
    within = d[(d >= 0) & (d <= max_minutes)]
    n_excluded = len(d) - len(within)

    sns.histplot(within, bins=max_minutes, color=PRIMARY, ax=ax)
    ax.axvline(d.median(), color=GOOD, linestyle="--", label=f"Médiane globale ({d.median():.1f} min)")
    ax.axvline(d.min(), color=ACCENT, linestyle="--", label=f"Minimum ({d.min():.1f} min)")
    ax.set_xlim(0, max_minutes)
    ax.set_xlabel("Attente au lancement (min)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Attente au lancement, 0-{max_minutes} min "
        f"({n_excluded}/{len(d)} valeur(s) au-delà, hors champ)"
    )
    ax.legend(fontsize=9)

    _finalize(ax, created_fig)


def plot_standard_vs_rework(operations, rolling_window: int = 30, ax: Optional[plt.Axes] = None) -> None:
    """
    Part du temps de rework (défauts) dans la durée totale de chaque
    opération, dans le temps (cf. `att_analysis.reconstruct_operations` pour
    `standard_duration_min`/`rework_duration_min`).

    Un histogramme empilé avec une barre par opération devient illisible
    (barres invisibles) dès quelques centaines d'opérations, et l'échelle en
    minutes brutes est écrasée par les quelques opérations à très forte
    durée. On trace donc un **pourcentage** (toujours dans [0, 100], insensible
    aux valeurs extrêmes en minutes) sous forme de nuage de points + moyenne
    mobile — même logique que `nop_analysis.plots.plot_rolling_capacity_over_time` —
    ce qui reste lisible à n'importe quel volume d'opérations et montre en
    prime la tendance dans le temps (le rework augmente-t-il, diminue-t-il ?).
    """
    created_fig = ax is None
    if created_fig:
        _, ax = plt.subplots(figsize=(14, 5))

    ops = operations.sort_values("start").reset_index(drop=True)
    pct_rework = ops["rework_duration_min"] / ops["duration_min"] * 100

    ax.scatter(ops["start"], pct_rework, color=ACCENT, s=10, alpha=0.4, label="% rework par opération")
    ax.plot(
        ops["start"],
        pct_rework.rolling(rolling_window, min_periods=1, center=True).mean(),
        color=PRIMARY, lw=2, label=f"Moyenne mobile ({rolling_window} opérations)",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("% de la durée en rework")
    ax.set_title("Part du rework (défauts) dans la durée des opérations, dans le temps")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)

    _finalize(ax, created_fig)
