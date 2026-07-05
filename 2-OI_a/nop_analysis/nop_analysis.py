"""
Fonctions métier pour l'analyse des opérations NOP.

Modèle de données
------------------
Le compteur brut `value_nop` reste constant pendant qu'une opération est en
cours, puis change de valeur (incrément) au passage à l'opération suivante.
Une "opération" est donc définie comme un plateau de valeur constante, et sa
durée est le temps écoulé entre le début de ce plateau et le début du
plateau suivant.

Toutes les fonctions de ce module sont vectorisées (numpy/pandas), sans
boucle Python sur les lignes, et valident leurs entrées (index temporel,
colonnes attendues) avant de calculer quoi que ce soit.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from .utils import ensure_columns, ensure_datetime_index, rolling_sum, zscore


def _to_month_period(timestamps: pd.Series) -> pd.Series:
    """Convertit une série de timestamps (tz-aware ou non) en période mensuelle sans warning."""
    ts = pd.to_datetime(timestamps)
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(None)
    return ts.dt.to_period("M")



# ---------------------------------------------------------------------------
# 1. Nettoyage
# ---------------------------------------------------------------------------
def clean_dataframe(df: pd.DataFrame, value_col: str = config.VALUE_COLUMN) -> pd.DataFrame:
    """
    Nettoie le DataFrame brut avant extraction des opérations.

    Étapes :
        - suppression des NaN
        - suppression des valeurs négatives (capteur invalide)
        - arrondi et conversion en int64 (le compteur est par nature entier)
        - tri chronologique

    Raises
    ------
    TypeError
        Si l'index n'est pas un DatetimeIndex.
    KeyError
        Si `value_col` est absent du DataFrame.
    """
    ensure_datetime_index(df)
    ensure_columns(df, [value_col])

    out = df[[value_col]].dropna(subset=[value_col])
    out = out[out[value_col] >= 0]
    out = out.assign(**{value_col: out[value_col].round().astype("int64")})
    return out.sort_index()


# ---------------------------------------------------------------------------
# 2. Extraction des opérations
# ---------------------------------------------------------------------------
def extract_operations(df: pd.DataFrame, value_col: str = config.VALUE_COLUMN) -> pd.DataFrame:
    """
    Reconstruit la liste des opérations à partir d'un compteur par plateaux.

    Parameters
    ----------
    df : DataFrame indexé par un DatetimeIndex, contenant `value_col`.
    value_col : nom de la colonne compteur.

    Returns
    -------
    DataFrame indexé 0..n-1 avec les colonnes :
        operation_id, value, start, end, duration, duration_min, n_samples
    """
    df = clean_dataframe(df, value_col)

    change = df[value_col].ne(df[value_col].shift())
    group_id = change.cumsum()

    ops = df.loc[change, [value_col]].rename(columns={value_col: "value"})
    ops["start"] = ops.index
    ops = ops.reset_index(drop=True)
    ops["operation_id"] = ops.index

    ops["end"] = ops["start"].shift(-1)
    ops.loc[ops.index[-1], "end"] = df.index[-1]

    ops["duration"] = ops["end"] - ops["start"]
    ops["duration_min"] = ops["duration"].dt.total_seconds() / 60

    ops["n_samples"] = df.groupby(group_id).size().to_numpy()

    return ops[
        ["operation_id", "value", "start", "end", "duration", "duration_min", "n_samples"]
    ]


def filter_operations_by_duration(
    ops: pd.DataFrame,
    min_duration: pd.Timedelta = config.MIN_OPERATION_DURATION,
    max_duration: pd.Timedelta = config.MAX_OPERATION_DURATION,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Écarte les opérations dont la durée sort de [min_duration, max_duration]
    (bruit capteur ou arrêt prolongé de la ligne).
    """
    ensure_columns(ops, ["duration"])
    mask = (ops["duration"] >= min_duration) & (ops["duration"] <= max_duration)

    if verbose:
        print(
            f"Opérations conservées : {mask.sum():,} / {len(ops):,} "
            f"(rejetées : {(~mask).sum():,} — hors [{min_duration}, {max_duration}])"
        )

    return ops.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Analyse des meilleures séries (runs)
# ---------------------------------------------------------------------------
def analyze_best_runs(
    ops: pd.DataFrame,
    window_sizes: Sequence[int] = config.DEFAULT_WINDOW_SIZES,
) -> Dict[int, dict]:
    """
    Pour chaque taille de fenêtre demandée, trouve la séquence de N opérations
    consécutives la plus rapide (durée totale minimale) et calcule ses
    statistiques descriptives ainsi que la capacité journalière équivalente.

    Returns
    -------
    dict {window_size: {..., "durations_sequence": [...], "capacity_per_day": float}}
    """
    ensure_columns(ops, ["start", "end", "value", "duration_min"])
    ops = ops.sort_values("start").reset_index(drop=True)

    durations = ops["duration_min"].to_numpy()
    values = ops["value"].to_numpy()
    starts = ops["start"].to_numpy()
    ends = ops["end"].to_numpy()

    results: Dict[int, dict] = {}
    for w in window_sizes:
        totals = rolling_sum(durations, w)
        if totals.size == 0:
            continue

        best_idx = int(np.argmin(totals))
        window_durations = durations[best_idx : best_idx + w]
        total_duration_min = float(window_durations.sum())
        mean_duration_min = float(window_durations.mean())

        results[w] = {
            "window": w,
            "start_index": best_idx,
            "end_index": best_idx + w - 1,
            "start_time": starts[best_idx],
            "end_time": ends[best_idx + w - 1],
            "values_sequence": values[best_idx : best_idx + w].tolist(),
            "durations_sequence": window_durations.tolist(),
            "total_duration_min": total_duration_min,
            "mean_duration_min": mean_duration_min,
            "median_duration_min": float(np.median(window_durations)),
            "std_duration_min": float(window_durations.std()),
            "min_duration_min": float(window_durations.min()),
            "max_duration_min": float(window_durations.max()),
            "cv": float(window_durations.std() / mean_duration_min) if mean_duration_min else np.nan,
            "capacity_per_day": (w / total_duration_min * 1440) if total_duration_min > 0 else np.nan,
        }

    return results


def get_reference_duration(ops: pd.DataFrame, window: int = config.GOLDEN_RUN_WINDOW) -> float:
    """Durée totale (minutes) de la meilleure séquence de `window` opérations — sert de référence de scoring."""
    ops = ops.sort_values("start").reset_index(drop=True)
    totals = rolling_sum(ops["duration_min"].to_numpy(), window)
    if totals.size == 0:
        raise ValueError(f"Pas assez d'opérations ({len(ops)}) pour une fenêtre de {window}.")
    return float(totals.min())


# ---------------------------------------------------------------------------
# 4. Fenêtres glissantes (remplace build_windows / build_sliding_windows)
# ---------------------------------------------------------------------------
def build_sliding_windows(ops: pd.DataFrame, window: int = config.GOLDEN_RUN_WINDOW) -> pd.DataFrame:
    """
    Construit, de façon vectorisée, toutes les fenêtres glissantes de
    `window` opérations consécutives avec leur horodatage de début/fin et
    leur durée totale.

    Remplace les boucles Python `for i in range(len(ops)-window+1): ...`
    du notebook original par un simple slicing + convolution.
    """
    ensure_columns(ops, ["start", "end", "duration_min"])
    ops = ops.sort_values("start").reset_index(drop=True)

    n = len(ops)
    if n < window:
        raise ValueError(f"Pas assez d'opérations ({n}) pour une fenêtre de {window}.")

    total_duration = rolling_sum(ops["duration_min"].to_numpy(), window)
    n_windows = n - window + 1

    return pd.DataFrame(
        {
            "start_time": ops["start"].to_numpy()[:n_windows],
            "end_time": ops["end"].to_numpy()[window - 1 :],
            "duration": total_duration,
        }
    )


def add_score(windows: pd.DataFrame, ref_duration: float) -> pd.DataFrame:
    """Ajoute un score de performance = ref_duration / duration (1 = performance de référence)."""
    ensure_columns(windows, ["start_time", "duration"])
    out = windows.copy()
    out["score"] = ref_duration / out["duration"]
    out["month"] = _to_month_period(out["start_time"])
    return out


def compute_monthly_scores(
    ops: pd.DataFrame,
    window: int = config.MONTHLY_ROLLING_WINDOW,
    ref_duration: Optional[float] = None,
) -> pd.DataFrame:
    """
    Score mensuel de performance des séries de `window` opérations
    consécutives : pour chaque fenêtre glissante, score = ref_duration / durée
    totale. La référence est la meilleure séquence de `window` opérations sur
    toute la période si `ref_duration` n'est pas fournie.
    """
    if ref_duration is None:
        ref_duration = get_reference_duration(ops, window)

    windows = build_sliding_windows(ops, window)
    scored = add_score(windows, ref_duration)

    return (
        scored.groupby("month")
        .agg(
            mean_score=("score", "mean"),
            median_score=("score", "median"),
            best_score=("score", "max"),
            worst_score=("score", "min"),
            n_sequences=("score", "count"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# 5. Z-score
# ---------------------------------------------------------------------------
def compute_zscore(ops: pd.DataFrame, window: int = config.GOLDEN_RUN_WINDOW) -> pd.DataFrame:
    """
    Z-score de performance par fenêtre glissante de `window` opérations.
    Convention : z > 0 = plus rapide que la moyenne, z < 0 = plus lent
    (le signe est inversé par rapport au z-score brut de la durée, puisqu'une
    durée courte = bonne performance).
    """
    windows = build_sliding_windows(ops, window)
    windows["z_score"] = -zscore(windows["duration"].to_numpy())
    windows["month"] = _to_month_period(windows["start_time"])
    return windows


def compute_monthly_zscore(zscored_windows: pd.DataFrame) -> pd.DataFrame:
    """Agrégation mensuelle d'un DataFrame produit par compute_zscore()."""
    ensure_columns(zscored_windows, ["month", "z_score"])
    return (
        zscored_windows.groupby("month")
        .agg(
            mean_z=("z_score", "mean"),
            median_z=("z_score", "median"),
            std_z=("z_score", "std"),
            best_z=("z_score", "max"),
            worst_z=("z_score", "min"),
            n_runs=("z_score", "count"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# 6. Golden run
# ---------------------------------------------------------------------------
def golden_run(ops: pd.DataFrame, window: int = config.GOLDEN_RUN_WINDOW) -> dict:
    """
    Calcule le "golden run" pour une taille de fenêtre donnée (35 opérations
    par défaut) : la meilleure séquence de `window` opérations consécutives
    observée sur toute la période.

    Volontairement découplé de `analyze_best_runs` sur plusieurs fenêtres :
    le golden run représente *une* référence à taille fixe, indépendante des
    tailles de fenêtre utilisées ailleurs (ex: heatmap comparative du
    dashboard, cf. `golden_run_dashboard`).

    Returns
    -------
    dict avec :
        window : taille de fenêtre utilisée
        result : résultat détaillé pour cette fenêtre (cf. analyze_best_runs)
    """
    results = analyze_best_runs(ops, window_sizes=(window,))
    if window not in results:
        raise ValueError(
            f"Pas assez d'opérations ({len(ops)}) pour calculer un golden run sur {window} opérations."
        )
    return {"window": window, "result": results[window]}


def golden_run_dashboard(
    ops: pd.DataFrame,
    golden: dict,
    comparison_window_sizes: Sequence[int] = config.DEFAULT_WINDOW_SIZES,
):
    """
    Construit le tableau de bord visuel du golden run (séquence, stats,
    heatmap comparative). Toute la mise en page graphique est déléguée à
    plots.py — cette fonction ne fait qu'assembler les données à afficher.

    `comparison_window_sizes` est indépendant de la fenêtre utilisée pour
    calculer `golden` : il ne sert qu'à peupler la heatmap comparative. La
    fenêtre du golden run est automatiquement incluse dans la comparaison
    même si elle n'apparaît pas dans `comparison_window_sizes`.
    """
    from . import plots  # import local pour éviter une dépendance circulaire

    window_sizes = tuple(sorted(set(comparison_window_sizes) | {golden["window"]}))
    all_results = analyze_best_runs(ops, window_sizes=window_sizes)

    return plots.plot_golden_run_dashboard(ops, golden, all_results)


# ---------------------------------------------------------------------------
# 7. Synthèse (pour la section "Conclusions" du notebook)
# ---------------------------------------------------------------------------
def _linear_trend(values: np.ndarray) -> tuple[float, str]:
    """Pente d'une régression linéaire simple + libellé qualitatif (amélioration/dégradation/stabilité)."""
    if len(values) < 2 or np.all(np.isnan(values)):
        return float("nan"), "non déterminable (< 2 mois de données)"

    x = np.arange(len(values))
    mask = ~np.isnan(values)
    slope = float(np.polyfit(x[mask], values[mask], 1)[0])

    if abs(slope) < 1e-9:
        label = "stabilité"
    else:
        label = "amélioration" if slope > 0 else "dégradation"
    return slope, label


def evaluate_golden_run(ops: pd.DataFrame, golden: dict) -> dict:
    """
    Avis "professionnel" sur la fiabilité du golden run comme référence OEE.

    Vérifie :
        - le nombre de jours couverts par le run (utile pour juger si une
          fenêtre de 35 correspond bien à ~7 jours sur vos données, ou pas)
        - le coefficient de variation du run (run homogène vs hétérogène)
        - si les opérations extrêmes du run (min/max) sont cohérentes avec
          la distribution globale des durées, ou si le run doit sa
          performance à 1-2 opérations très atypiques

    Ne fait aucune recommandation automatique de suppression de données :
    signale seulement les points à vérifier manuellement.
    """
    result = golden["result"]
    days_covered = (result["end_time"] - result["start_time"]).total_seconds() / 86400

    cv = result["cv"]
    if cv < 0.15:
        cv_verdict = "run homogène (CV < 0.15) : moyenne et médiane doivent être proches, référence fiable."
    elif cv < 0.30:
        cv_verdict = "run modérément hétérogène (0.15 ≤ CV < 0.30) : comparer moyenne et médiane avant de trancher."
    else:
        cv_verdict = "run très hétérogène (CV ≥ 0.30) : probablement porté par 1-2 opérations exceptionnelles, à examiner avant de l'ériger en référence."

    global_mean, global_std = ops["duration_min"].mean(), ops["duration_min"].std()
    run_durations = np.array(result["durations_sequence"])
    n_low_outliers = int((run_durations < global_mean - 2 * global_std).sum())
    n_high_outliers = int((run_durations > global_mean + 2 * global_std).sum())

    return {
        "days_covered": days_covered,
        "close_to_one_week": abs(days_covered - 7) <= 1,
        "cv": cv,
        "cv_verdict": cv_verdict,
        "mean_vs_median_gap_pct": abs(result["mean_duration_min"] - result["median_duration_min"]) / result["mean_duration_min"] * 100,
        "n_atypical_ops_in_run": n_low_outliers + n_high_outliers,
    }


def build_summary_report(
    ops: pd.DataFrame,
    results: Dict[int, dict],
    golden: dict,
    monthly_scores: pd.DataFrame,
    monthly_zscore: pd.DataFrame,
) -> str:
    """
    Construit un résumé markdown de l'analyse à partir des résultats déjà
    calculés (durées, meilleures séries, golden run, tendance mensuelle,
    avis sur la fiabilité du golden run comme référence OEE).

    Conçu pour être affiché tel quel dans la section "Conclusions" du
    notebook via `IPython.display.Markdown`.
    """
    ensure_columns(ops, ["duration_min"])
    d = ops["duration_min"]

    lines = ["### Résultats\n"]
    lines.append(f"- Nombre d'opérations analysées : **{len(ops):,}**")
    lines.append(f"- Durée moyenne : **{d.mean():.1f} min** ({d.mean() / 60:.2f} h)")
    lines.append(f"- Durée médiane : **{d.median():.1f} min**")
    lines.append(f"- Durée min / max : **{d.min():.1f} min** / **{d.max():.1f} min**")
    lines.append(f"- Écart-type : **{d.std():.1f} min**")
    lines.append("")

    lines.append("**Meilleures séries consécutives :**\n")
    for w in sorted(results.keys()):
        r = results[w]
        lines.append(
            f"- Fenêtre de {w} opérations : {r['total_duration_min']:.1f} min au total "
            f"({r['start_time']:%Y-%m-%d %H:%M} → {r['end_time']:%Y-%m-%d %H:%M}) "
            f"→ capacité équivalente **{r['capacity_per_day']:.2f} ops/jour**"
        )
    lines.append("")

    gr = golden["result"]
    evaluation = evaluate_golden_run(ops, golden)
    lines.append(
        f"**Golden run** (fenêtre de référence : {golden['window']} opérations) : "
        f"capacité équivalente **{gr['capacity_per_day']:.2f} ops/jour**, "
        f"sur **{evaluation['days_covered']:.1f} jours**, "
        f"coefficient de variation **{gr['cv']:.2f}**."
    )
    lines.append("")

    lines.append("### Maille de référence pour l'OEE\n")
    lines.append(
        f"- Pour le **débit/capacité**, c'est la **moyenne** qui doit servir de référence "
        f"(capacité = 1440 / durée moyenne) — la médiane ne reconstitue pas le temps total écoulé."
    )
    gap_pct = evaluation["mean_vs_median_gap_pct"]
    if gap_pct < 10:
        gap_comment = "proches, run homogène."
    else:
        gap_comment = "écart notable : vérifier que la moyenne n'est pas tirée par quelques opérations atypiques."
    lines.append(f"- Écart moyenne/médiane sur ce run : **{gap_pct:.1f}%** — {gap_comment}")
    if evaluation["close_to_one_week"]:
        lines.append(
            f"- La fenêtre de {golden['window']} opérations couvre **{evaluation['days_covered']:.1f} jours**, "
            f"proche d'une semaine : maille hebdomadaire cohérente pour un suivi OEE."
        )
    else:
        lines.append(
            f"- La fenêtre de {golden['window']} opérations couvre **{evaluation['days_covered']:.1f} jours** "
            f"sur ce run — **pas exactement 7 jours** ici : si une maille hebdomadaire est visée, ajuster la "
            f"taille de fenêtre (`config.GOLDEN_RUN_WINDOW`) plutôt que de supposer que 35 = 7 jours."
        )
    lines.append(f"- {evaluation['cv_verdict']}")
    if evaluation["n_atypical_ops_in_run"] > 0:
        lines.append(
            f"- ⚠️ **{evaluation['n_atypical_ops_in_run']} opération(s) atypique(s)** "
            f"(hors moyenne ± 2σ globale) à l'intérieur de ce run — à inspecter avant de le retenir comme référence "
            f"(cf. `short_operations()` / `duration_threshold_report()`)."
        )
    else:
        lines.append("- Aucune opération atypique détectée à l'intérieur du run par rapport à la distribution globale.")
    lines.append(
        "- Recommandation générale : le **minimum absolu** est un extrême par construction (référence "
        "structurellement optimiste). Pour une référence OEE soutenable, préférer le **10ᵉ percentile** des "
        "fenêtres glissantes de même taille (`build_sliding_windows` + `.quantile(0.1)`) plutôt que le record."
    )
    lines.append("")

    slope_score, trend_score = _linear_trend(monthly_scores["mean_score"].to_numpy())
    slope_z, trend_z = _linear_trend(monthly_zscore["mean_z"].to_numpy())

    lines.append("### Tendance\n")
    lines.append(
        f"**Tendance mensuelle (z-score, indicateur principal)** : {trend_z}"
        + (f" (pente = {slope_z:+.4f} / mois)" if not np.isnan(slope_z) else "")
    )
    lines.append(
        f"**Écart au record (score, indicateur secondaire)** : {trend_score}"
        + (f" (pente = {slope_score:+.4f} / mois)" if not np.isnan(slope_score) else "")
    )
    lines.append("")

    if len(monthly_scores) > 0:
        best_month = monthly_scores.loc[monthly_scores["mean_score"].idxmax(), "month"]
        worst_month = monthly_scores.loc[monthly_scores["mean_score"].idxmin(), "month"]
        lines.append(f"- Période couverte : **{monthly_scores['month'].iloc[0]}** → **{monthly_scores['month'].iloc[-1]}**")
        lines.append(f"- Meilleur mois (score moyen) : **{best_month}** (score = {monthly_scores['mean_score'].max():.3f})")
        lines.append(f"- Mois le plus faible (score moyen) : **{worst_month}** (score = {monthly_scores['mean_score'].min():.3f})")
    lines.append("")

    lines.append("### Interprétation\n")
    if trend_z == "amélioration":
        lines.append("Le z-score indique une **amélioration** de la performance dans le temps.")
    elif trend_z == "dégradation":
        lines.append("Le z-score indique une **dégradation** de la performance dans le temps — à surveiller.")
    else:
        lines.append("Le z-score ne montre pas de tendance nette — performance stable ou recul insuffisant pour conclure.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 8. Zoom sur les opérations courtes (aide au choix de MIN_OPERATION_DURATION)
# ---------------------------------------------------------------------------
def duration_threshold_report(
    ops: pd.DataFrame,
    thresholds_hours: Sequence[float] = (1, 2, 3, 4, 5, 6),
) -> pd.DataFrame:
    """
    Pour chaque seuil testé (en heures), indique combien d'opérations
    seraient exclues si MIN_OPERATION_DURATION = seuil. Aide à choisir le
    seuil sans le fixer à l'aveugle.
    """
    ensure_columns(ops, ["duration"])
    rows = []
    for h in thresholds_hours:
        excluded = ops["duration"] < pd.Timedelta(hours=h)
        rows.append({
            "threshold_h": h,
            "n_excluded": int(excluded.sum()),
            "pct_excluded": round(excluded.mean() * 100, 1),
            "n_remaining": int((~excluded).sum()),
        })
    return pd.DataFrame(rows)


def short_operations(ops: pd.DataFrame, threshold_hours: float = 5) -> pd.DataFrame:
    """Liste les opérations sous `threshold_hours`, triées par durée croissante (pour inspection manuelle)."""
    ensure_columns(ops, ["duration", "duration_min"])
    mask = ops["duration"] < pd.Timedelta(hours=threshold_hours)
    return ops.loc[mask].sort_values("duration_min").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 9. CV mensuel (stabilité, à lire à côté du z-score)
# ---------------------------------------------------------------------------
def compute_monthly_cv(ops: pd.DataFrame) -> pd.DataFrame:
    """
    Coefficient de variation mensuel des durées d'opération.

    À lire à côté de `compute_monthly_zscore` : le z-score suit le *niveau*
    (la moyenne), le CV suit la *dispersion*. Un CV qui augmente peut être un
    signal précoce de dégradation, même quand le z-score de niveau est
    encore stable.
    """
    ensure_columns(ops, ["start", "duration_min"])
    tmp = ops[["start", "duration_min"]].copy()
    tmp["month"] = _to_month_period(tmp["start"])

    grouped = tmp.groupby("month")["duration_min"]
    monthly = grouped.agg(mean_duration_min="mean", std_duration_min="std", n_operations="count")
    monthly["cv"] = monthly["std_duration_min"] / monthly["mean_duration_min"]

    return monthly.reset_index()


# ---------------------------------------------------------------------------
# 10. Carte de contrôle SPC (individus / étendue mobile - I-MR)
# ---------------------------------------------------------------------------
# Constantes standard pour une carte I-MR (étendue mobile calculée sur 2 points).
_D2_N2 = 1.128   # convertit l'étendue mobile moyenne en écart-type estimé
_D4_N2 = 3.267   # limite de contrôle haute de la carte des étendues mobiles


def compute_control_limits(ops: pd.DataFrame, value_col: str = "duration_min") -> dict:
    """
    Limites de contrôle SPC (carte individus, méthode I-MR).

    L'écart-type est estimé à partir de l'étendue mobile (écart entre
    valeurs consécutives) plutôt que d'un écart-type brut : cette méthode
    est la référence en contrôle statistique de procédé (Shewhart) et est
    moins sensible qu'un écart-type classique à une dérive lente ou à
    quelques valeurs extrêmes isolées.
    """
    ensure_columns(ops, [value_col])
    x = ops[value_col].to_numpy()
    if len(x) < 2:
        raise ValueError("Au moins 2 opérations sont nécessaires pour calculer une étendue mobile.")

    center = float(x.mean())
    moving_range = np.abs(np.diff(x))
    mr_bar = float(moving_range.mean())
    sigma_hat = mr_bar / _D2_N2

    return {
        "center": center,
        "sigma_hat": sigma_hat,
        "ucl": center + 3 * sigma_hat,
        "lcl": max(0.0, center - 3 * sigma_hat),
        "mr_bar": mr_bar,
        "mr_ucl": _D4_N2 * mr_bar,
    }


def flag_out_of_control(ops: pd.DataFrame, limits: dict, value_col: str = "duration_min") -> pd.DataFrame:
    """
    Ajoute une colonne booléenne `out_of_control` : True si la durée sort
    des limites de contrôle ±3σ (règle SPC n°1, la plus basique et la plus
    fiable — signale une variation "cause spéciale" à investiguer, par
    opposition au bruit normal du procédé).
    """
    ensure_columns(ops, [value_col])
    out = ops.copy()
    out["out_of_control"] = (out[value_col] > limits["ucl"]) | (out[value_col] < limits["lcl"])
    return out
