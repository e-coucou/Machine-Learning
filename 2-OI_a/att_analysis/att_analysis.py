"""
Fonctions métier pour l'exploration des repères Att.

Modèle de données
------------------
Le repère brut (colonne Att) reste constant pendant qu'un pas de process ou
d'attente est en cours, puis change de valeur au passage au pas suivant :
même modèle "plateau" que `nop_analysis.extract_operations`, réutilisé tel
quel ici (`extract_steps`). La différence avec le modèle NOP est que le
repère n'est pas un compteur qui n'augmente que dans un seul sens : il
parcourt une séquence de pas (un "pas" = un plateau) puis revient à un état
de repos (`config.IDLE_VALUE`) avant la prochaine opération. Une opération
complète est donc une suite ordonnée de plusieurs pas, pas un pas isolé —
la reconstruction des opérations à partir de cette suite est une étape
ultérieure (cf. notebook, section 5), qui s'appuie sur les repères de
début/fin identifiés ici.

Toutes les fonctions de ce module sont vectorisées (numpy/pandas).
"""

from __future__ import annotations

import json
import re
from typing import Tuple

import numpy as np
import pandas as pd

from nop_analysis import compute_control_limits as _nop_compute_control_limits
from nop_analysis import extract_operations
from nop_analysis import clean_dataframe as _nop_clean_dataframe
from nop_analysis import get_reference_duration as _nop_get_reference_duration
from nop_analysis.utils import ensure_columns

from . import config


# ---------------------------------------------------------------------------
# 0. Nettoyage — exclusion des repères non significatifs
# ---------------------------------------------------------------------------
def clean_dataframe(
    df: pd.DataFrame,
    value_col: str,
    invalid_values: Tuple = config.INVALID_REPERE_VALUES,
) -> pd.DataFrame:
    """
    Nettoyage générique (`nop_analysis.clean_dataframe`) puis exclusion des
    valeurs de repère non significatives (ex: 0, cf. `config.INVALID_REPERE_VALUES`
    et sa justification issue de la doc process).
    """
    out = _nop_clean_dataframe(df, value_col=value_col)
    return out[~out[value_col].isin(invalid_values)]


# ---------------------------------------------------------------------------
# 1. Extraction des pas (plateaux)
# ---------------------------------------------------------------------------
def extract_steps(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Reconstruit les pas (plateaux de repère constant) à partir du signal Att
    brut échantillonné à la minute.

    Réutilise `nop_analysis.extract_operations` (même modèle de plateau) et
    renomme simplement les colonnes au vocabulaire du domaine Att :
    `operation_id` -> `step_id`, `value` -> `repere`.

    Returns
    -------
    DataFrame avec les colonnes :
        step_id, repere, start, end, duration, duration_min, n_samples
    """
    steps = extract_operations(df, value_col=value_col)
    return steps.rename(columns={"operation_id": "step_id", "value": "repere"})


# ---------------------------------------------------------------------------
# 2. Fréquence par repère (Pareto)
# ---------------------------------------------------------------------------
def repere_frequency_table(steps: pd.DataFrame) -> pd.DataFrame:
    """
    Table de fréquence par repère : nombre d'occurrences, durée totale /
    moyenne / médiane / écart-type / min / max, et part cumulée du temps
    total (base d'un diagramme de Pareto).

    Triée par durée totale décroissante : les repères en tête sont ceux qui
    pèsent le plus dans le temps de cycle cumulé, indépendamment de leur
    code numérique ou de leur position dans la séquence.
    """
    ensure_columns(steps, ["repere", "duration_min"])
    grouped = steps.groupby("repere")["duration_min"]
    table = grouped.agg(
        n="count",
        total_duration_min="sum",
        mean_duration_min="mean",
        median_duration_min="median",
        std_duration_min="std",
        min_duration_min="min",
        max_duration_min="max",
    ).reset_index()

    table["cv"] = table["std_duration_min"] / table["mean_duration_min"]
    table = table.sort_values("total_duration_min", ascending=False).reset_index(drop=True)

    total = table["total_duration_min"].sum()
    table["pct_of_total_time"] = table["total_duration_min"] / total * 100
    table["cum_pct_of_total_time"] = table["pct_of_total_time"].cumsum()
    return table


# ---------------------------------------------------------------------------
# 3. Matrice de transition
# ---------------------------------------------------------------------------
def transition_counts(steps: pd.DataFrame) -> pd.DataFrame:
    """
    Matrice de transition repère(t) -> repère(t+1), sous forme de table
    (from_repere, to_repere, n, probability).

    `probability` est normalisée par `from_repere` (les lignes partageant le
    même `from_repere` totalisent 1.0) : elle répond à la question "sachant
    que je pars de ce repère, vers quel repère suivant est-ce que je vais le
    plus souvent ?". Objective la séquence réelle du procédé, sans supposer
    que l'ordre des codes numériques reflète l'ordre chronologique des pas.
    """
    ensure_columns(steps, ["repere", "start"])
    ordered = steps.sort_values("start").reset_index(drop=True)

    pairs = pd.DataFrame({
        "from_repere": ordered["repere"],
        "to_repere": ordered["repere"].shift(-1),
    }).dropna()
    pairs["to_repere"] = pairs["to_repere"].astype(ordered["repere"].dtype)

    counts = pairs.groupby(["from_repere", "to_repere"]).size().reset_index(name="n")
    counts["probability"] = counts["n"] / counts.groupby("from_repere")["n"].transform("sum")
    return counts.sort_values(["from_repere", "n"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Repères candidats début / fin d'opération
# ---------------------------------------------------------------------------
def candidate_start_reperes(transitions: pd.DataFrame, idle_value: int = config.IDLE_VALUE) -> pd.DataFrame:
    """
    Repères candidats "début d'opération" : ceux qui suivent le plus souvent
    `idle_value` (l'état de repos entre deux opérations). Triés par nombre
    d'occurrences décroissant — les premiers de la liste sont les points de
    départ d'opération les plus probables (ex: 100, 110 dans le cas décrit).
    """
    ensure_columns(transitions, ["from_repere", "to_repere", "n", "probability"])
    candidates = transitions[transitions["from_repere"] == idle_value]
    return (
        candidates.rename(columns={"to_repere": "repere"})[["repere", "n", "probability"]]
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )


def candidate_end_reperes(transitions: pd.DataFrame, idle_value: int = config.IDLE_VALUE) -> pd.DataFrame:
    """
    Repères candidats "fin d'opération" : ceux qui précèdent le plus souvent
    le retour à `idle_value`. Triés par nombre d'occurrences décroissant.
    """
    ensure_columns(transitions, ["from_repere", "to_repere", "n", "probability"])
    candidates = transitions[transitions["to_repere"] == idle_value]
    return (
        candidates.rename(columns={"from_repere": "repere"})[["repere", "n", "probability"]]
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 5. Table de référence process (doc constructeur/automaticien)
# ---------------------------------------------------------------------------
def load_pas_reference(path: str) -> pd.DataFrame:
    """
    Charge une table de référence pas -> plage de repères Att, extraite de la
    doc process du PU (ex: OP1410_pas_reference.csv, dérivée de OP1410.DOC).

    Colonnes attendues : pas_num, phase, code_court, description, att_min,
    att_max, type_attente, type_attente_detail, operation_liee, sens_liaison.
    `att_min`/`att_max` peuvent être vides (NaN) pour les pas dont la doc ne
    précise pas de plage de repère observable (ex: pas de défaut, qui
    déclenchent une action immédiate plutôt qu'une attente). `type_attente`
    catégorise la nature de l'attente caractéristique du pas — OPERATEUR
    (confirmation humaine), AUTRE_OPERATION (interlock avec une autre
    opération/PU), REMPLISSAGE, VIDANGE, CHAUFFE, REFROIDISSEMENT, PROCESS
    (condition de procédé non classable ailleurs), EQUIPEMENT (mise en
    route/arrêt d'un équipement) ou NA (aucune attente identifiée, pandas la
    lit comme NaN) — `type_attente_detail` donne le détail texte
    (message/condition observée dans la doc) qui a permis de trancher.
    `operation_liee`/`sens_liaison` (vides pour la plupart des pas)
    documentent un enchaînement avec une **autre** opération/PU repéré dans
    la doc — `operation_liee` est le code de l'autre opération (ex:
    `OP1420VA`), ou plusieurs séparés par `;` quand le pas dépend de
    **plusieurs** opérations à la fois (ex: `OP2330VA;OP1610VA` pour un pas
    de convergence qui vérifie le niveau de deux lignes amont différentes —
    cf. `OP2410_pas_reference.csv` pas 4, où la Condensation C5/SC15 lit à la
    fois le niveau SC15 (OP2330) et le niveau C5 (OP1610) avant de charger).
    `sens_liaison` vaut ATTEND (ce pas attend une info/autorisation de
    l'autre opération), ENVOIE (ce pas envoie une info/autorisation à
    l'autre opération) ou BIDIRECTIONNEL (les deux, ex: interlock partagé) —
    même sens appliqué à toutes les opérations listées quand il y en a
    plusieurs. Cf. `build_reference`/`enchainements` pour l'agrégation par
    tag et `documents/confidentiel/OPERATION.md` pour la méthode de
    construction.

    Cette table est un extrait **best-effort** du document source (extraction
    automatisée d'un fichier Word) : à vérifier/compléter au besoin plutôt
    qu'à considérer comme exhaustive à 100%.
    """
    ref = pd.read_csv(path)
    ensure_columns(ref, [
        "pas_num", "phase", "code_court", "description", "att_min", "att_max",
        "type_attente", "type_attente_detail", "operation_liee", "sens_liaison",
    ])
    return ref.sort_values("pas_num").reset_index(drop=True)


def label_with_pas(steps: pd.DataFrame, pas_reference: pd.DataFrame) -> pd.DataFrame:
    """
    Associe à chaque pas (plateau de repère) le numéro de PAS process
    correspondant, par recherche du plus grand `att_min` <= repere parmi les
    pas dont la plage est connue dans `pas_reference`.

    Un repère qui tombe avant le premier `att_min` connu (ou lorsque
    `pas_reference` ne couvre aucune plage) reçoit `pas_num = NaN` — non
    mappé, à exclure explicitement des étapes suivantes de reconstruction.
    """
    ensure_columns(steps, ["repere"])
    ensure_columns(pas_reference, [
        "pas_num", "phase", "code_court", "description", "att_min", "att_max", "type_attente",
    ])

    known = pas_reference.dropna(subset=["att_min"]).sort_values("att_min").reset_index(drop=True)
    out = steps.copy()

    if known.empty:
        out["pas_num"] = np.nan
    else:
        thresholds = known["att_min"].to_numpy()
        idx = np.searchsorted(thresholds, out["repere"].to_numpy(), side="right") - 1
        out["pas_num"] = np.where(idx >= 0, known["pas_num"].to_numpy()[idx.clip(min=0)], np.nan)

    return out.merge(
        known[["pas_num", "phase", "code_court", "description", "type_attente"]],
        on="pas_num",
        how="left",
    )


# ---------------------------------------------------------------------------
# 6. Reconstruction des opérations complètes (à partir des pas labellisés)
# ---------------------------------------------------------------------------
def _progress_flags(pas_num: np.ndarray, is_defaut: np.ndarray, operation_id: np.ndarray) -> np.ndarray:
    """
    Marque chaque pas comme "progrès standard" (True) ou "rework" (False).

    Un pas est du rework si et seulement si : c'est un pas de défaut
    lui-même, OU une régression **future** (dans la même opération) ramène
    à un niveau de `pas_num` inférieur ou égal au sien — auquel cas ce pas
    devra être refait, sa durée est donc perdue. Un pas jamais rattrapé par
    une régression ultérieure compte toujours comme standard, même s'il
    re-parcourt une plage de repère déjà vue *sans* défaut associé
    (plusieurs plateaux successifs du même `pas_num` sans aucune régression
    entre eux ne sont PAS du rework).

    ⚠️ Ne pas confondre avec "dépasse le maximum déjà atteint" (1ère
    version de ce calcul, abandonnée) : cette version-là excluait à tort la
    dernière tentative d'un pas quand elle ne fait que *rattraper* (sans le
    dépasser) le niveau atteint avant un défaut — cas fréquent d'un pas qui
    échoue une 1ère fois puis réussit au 2e essai (constaté sur `OP1410`,
    pas `EGOUTTAGE`/`AUTRE_OPERATION` : la tentative courte et ratée était
    comptée comme "standard", la tentative longue et réussie comme
    "rework", l'inverse de ce qu'il fallait). Le critère "régression future
    vers un niveau <= le sien" corrige ce biais : seule la présence d'une
    régression ultérieure invalide un pas, jamais le simple fait de revoir
    la même valeur.

    L'état (minimum des destinations de régression rencontrées plus loin
    dans la même opération) est réinitialisé à chaque changement
    d'`operation_id` — parcours de droite à gauche, les lignes étant déjà
    triées chronologiquement (`operation_id` croissant).
    """
    n = len(pas_num)
    progress = np.zeros(n, dtype=bool)
    min_later_defaut_dest = np.inf
    current_op = None
    for i in range(n - 1, -1, -1):
        if operation_id[i] != current_op:
            current_op = operation_id[i]
            min_later_defaut_dest = np.inf
        progress[i] = (not is_defaut[i]) and (pas_num[i] < min_later_defaut_dest)
        if is_defaut[i]:
            min_later_defaut_dest = min(min_later_defaut_dest, pas_num[i])
    return progress


def _real_pas_nums(pas_reference: pd.DataFrame) -> pd.Series:
    """Numéros de pas réels (plage de repère connue, hors défaut/HOLD)."""
    code_court = pas_reference["code_court"].astype(str).str.upper()
    is_defaut_like = code_court.str.startswith("DEF") | (code_court == "HOLD")
    return pas_reference.loc[pas_reference["att_min"].notna() & ~is_defaut_like, "pas_num"].dropna()


def find_restart_reference(
    labeled_steps: pd.DataFrame,
    pas_reference: pd.DataFrame,
    early_pas_fraction: float = 0.5,
) -> float:
    """
    Détermine, à partir des données réelles, la valeur de repère qui sert de
    référence de "nouveau départ" pour `reconstruct_operations`, sans aucune
    configuration par opération.

    Principe : parmi toutes les régressions du numéro de pas (hors celles
    dont le pas d'origine est un défaut/HOLD documenté, cf. `code_court`) qui
    atterrissent dans la **première moitié des pas réels de la séquence**
    (`early_pas_fraction`, appliquée au numéro de pas via la médiane des
    `pas_num` connus de `pas_reference`, hors défaut/HOLD), la valeur de
    repère la plus fréquente est le point d'entrée réel de l'opération —
    qu'il s'agisse du lancement normal (pas 1) ou d'un chaînage automatique
    qui saute directement à un pas plus loin, mais toujours situé tôt dans
    la séquence documentée.

    Ce filtre **structurel** (position dans la séquence de pas connue,
    *pas* une valeur de repère ou une fréquence d'occurrence) est essentiel
    pour ignorer les boucles internes documentées (répétitions de
    filtration, alternance d'alimentation, ...), qui rebouclent toujours
    vers un pas de phase *plus tardif* dans la séquence :
    - constaté sur `OP3120`, où la boucle de filtration (retour à un pas
      situé dans la 2e moitié de la séquence) est **plus fréquente** que
      les vrais redémarrages — un simple mode global ou une pondération
      par fréquence la ferait dominer à tort ; le filtre par position
      l'exclut quelle que soit sa fréquence ;
    - garde en revanche, quelle que soit sa fréquence relative, tout
      redémarrage rare mais légitime qui atterrit encore tôt dans la
      séquence (ex: une relance manuelle au pas 1, à côté d'un chaînage
      dominant au pas 2 ou 4) — les deux contribuent au même pool de
      candidats, dont on prend ensuite le mode.

    Returns
    -------
    La valeur de repère de référence (le mode des destinations éligibles),
    ou `np.nan` si aucune régression exploitable n'est observée (aucun
    redémarrage à détecter : traiter tout le jeu de données comme une seule
    opération).
    """
    ensure_columns(labeled_steps, ["pas_num", "repere", "start"])
    ensure_columns(pas_reference, ["pas_num", "code_court", "att_min"])

    steps = labeled_steps.dropna(subset=["pas_num"]).sort_values("start").reset_index(drop=True)
    pas_nums = steps["pas_num"].to_numpy()
    reperes = steps["repere"].to_numpy(dtype=float)
    n = len(steps)

    regressed = np.zeros(n, dtype=bool)
    regressed[1:] = pas_nums[1:] < pas_nums[:-1]

    code_court = steps["code_court"].astype(str).str.upper() if "code_court" in steps.columns else pd.Series("", index=steps.index)
    prev_code_court = code_court.shift().fillna("")
    came_from_defaut = (prev_code_court.str.startswith("DEF") | (prev_code_court == "HOLD")).to_numpy()

    real_pas_nums = _real_pas_nums(pas_reference)
    early_pas_cutoff = (
        real_pas_nums.quantile(early_pas_fraction) if not real_pas_nums.empty else np.inf
    )
    is_early = pas_nums <= early_pas_cutoff

    eligible = regressed & ~came_from_defaut & is_early
    candidates = reperes[eligible]
    if len(candidates) == 0:
        return float("nan")

    values, counts = np.unique(candidates, return_counts=True)
    return float(values[np.argmax(counts)])


def reconstruct_operations(
    labeled_steps: pd.DataFrame,
    pas_reference: pd.DataFrame,
    restart_tolerance: float = config.RESTART_TOLERANCE,
    restart_reference: float = None,
    large_jump_fraction: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reconstruit les opérations complètes à partir des pas labellisés par
    `label_with_pas`.

    Modèle métier — 3e génération (cf. OPERATION.md section 7 pour
    l'historique des deux générations précédentes, abandonnées) :

    le numéro de pas croît pendant une opération ; une régression du numéro
    de pas signale un NOUVEAU DÉPART si l'une de ces deux conditions est
    vraie, un DÉFAUT du pas *précédent* (retry) sinon :

    1. sa valeur de repère revient au point d'entrée réel de l'opération
       (`restart_reference`, ± `restart_tolerance`) — "on remonte un
       escalier, puis hop, on retombe à l'origine, et ça repart" ;
    2. le saut en numéro de pas (origine − destination) est **trop grand
       pour être un défaut plausible** : un défaut ne peut renvoyer que
       vers un pas proche de celui qui vient d'échouer (le "PAS DE
       REESSAI" documenté, toujours à quelques pas de là) — jamais à
       l'autre bout de la séquence. Un saut ≥ `large_jump_fraction` de
       l'étendue totale des pas réels de `pas_reference` (50% par défaut)
       ne peut donc être qu'un redémarrage, quelle que soit sa valeur de
       repère. Constaté en pratique : un retour du pas 21 au pas 4 classé
       à tort en défaut par le seul critère de valeur (le pas 4 n'était
       pas assez proche de `restart_reference` pour ce tag), alors qu'un
       défaut du pas 21 ne peut structurellement pas renvoyer si loin en
       arrière.

    `restart_reference` est calculé automatiquement par
    `find_restart_reference` si non fourni — pas besoin de connaître à
    l'avance quel pas de lancement est atteint, ni de deviner combien de
    pas le chaînage automatique saute. Contrairement à une 1ère tentative
    de ce modèle (comparer chaque régression à la valeur de départ de
    l'opération *en cours*, abandonnée : elle se bloque dès qu'un
    redémarrage rare mais plus bas fixe une référence trop basse pour
    reconnaître ensuite les redémarrages chaînés, bien plus fréquents mais
    à une valeur plus haute, fusionnant alors tout le reste de l'historique
    en une seule opération), cette référence est calculée **une seule
    fois, globalement** — donc stable, sans effet de blocage.

    Les pas non mappés (`pas_num` NaN, ex: repère hors de toute plage
    connue) sont ignorés pour la reconstruction (ni un début, ni un défaut).

    Returns
    -------
    (operations, labeled_steps_with_flags)
        operations : une ligne par opération reconstruite — start, end,
            duration_min (durée totale), n_pas, n_defauts, max_pas_reached,
            standard_duration_min (durée totale moins le temps de
            récupération après défaut, cf. `_progress_flags`), rework_duration_min
            (duration_min - standard_duration_min, le temps "perdu" en retries)
        labeled_steps_with_flags : `labeled_steps` (pas non mappés exclus)
            + colonnes `operation_id`, `is_defaut`, `is_progress`
    """
    ensure_columns(labeled_steps, ["pas_num", "repere", "start", "end", "duration_min"])

    steps = labeled_steps.dropna(subset=["pas_num"]).sort_values("start").reset_index(drop=True)
    steps["pas_num"] = steps["pas_num"].astype(int)

    n = len(steps)
    pas_nums = steps["pas_num"].to_numpy()
    reperes = steps["repere"].to_numpy(dtype=float)

    regressed = np.zeros(n, dtype=bool)
    regressed[1:] = pas_nums[1:] < pas_nums[:-1]

    if "code_court" in steps.columns:
        prev_code_court = steps["code_court"].shift().astype(str).str.upper()
        came_from_defaut = (prev_code_court.str.startswith("DEF") | (prev_code_court == "HOLD")).to_numpy()
    else:
        came_from_defaut = np.zeros(n, dtype=bool)

    if restart_reference is None:
        restart_reference = find_restart_reference(labeled_steps, pas_reference)

    eligible = regressed & ~came_from_defaut

    matches_reference = np.zeros(n, dtype=bool)
    if not np.isnan(restart_reference):
        threshold = restart_reference * restart_tolerance
        matches_reference[eligible] = reperes[eligible] <= threshold

    real_pas_nums = _real_pas_nums(pas_reference)
    pas_span = real_pas_nums.max() - real_pas_nums.min() if not real_pas_nums.empty else 0
    is_large_jump = np.zeros(n, dtype=bool)
    if pas_span > 0:
        jump = np.zeros(n, dtype=int)
        jump[1:] = pas_nums[:-1] - pas_nums[1:]
        is_large_jump[eligible] = jump[eligible] > large_jump_fraction * pas_span

    is_new_operation = eligible & (matches_reference | is_large_jump)

    if n > 0:
        is_new_operation[0] = True  # première ligne du jeu de données = début d'une opération (peut être tronquée)

    steps["operation_id"] = np.cumsum(is_new_operation) - 1
    steps["is_defaut"] = regressed & ~is_new_operation

    # Temps de rework = uniquement le temps de RÉCUPÉRATION après un défaut
    # (le pas de défaut lui-même + tout le temps mis à revenir au niveau
    # déjà atteint avant ce défaut), pas "tout pas qui ne dépasse pas
    # strictement le maximum déjà vu" : un pas peut légitimement être
    # retraversé plusieurs fois (plusieurs plateaux de repère dans sa
    # même plage) sans aucun défaut associé — ce temps compte comme
    # standard, pas comme perdu. Corrige un biais systématique constaté
    # sur OP1410 (temps standard médian ~190 min alors que l'exploitant
    # sait qu'un cycle sans aléas prend au moins ~500 min) : l'ancienne
    # définition ("ne compte que la 1ère fois qu'un pas_num dépasse le
    # max déjà atteint") écartait à tort la durée du pas qui RÉUSSIT
    # après un défaut (ex: attente d'une autre opération qui échoue vite
    # une 1ère fois puis aboutit après une attente réelle plus longue au
    # 2e essai) — c'est la 1ère tentative, courte et ratée, qui était
    # comptée comme "standard", et la 2e tentative, longue et réussie,
    # qui était classée à tort en "rework".
    steps["is_progress"] = _progress_flags(
        steps["pas_num"].to_numpy(), steps["is_defaut"].to_numpy(), steps["operation_id"].to_numpy()
    )
    steps["progress_duration_min"] = steps["duration_min"].where(steps["is_progress"], 0.0)

    operations = (
        steps.groupby("operation_id")
        .agg(
            start=("start", "first"),
            end=("end", "last"),
            n_pas=("pas_num", "size"),
            n_defauts=("is_defaut", "sum"),
            max_pas_reached=("pas_num", "max"),
            standard_duration_min=("progress_duration_min", "sum"),
        )
        .reset_index()
    )
    operations["duration"] = operations["end"] - operations["start"]
    operations["duration_min"] = operations["duration"].dt.total_seconds() / 60
    operations["rework_duration_min"] = operations["duration_min"] - operations["standard_duration_min"]

    return operations, steps


def exclude_boundary_operations(operations: pd.DataFrame) -> pd.DataFrame:
    """
    Retire la première et la dernière opération (par ordre chronologique de
    démarrage) des statistiques de durée.

    `reconstruct_operations` avertit déjà que la toute première opération
    reconstruite peut être **tronquée** : la période interrogée commence en
    cours de cycle, pas nécessairement au pas de départ — sa durée totale et
    son temps standard sont alors mesurés sur un cycle incomplet (il manque
    le temps écoulé avant le début de la période). Symétriquement, la
    dernière opération peut être **en cours** au moment où la période
    interrogée s'arrête (surtout si `end` est proche d'aujourd'hui) : son
    dernier pas connu n'est peut-être pas le vrai dernier pas du cycle.

    Une seule opération tronquée suffit à fausser un minimum (constaté :
    52 min de "temps standard minimum" sur `OP1410`, alors qu'un cycle
    complet ne descend jamais sous ~500 min) — d'où l'intérêt de l'exclure
    des statistiques de durée (moyenne/médiane/minimum/SPC) plutôt que de
    la laisser polluer silencieusement des chiffres censés représenter des
    cycles complets. Le nombre d'opérations (`n_operations`) et les
    décomptes de défauts, eux, restent basés sur la table complète : ces
    deux opérations ont bien eu lieu, seule leur *durée* n'est pas fiable.

    Ne retire rien si `operations` a moins de 3 lignes (il ne resterait
    rien d'exploitable).
    """
    if len(operations) < 3:
        return operations
    return operations.sort_values("start").iloc[1:-1].reset_index(drop=True)


def defaut_frequency_table(labeled_steps_with_flags: pd.DataFrame) -> pd.DataFrame:
    """
    Compte, pour chaque pas, le nombre de défauts détectés — un défaut est
    attribué au pas *précédent* la régression (cf. `reconstruct_operations`),
    puisque c'est ce pas qui a échoué avant le retry.
    """
    ensure_columns(labeled_steps_with_flags, ["pas_num", "is_defaut"])
    steps = labeled_steps_with_flags.sort_values("start").reset_index(drop=True)

    faulty_pas = steps["pas_num"].shift()[steps["is_defaut"]]
    counts = faulty_pas.value_counts().rename_axis("pas_num").reset_index(name="n_defauts")
    counts["pas_num"] = counts["pas_num"].astype(int)
    return counts.sort_values("n_defauts", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Temps de cycle et chaînage entre opérations (ancré sur un repère précis)
# ---------------------------------------------------------------------------
def cycle_times(steps: pd.DataFrame, anchor_repere: int) -> pd.DataFrame:
    """
    Temps de cycle (lancement à lancement), mesuré entre deux occurrences
    consécutives d'un repère précis qui marque le lancement confirmé d'une
    opération (`anchor_repere` — ex: 110, cf. doc process : Att=100 est la
    question "Voulez-vous lancer l'opération ?", Att=110 la confirmation qui
    fait immédiatement passer au pas suivant).

    Contrairement à `reconstruct_operations` (ancré sur toute la plage du
    pas de départ), cette fonction isole précisément l'instant de lancement
    confirmé, ce qui sépare deux temps différents :

    - `attente_lancement_min` : durée du plateau à `anchor_repere` lui-même
      — le temps où le PU est prêt/en attente avant que l'opération suivante
      ne démarre réellement. C'est le temps de **chaînage** entre deux
      opérations (changeover), pas un temps de process.
    - `cycle_time_min` : écart entre ce lancement et le suivant — le temps
      total d'une opération, lancement à lancement (inclut l'`attente_lancement_min`
      de l'opération *suivante*, puisqu'elle se termine juste avant celle-ci).

    La dernière ligne a `cycle_time_min` = NaN (pas de lancement suivant
    observé sur la période).
    """
    ensure_columns(steps, ["repere", "start", "duration_min"])
    anchors = (
        steps[steps["repere"] == anchor_repere]
        .sort_values("start")
        .reset_index(drop=True)
        .rename(columns={"duration_min": "attente_lancement_min"})
    )
    anchors["cycle_time_min"] = (
        anchors["start"].shift(-1) - anchors["start"]
    ).dt.total_seconds() / 60
    return anchors[["start", "attente_lancement_min", "cycle_time_min"]]


def best_chaining_window(cycle_times_df: pd.DataFrame, window: int = 5) -> float:
    """
    Temps de chaînage minimal **soutenable** sur `window` lancements
    consécutifs : moyenne par lancement de la meilleure séquence de
    `window` `attente_lancement_min` consécutives.

    Réutilise la recherche de meilleure série glissante de `nop_analysis`
    (même logique que le "golden run") plutôt que le simple minimum
    ponctuel : un minimum isolé peut être un concours de circonstances,
    une série soutenue sur plusieurs lancements est une cible plus robuste.

    Retourne NaN si `anchor_repere` n'a jamais été observé comme un
    plateau distinct (ou moins de `window` fois) dans les données —
    typique d'un repère de lancement transitoire, jamais capturé par
    l'échantillonnage minute, plutôt qu'une erreur à propager.
    """
    ensure_columns(cycle_times_df, ["start", "attente_lancement_min"])
    ops_like = (
        cycle_times_df.dropna(subset=["attente_lancement_min"])
        .rename(columns={"attente_lancement_min": "duration_min"})
        .sort_values("start")
        .reset_index(drop=True)
    )
    if len(ops_like) < window:
        return float("nan")
    ops_like["end"] = ops_like["start"] + pd.to_timedelta(ops_like["duration_min"], unit="min")
    total = _nop_get_reference_duration(ops_like, window=window)
    return total / window


# ---------------------------------------------------------------------------
# 8. Statistiques par pas (base de comparaison pour une opération future)
# ---------------------------------------------------------------------------
def pas_duration_stats(labeled_steps_with_flags: pd.DataFrame, pas_reference: pd.DataFrame) -> pd.DataFrame:
    """
    Statistiques de durée par pas (n, moyenne, médiane, écart-type, P10/P90,
    CV), enrichies du libellé du pas — sert de base de comparaison pour
    positionner une opération future pas par pas (cf. `build_reference`).
    """
    ensure_columns(labeled_steps_with_flags, ["pas_num", "duration_min"])
    grouped = labeled_steps_with_flags.groupby("pas_num")["duration_min"]

    stats = grouped.agg(
        n="count",
        mean_duration_min="mean",
        median_duration_min="median",
        std_duration_min="std",
        min_duration_min="min",
        max_duration_min="max",
    ).reset_index()
    stats["p10_duration_min"] = grouped.quantile(0.10).to_numpy()
    stats["p90_duration_min"] = grouped.quantile(0.90).to_numpy()
    stats["cv"] = stats["std_duration_min"] / stats["mean_duration_min"]
    stats["pas_num"] = stats["pas_num"].astype(int)

    return stats.merge(
        pas_reference[["pas_num", "phase", "code_court", "description", "type_attente", "type_attente_detail"]],
        on="pas_num",
        how="left",
    ).sort_values("pas_num").reset_index(drop=True)


def reference_cycle_time(
    labeled_steps: pd.DataFrame,
    pas_reference: pd.DataFrame,
    percentile: float = 10.0,
) -> dict:
    """
    Temps de cycle de référence ("BDP") : somme, sur tous les pas réels
    documentés (hors défaut/HOLD), du `percentile`-ième percentile de durée
    observée pour ce pas sur tout l'historique (P10 par défaut).

    ⚠️ Le minimum brut (percentile=0, essayé en premier) est une
    statistique extrême : sur les 300-500 occurrences typiquement
    observées par pas sur un an d'historique, il est presque garanti qu'au
    moins une tombe très bas par pur hasard (constaté sur `OP1410` : somme
    des minimums = 63 min, contre une référence terrain connue de
    425 min). Le P10 (10% des occurrences réelles font aussi bien ou
    mieux) reste ambitieux sans être dominé par un unique outlier —
    vérifié sur `OP1410` : somme des P10 = 382 min, somme des médianes =
    491 min, qui encadrent bien la référence terrain (425 min).

    Indépendant de `reconstruct_operations` (donc insensible aux aléas de
    reconstruction des frontières d'opération) : chaque pas réel est
    mesuré minute par minute par `extract_steps`/`label_with_pas` (un
    plateau = tant que le repère reste dans la même plage), sans avoir à
    décider si telle régression est un défaut ou un redémarrage.

    Returns
    -------
    dict : total_min (somme du percentile par pas réel), n_pas_reels,
        par_pas (liste de {pas_num, code_court, duration_min}).
    """
    ensure_columns(labeled_steps, ["pas_num", "duration_min"])
    real_pas_nums = _real_pas_nums(pas_reference)

    real_steps = labeled_steps[labeled_steps["pas_num"].isin(real_pas_nums)]
    per_pas = (
        real_steps.groupby("pas_num")["duration_min"]
        .quantile(percentile / 100)
        .reset_index()
        .merge(pas_reference[["pas_num", "code_court"]], on="pas_num", how="left")
    )

    return {
        "total_min": float(per_pas["duration_min"].sum()),
        "n_pas_reels": int(len(per_pas)),
        "par_pas": per_pas.to_dict("records"),
    }


def extract_enchainements(pas_reference: pd.DataFrame) -> list:
    """
    Liste des pas qui enchaînent avec une **autre** opération/PU
    (`operation_liee` renseigné dans la table process), avec le sens de la
    liaison (ATTEND / ENVOIE / BIDIRECTIONNEL).

    `operation_liee` peut lister plusieurs opérations séparées par `;` (pas
    de convergence qui dépend de plusieurs lignes amont à la fois) : une
    entrée est émise par opération listée, avec le même `pas_num`/`sens`.

    Sert à documenter, indépendamment des données mesurées, les dépendances
    inter-opérations identifiées dans la doc process (étape 1, analyse du
    document — cf. `documents/confidentiel/OPERATION.md`) ; `build_reference`
    l'inclut telle quelle dans le bundle de référence (étape 2).
    """
    ensure_columns(pas_reference, ["pas_num", "code_court", "operation_liee", "sens_liaison", "type_attente_detail"])
    linked = pas_reference.dropna(subset=["operation_liee"])
    records = []
    for row in linked.to_dict("records"):
        for op in str(row["operation_liee"]).split(";"):
            op = op.strip()
            if not op:
                continue
            records.append({
                "pas_num": row["pas_num"],
                "code_court": row["code_court"],
                "operation_liee": op,
                "sens_liaison": row["sens_liaison"],
                "detail": row["type_attente_detail"],
            })
    return records


# ---------------------------------------------------------------------------
# 9. Synthèse et référence exportable (repère pour une opération future)
# ---------------------------------------------------------------------------
def build_reference(
    pas_reference: pd.DataFrame,
    operations: pd.DataFrame,
    labeled_steps: pd.DataFrame,
    defauts: pd.DataFrame,
    cycle_times_df: pd.DataFrame,
    tag: str,
    anchor_repere: int,
    chaining_window: int = 5,
    description: str = "",
    operation: str = "",
    produit: str = "",
) -> dict:
    """
    Assemble un bundle de référence (statistiques globales + par pas) pour
    un tag/opération, à exporter (`save_reference`) et réutiliser comme
    repère pour diagnostiquer une opération future
    (`compare_operation_to_reference`) : carte SPC, positionnement en
    défauts, temps de cycle, temps d'attente au lancement.

    Inclut aussi `pas_reference` (le contenu intégral de la table process,
    ex: OP1410_pas_reference.csv) et `enchainements` (les pas qui dépendent
    d'une autre opération/PU, cf. `extract_enchainements`) : le bundle JSON
    reste ainsi autoporteur — pas besoin de recharger le CSV séparément pour
    connaître les plages de repère, phases, libellés et dépendances
    inter-opérations de chaque pas.

    `description`, `operation`, `produit` sont de simples métadonnées
    d'identification (cf. `documents/confidentiel/att_operations_config.json`,
    le catalogue source des opérations suivies) reprises telles quelles dans
    le bundle — utile pour retrouver, à la lecture d'un JSON isolé, à quel
    produit/opération il correspond sans revenir au catalogue.
    """
    ensure_columns(operations, ["duration_min", "standard_duration_min", "rework_duration_min", "n_defauts"])

    # Les statistiques de DURÉE (moyenne/médiane/minimum/SPC) s'appuient sur
    # les opérations "complètes" uniquement : la première et la dernière
    # peuvent être tronquées (période interrogée qui commence/s'arrête en
    # cours de cycle, cf. exclude_boundary_operations) et fausseraient sinon
    # gravement un minimum en particulier. n_operations et les décomptes de
    # défauts, eux, restent basés sur la table complète (ces opérations ont
    # bien eu lieu, seule leur durée n'est pas fiable comme référence).
    complete_ops = exclude_boundary_operations(operations)
    if complete_ops.empty:
        complete_ops = operations

    if len(complete_ops) >= 2:
        limits = _nop_compute_control_limits(complete_ops, value_col="duration_min")
    else:
        # Moins de 2 opérations (historique trop court, ou tag encore mal
        # reconstruit, cf. OPERATION.md section 7) : pas d'étendue mobile
        # calculable, on ne bloque pas tout le bundle pour autant.
        limits = {"center": float("nan"), "sigma_hat": float("nan"), "ucl": float("nan"),
                  "lcl": float("nan"), "mr_bar": float("nan")}
    pas_stats = pas_duration_stats(labeled_steps, pas_reference)
    enchainements = extract_enchainements(pas_reference)

    n_ops = len(operations)
    n_ops_with_defaut = int((operations["n_defauts"] > 0).sum())
    total_defauts = int(defauts["n_defauts"].sum())

    defauts_by_pas = defauts.merge(
        pas_reference[["pas_num", "code_court", "type_attente"]], on="pas_num", how="left"
    )
    defauts_by_pas["pct_of_total_defauts"] = (
        defauts_by_pas["n_defauts"] / total_defauts * 100 if total_defauts else 0.0
    )

    valid_cycle = cycle_times_df["cycle_time_min"].dropna()
    attente = cycle_times_df["attente_lancement_min"]

    return {
        "tag": tag,
        "description": description,
        "operation": operation,
        "produit": produit,
        "anchor_repere": int(anchor_repere),
        "n_operations": n_ops,
        "n_operations_used_for_duration_stats": len(complete_ops),
        "chaining_window": chaining_window,
        "pas_reference": pas_reference.to_dict("records"),
        "enchainements": enchainements,
        "cycle_time_min": {
            "mean": float(valid_cycle.mean()),
            "median": float(valid_cycle.median()),
            "p10": float(valid_cycle.quantile(0.10)),
            "p90": float(valid_cycle.quantile(0.90)),
        },
        "attente_lancement_min": {
            "mean": float(attente.mean()),
            "median": float(attente.median()),
            "min": float(attente.min()),
            "best_window_mean": float(best_chaining_window(cycle_times_df, chaining_window)),
        },
        "duration_min": {
            "mean": float(complete_ops["duration_min"].mean()),
            "median": float(complete_ops["duration_min"].median()),
            "spc": {k: float(v) for k, v in limits.items()},
        },
        "standard_duration_min": {
            "mean": float(complete_ops["standard_duration_min"].mean()),
            "median": float(complete_ops["standard_duration_min"].median()),
            "min": float(complete_ops["standard_duration_min"].min()),
        },
        "rework_duration_min": {
            "mean": float(complete_ops["rework_duration_min"].mean()),
            "median": float(complete_ops["rework_duration_min"].median()),
            "pct_of_total_time": float(
                complete_ops["rework_duration_min"].sum() / complete_ops["duration_min"].sum() * 100
            ),
        },
        "defauts": {
            "n_operations_with_defaut": n_ops_with_defaut,
            "pct_operations_with_defaut": float(n_ops_with_defaut / n_ops * 100) if n_ops else 0.0,
            "total": total_defauts,
            "by_pas": defauts_by_pas[
                ["pas_num", "code_court", "type_attente", "n_defauts", "pct_of_total_defauts"]
            ].to_dict("records"),
        },
        "pas_stats": pas_stats.drop(columns=["phase", "description"], errors="ignore").to_dict("records"),
    }


def _json_default(obj):
    """Convertit les scalaires numpy (int64, float64, bool_...) en types natifs pour json.dump."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_reference(reference: dict, path: str) -> None:
    """
    Exporte le bundle de référence en JSON (préféré à un CSV à plat : la
    référence mélange statistiques globales et détail par pas, une
    structure imbriquée que JSON représente nativement).
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2, ensure_ascii=False, default=_json_default)


def load_reference(path: str) -> dict:
    """Recharge un bundle de référence exporté par `save_reference`."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 10. Diagnostic d'une opération future par rapport à la référence
# ---------------------------------------------------------------------------
def compare_operation_to_reference(operation_steps: pd.DataFrame, reference: dict) -> dict:
    """
    Compare une opération (ses pas labellisés, ex: `labeled_steps` d'un
    `reconstruct_operations` filtré sur un seul `operation_id`) au bundle de
    référence produit par `build_reference`.

    Retourne un diagnostic : durée totale vs limites SPC (dans/hors
    contrôle), écart à la médiane historique, défauts détectés et pas
    concernés, à comparer au taux de défaut historique par opération.
    """
    ensure_columns(operation_steps, ["pas_num", "duration_min", "is_defaut"])

    duration = float(operation_steps["duration_min"].sum())
    spc = reference["duration_min"]["spc"]
    median_ref = reference["duration_min"]["median"]

    faulty_pas = (
        operation_steps["pas_num"].shift()[operation_steps["is_defaut"]].dropna().astype(int).tolist()
    )
    ref_rate_per_operation = (
        reference["defauts"]["total"] / reference["n_operations"] if reference["n_operations"] else float("nan")
    )

    return {
        "duration_min": duration,
        "duration_vs_median_pct": (duration / median_ref - 1) * 100 if median_ref else float("nan"),
        "spc_out_of_control": bool(duration > spc["ucl"] or duration < spc["lcl"]),
        "spc_limits": spc,
        "n_defauts": len(faulty_pas),
        "defaut_pas": faulty_pas,
        "reference_defaut_rate_per_operation": ref_rate_per_operation,
    }


# ---------------------------------------------------------------------------
# 11. Réseau des liaisons inter-opérations (graphe + base de chronogramme)
# ---------------------------------------------------------------------------
def build_operations_network(entries: list) -> dict:
    """
    Assemble le réseau des liaisons inter-opérations à partir des tables
    process déjà construites (étape 1), pour servir de base à un graphe de
    dépendances ("network") et à un chronogramme.

    `entries` : liste de dicts, un par opération dont la `pas_reference` a
    été construite —
        {"operation": str, "nom": str, "tag": str, "produit": str,
         "description": str, "pas_reference": DataFrame}
    (`pas_reference` déjà chargée via `load_pas_reference`).

    Returns
    -------
    dict {"operations": [...], "liaisons": [...]}
        "operations" : une entrée par opération citée (celles de `entries`
            **et** toute opération référencée en `operation_liee` même si
            elle n'est pas encore dans `entries` — `in_catalog=False` dans
            ce cas, pour ne pas perdre le lien tant que cette opération
            n'a pas sa propre table construite).
        "liaisons" : une entrée par pas avec un enchaînement documenté
            (`extract_enchainements`), `operation_cible` normalisée (le
            suffixe VA des codes type `OP1410VA` est retiré pour matcher
            le code `operation` du catalogue, ex: `OP1410`).
    """
    catalog_by_op = {e["operation"]: e for e in entries}
    node_ops = dict.fromkeys(catalog_by_op)  # dict = set ordonné (ordre d'apparition)

    liaisons = []
    for e in entries:
        for link in extract_enchainements(e["pas_reference"]):
            target_op = re.sub(r"VA$", "", link["operation_liee"])
            node_ops.setdefault(target_op, None)
            liaisons.append({
                "operation_source": e["operation"],
                "operation_cible": target_op,
                "pas_source": link["pas_num"],
                "code_court_source": link["code_court"],
                "sens": link["sens_liaison"],
                "detail": link["detail"],
            })

    operations = []
    for op in node_ops:
        e = catalog_by_op.get(op)
        if e is not None:
            operations.append({
                "operation": op, "nom": e["nom"], "tag": e["tag"],
                "produit": e["produit"], "description": e["description"],
                "in_catalog": True,
            })
        else:
            operations.append({
                "operation": op, "nom": None, "tag": None,
                "produit": None, "description": None,
                "in_catalog": False,
            })

    return {"operations": operations, "liaisons": liaisons}
