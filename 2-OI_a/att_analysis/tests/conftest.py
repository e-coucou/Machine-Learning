"""
Fixtures partagées pour les tests unitaires de att_analysis.

Le jeu de données `sample_df` simule deux opérations identiques séparées par
un retour à l'état de repos (repère 0), chacune composée de 3 pas
(100 -> 101 -> 102) — un modèle minimal mais suffisant pour vérifier la
fréquence par repère, la matrice de transition et les repères candidats
début/fin de façon déterministe (chaque repère n'a qu'un seul successeur
possible dans ce jeu de données, donc `probability` vaut toujours 1.0).

Pas (9 plateaux, 23 échantillons à la minute) :
    repere=0   : lignes  0- 2 (3 échantillons) -> durée 3 min
    repere=100 : lignes  3- 4 (2 échantillons) -> durée 2 min
    repere=101 : lignes  5- 7 (3 échantillons) -> durée 3 min
    repere=102 : lignes  8- 9 (2 échantillons) -> durée 2 min
    repere=0   : lignes 10-12 (3 échantillons) -> durée 3 min
    repere=100 : lignes 13-14 (2 échantillons) -> durée 2 min
    repere=101 : lignes 15-17 (3 échantillons) -> durée 3 min
    repere=102 : lignes 18-19 (2 échantillons) -> durée 2 min
    repere=0   : lignes 20-22 (3 échantillons) -> durée 2 min (dernier pas :
                 sa "fin" est le dernier timestamp du jeu de données, pas le
                 début d'un pas suivant — même convention que
                 nop_analysis.extract_operations)
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def att_pas_reference():
    """
    Table de référence pas -> plage de repère minimale (4 pas), sur le
    modèle de OP1410_pas_reference.csv : pas 1 = attente de lancement
    (100-110), pas 2 = préparation (200-250), pas 3 = production (300),
    pas 4 = défaut (pas de plage connue, comme les vrais pas de défaut de
    la doc process).
    """
    return pd.DataFrame([
        {"pas_num": 1, "phase": "ATTENTE", "code_court": "ATTENTE", "description": "Attente",
         "att_min": 100, "att_max": 110,
         "type_attente": "OPERATEUR", "type_attente_detail": "Confirmation de lancement",
         "operation_liee": np.nan, "sens_liaison": np.nan},
        {"pas_num": 2, "phase": "PREP", "code_court": "PREP", "description": "Preparation",
         "att_min": 200, "att_max": 250,
         "type_attente": "PROCESS", "type_attente_detail": "Verification avant demarrage",
         "operation_liee": np.nan, "sens_liaison": np.nan},
        {"pas_num": 3, "phase": "PROD", "code_court": "PROD", "description": "Production",
         "att_min": 300, "att_max": 300,
         "type_attente": "CHAUFFE", "type_attente_detail": "Attente temperature",
         "operation_liee": "OP_TEST", "sens_liaison": "ATTEND"},
        {"pas_num": 4, "phase": "DEF", "code_court": "DEF.PREP", "description": "Defaut preparation",
         "att_min": np.nan, "att_max": np.nan,
         "type_attente": "OPERATEUR", "type_attente_detail": "Relance apres defaut",
         "operation_liee": np.nan, "sens_liaison": np.nan},
    ])


@pytest.fixture
def att_labeled_steps_two_ops(att_pas_reference):
    """
    13 pas simulant deux opérations séparées par un retour au pas 1 (porte
    de lancement), avec un défaut au milieu de la première opération : la
    séquence redescend de PROD (pas 3) à PREP (pas 2) sans repasser par le
    pas 1 — donc un défaut du pas PROD, pas une nouvelle opération.

        100(pas1) 110(pas1) 200(pas2) 250(pas2) 220(pas2) 300(pas3)
        210(pas2, défaut du pas3) 250(pas2) 300(pas3)          -> opération 0
        100(pas1) 110(pas1) 200(pas2) 300(pas3)                -> opération 1
    """
    from att_analysis import label_with_pas

    reperes = [100, 110, 200, 250, 220, 300, 210, 250, 300, 100, 110, 200, 300]
    n = len(reperes)
    starts = pd.date_range("2026-01-01", periods=n, freq="10min", tz="UTC")
    ends = starts + pd.Timedelta(minutes=10)
    steps = pd.DataFrame({
        "step_id": range(n),
        "repere": reperes,
        "start": starts,
        "end": ends,
        "duration": ends - starts,
        "duration_min": 10.0,
        "n_samples": 10,
    })
    return label_with_pas(steps, att_pas_reference)


@pytest.fixture
def att_pas_reference_with_hold():
    """
    Table pas -> plage de repère sur le modèle des documents "MANUEL
    UTILISATEUR" (OP1510/1520/1530/1540) : un unique pas `HOLD` partagé pour
    tous les défauts/pauses de l'opération, qui — contrairement aux pas
    `DEF.*` des documents "SEQ PAS\\H" — a sa propre plage de repère
    (`att_min`/`att_max` non vides) et apparaît donc explicitement dans les
    pas labellisés.
    """
    return pd.DataFrame([
        {"pas_num": 1, "phase": "RUN", "code_court": "ATTENTE", "description": "Attente",
         "att_min": 10, "att_max": 200,
         "type_attente": "OPERATEUR", "type_attente_detail": "Confirmation de lancement",
         "operation_liee": np.nan, "sens_liaison": np.nan},
        {"pas_num": 2, "phase": "RUN", "code_court": "INIT_PHASE", "description": "Initialisation phase",
         "att_min": 220, "att_max": 300,
         "type_attente": "PROCESS", "type_attente_detail": "Verification avant demarrage",
         "operation_liee": np.nan, "sens_liaison": np.nan},
        {"pas_num": 3, "phase": "RUN", "code_court": "CHARGE", "description": "Chargement",
         "att_min": 310, "att_max": 400,
         "type_attente": "REMPLISSAGE", "type_attente_detail": "Chargement",
         "operation_liee": np.nan, "sens_liaison": np.nan},
        {"pas_num": 4, "phase": np.nan, "code_court": "HOLD", "description": "Defaut/pause commun",
         "att_min": 9900, "att_max": 9900,
         "type_attente": "OPERATEUR", "type_attente_detail": "Relance apres defaut",
         "operation_liee": np.nan, "sens_liaison": np.nan},
    ])


@pytest.fixture
def att_labeled_steps_hold_defaut_returns_to_widened_range(att_pas_reference_with_hold):
    """
    6 pas simulant une seule opération réelle : cycle complet (pas1->pas2->
    pas3), un défaut au pas3 qui bascule au HOLD, puis une relance
    ("PAS DE REESSAI") qui revient directement au pas2 (repère 220) — une
    valeur supérieure à la valeur de départ de l'opération (repère 100),
    donc reconnue comme un défaut de l'opération en cours, pas comme un
    nouveau départ.

        100(pas1) 220(pas2) 310(pas3) 9900(HOLD, defaut) 220(pas2, relance) 310(pas3)
    """
    from att_analysis import label_with_pas

    reperes = [100, 220, 310, 9900, 220, 310]
    n = len(reperes)
    starts = pd.date_range("2026-01-01", periods=n, freq="10min", tz="UTC")
    ends = starts + pd.Timedelta(minutes=10)
    steps = pd.DataFrame({
        "step_id": range(n),
        "repere": reperes,
        "start": starts,
        "end": ends,
        "duration": ends - starts,
        "duration_min": 10.0,
        "n_samples": 10,
    })
    return label_with_pas(steps, att_pas_reference_with_hold)


@pytest.fixture
def att_labeled_steps_chained_ops(att_pas_reference):
    """
    9 pas simulant trois opérations enchaînées automatiquement : elles ne
    repassent jamais par le pas 1 (ATTENTE, 100-110) et redémarrent
    directement au pas 2 (PREP) — motif "Si chaînage automatique de
    l'opération ⇒ ..." documenté dans OPERATION.md §7. Chaque redémarrage
    revient exactement à la valeur de départ de l'opération (200), donc
    `reconstruct_operations` les détecte sans aucune configuration.

        200(pas2) 220(pas2) 300(pas3)   -> opération 0 (première ligne du jeu)
        200(pas2) 300(pas3)             -> opération 1 (redémarrage chaîné)
        200(pas2) 250(pas2) 300(pas3)   -> opération 2 (redémarrage chaîné)
    """
    from att_analysis import label_with_pas

    reperes = [200, 220, 300, 200, 300, 200, 250, 300]
    n = len(reperes)
    starts = pd.date_range("2026-01-01", periods=n, freq="10min", tz="UTC")
    ends = starts + pd.Timedelta(minutes=10)
    steps = pd.DataFrame({
        "step_id": range(n),
        "repere": reperes,
        "start": starts,
        "end": ends,
        "duration": ends - starts,
        "duration_min": 10.0,
        "n_samples": 10,
    })
    return label_with_pas(steps, att_pas_reference)


@pytest.fixture
def att_labeled_steps_rare_low_restart(att_pas_reference):
    """
    9 pas simulant 4 opérations : 3 redémarrages chaînés au pas 2 (repère
    200, le cas dominant) et 1 redémarrage rare mais légitime au pas 1
    (repère 100, ex: une relance manuelle) — reproduit le cas qui a fait
    échouer une 1ère version de `find_restart_reference` (un simple
    "médiane des valeurs distinctes" isole le 100 tout seul dans le
    "cluster bas" et l'utilise à tort comme référence, bloquant alors la
    reconnaissance des redémarrages, bien plus fréquents, au pas 2 -- cf.
    OPERATION.md §7). Le filtre par *position* (pas_num tôt dans la
    séquence documentée) garde les deux dans le même pool de candidats.

        200(pas2) 300(pas3)             -> opération 0
        200(pas2) 300(pas3)             -> opération 1 (chaîné)
        100(pas1) 200(pas2) 300(pas3)   -> opération 2 (relance rare)
        200(pas2) 300(pas3)             -> opération 3 (chaîné)
    """
    from att_analysis import label_with_pas

    reperes = [200, 300, 200, 300, 100, 200, 300, 200, 300]
    n = len(reperes)
    starts = pd.date_range("2026-01-01", periods=n, freq="10min", tz="UTC")
    ends = starts + pd.Timedelta(minutes=10)
    steps = pd.DataFrame({
        "step_id": range(n),
        "repere": reperes,
        "start": starts,
        "end": ends,
        "duration": ends - starts,
        "duration_min": 10.0,
        "n_samples": 10,
    })
    return label_with_pas(steps, att_pas_reference)


@pytest.fixture
def att_pas_reference_with_late_loop():
    """
    Table pas -> plage de repère à 5 pas réels, sur le modèle d'`OP3120`
    (boucle "plusieurs filtrations") : pas 4/5 forment une boucle interne
    documentée (`DEGAZ` reboucle vers `FILTRATION`) plus fréquente que les
    vrais redémarrages, mais toujours *tardive* dans la séquence.
    """
    return pd.DataFrame([
        {"pas_num": 1, "phase": "ATTENTE", "code_court": "ATTENTE", "description": "Attente",
         "att_min": 100, "att_max": 110,
         "type_attente": "OPERATEUR", "type_attente_detail": "Confirmation de lancement",
         "operation_liee": np.nan, "sens_liaison": np.nan},
        {"pas_num": 2, "phase": "PREP", "code_court": "PREP", "description": "Preparation",
         "att_min": 200, "att_max": 250,
         "type_attente": "PROCESS", "type_attente_detail": "Verification avant demarrage",
         "operation_liee": np.nan, "sens_liaison": np.nan},
        {"pas_num": 3, "phase": "CHARGE", "code_court": "CHARGE", "description": "Chargement",
         "att_min": 300, "att_max": 350,
         "type_attente": "REMPLISSAGE", "type_attente_detail": "Chargement",
         "operation_liee": np.nan, "sens_liaison": np.nan},
        {"pas_num": 4, "phase": "FILTRATION", "code_court": "FILTRATION", "description": "Filtration",
         "att_min": 400, "att_max": 450,
         "type_attente": "PROCESS", "type_attente_detail": "Filtration",
         "operation_liee": np.nan, "sens_liaison": np.nan},
        {"pas_num": 5, "phase": "DEGAZ", "code_court": "DEGAZ", "description": "Degazage",
         "att_min": 500, "att_max": 550,
         "type_attente": "PROCESS", "type_attente_detail": "Degazage, reboucle vers FILTRATION si nb filtrations > 1",
         "operation_liee": np.nan, "sens_liaison": np.nan},
    ])


@pytest.fixture
def att_labeled_steps_dominant_internal_loop(att_pas_reference_with_late_loop):
    """
    16 pas simulant 2 opérations, chacune avec une boucle interne
    "filtration" répétée (retour DEGAZ -> FILTRATION, pas 5 -> pas 4) plus
    fréquente (3 occurrences) que les vrais redémarrages (1 occurrence, au
    pas 1) -- reproduit `OP3120` (boucle ~50% des régressions, largement
    devant le pas de lancement). Le filtre par position doit exclure la
    boucle (pas 4, tardif) du calcul de référence quelle que soit sa
    fréquence, et ne garder que le pas 1 (précoce) comme point d'entrée.

        100(pas1) 200(pas2) 300(pas3) 400(pas4) 500(pas5)
        400(pas4, boucle) 500(pas5) 400(pas4, boucle) 500(pas5)  -> opération 0
        100(pas1, redémarrage) 200(pas2) 300(pas3) 400(pas4) 500(pas5)
        400(pas4, boucle) 500(pas5)                                -> opération 1
    """
    from att_analysis import label_with_pas

    reperes = [
        100, 200, 300, 400, 500, 400, 500, 400, 500,
        100, 200, 300, 400, 500, 400, 500,
    ]
    n = len(reperes)
    starts = pd.date_range("2026-01-01", periods=n, freq="10min", tz="UTC")
    ends = starts + pd.Timedelta(minutes=10)
    steps = pd.DataFrame({
        "step_id": range(n),
        "repere": reperes,
        "start": starts,
        "end": ends,
        "duration": ends - starts,
        "duration_min": 10.0,
        "n_samples": 10,
    })
    return label_with_pas(steps, att_pas_reference_with_late_loop)


@pytest.fixture
def att_pas_reference_long_sequence():
    """
    Table pas -> plage de repère à 21 pas réels (pas_num 1..21, repère
    i*100..i*100+90), sur le modèle d'une opération longue (ex: OP1610,
    24 pas au total) -- sert à reproduire un saut de pas_num important
    (pas 21 -> pas 4) sans qu'un fixture à 3-5 pas ne rende la notion de
    "grand saut" dégénérée (cf. `att_labeled_steps_large_jump_restart`).
    """
    return pd.DataFrame([
        {"pas_num": i, "phase": f"PHASE{i}", "code_court": f"CODE{i}", "description": f"Pas {i}",
         "att_min": i * 100, "att_max": i * 100 + 90,
         "type_attente": "PROCESS", "type_attente_detail": f"Attente pas {i}",
         "operation_liee": np.nan, "sens_liaison": np.nan}
        for i in range(1, 22)
    ])


@pytest.fixture
def att_labeled_steps_large_jump_restart(att_pas_reference_long_sequence):
    """
    7 pas simulant 4 opérations, dont une où le redémarrage chaîné atterrit
    au pas 4 (repère 400) -- une valeur trop éloignée de la référence
    dominante (pas 1, repère 100) pour passer le test de valeur seul, mais
    dont le saut en pas_num (pas 21 -> pas 4, soit 17 pas en arrière sur une
    étendue réelle de 20) est bien trop grand pour être un défaut plausible
    (un défaut ne peut renvoyer que vers un pas proche de celui qui vient
    d'échouer) -- reproduit le cas signalé : un retour pas21->pas4 classé à
    tort en défaut par le seul critère de valeur.

        100(pas1) 2100(pas21)                       -> opération 0
        100(pas1, redémarrage) 2100(pas21) 400(pas4, GRAND SAUT) 500(pas5) -> opérations 1 et 2
        100(pas1, redémarrage)                       -> opération 3
    """
    from att_analysis import label_with_pas

    reperes = [100, 2100, 100, 2100, 400, 500, 100]
    n = len(reperes)
    starts = pd.date_range("2026-01-01", periods=n, freq="10min", tz="UTC")
    ends = starts + pd.Timedelta(minutes=10)
    steps = pd.DataFrame({
        "step_id": range(n),
        "repere": reperes,
        "start": starts,
        "end": ends,
        "duration": ends - starts,
        "duration_min": 10.0,
        "n_samples": 10,
    })
    return label_with_pas(steps, att_pas_reference_long_sequence)


@pytest.fixture
def att_pas_reference_nine_steps():
    """Table pas -> plage de repère à 9 pas réels (pas_num 1..9, repère i*100)."""
    return pd.DataFrame([
        {"pas_num": i, "phase": f"PHASE{i}", "code_court": f"CODE{i}", "description": f"Pas {i}",
         "att_min": i * 100, "att_max": i * 100 + 90,
         "type_attente": "PROCESS", "type_attente_detail": f"Attente pas {i}",
         "operation_liee": np.nan, "sens_liaison": np.nan}
        for i in range(1, 10)
    ])


@pytest.fixture
def att_labeled_steps_fails_twice_then_succeeds(att_pas_reference_nine_steps):
    """
    15 pas simulant une seule opération avec un pas (pas 8, via la boucle
    pas6->pas7->pas8) qui échoue deux fois puis réussit à la 3e tentative
    -- reproduit `OP1410`/`EGOUTTAGE` (attente d'une autre opération qui
    échoue vite, retente, et n'aboutit qu'après une attente réelle plus
    longue) : les 2 premières tentatives (ratées) doivent être exclues du
    temps standard, la 3e (réussie) doit compter, même si elle ne fait que
    RATTRAPER (sans le dépasser) le niveau déjà atteint par les tentatives
    précédentes.

        1,2,3,4,5 (montée initiale propre)
        6,7,8 (tentative 1, échoue) -> 6,7,8 (tentative 2, échoue)
        -> 6,7,8 (tentative 3, réussit) -> 9 (final)
    """
    from att_analysis import label_with_pas

    pas_nums = [1, 2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 6, 7, 8, 9]
    reperes = [p * 100 for p in pas_nums]
    n = len(reperes)
    starts = pd.date_range("2026-01-01", periods=n, freq="10min", tz="UTC")
    ends = starts + pd.Timedelta(minutes=10)
    steps = pd.DataFrame({
        "step_id": range(n),
        "repere": reperes,
        "start": starts,
        "end": ends,
        "duration": ends - starts,
        "duration_min": 10.0,
        "n_samples": 10,
    })
    return label_with_pas(steps, att_pas_reference_nine_steps)


@pytest.fixture
def att_sample_index():
    return pd.date_range("2026-01-01", periods=23, freq="1min", tz="UTC")


@pytest.fixture
def att_sample_df(att_sample_index):
    values = [0] * 3 + [100] * 2 + [101] * 3 + [102] * 2 + [0] * 3 + [100] * 2 + [101] * 3 + [102] * 2 + [0] * 3
    return pd.DataFrame({"att": values}, index=att_sample_index)


@pytest.fixture
def att_sample_steps(att_sample_df):
    from att_analysis import extract_steps
    return extract_steps(att_sample_df, value_col="att")


@pytest.fixture
def att_sample_transitions(att_sample_steps):
    from att_analysis import transition_counts
    return transition_counts(att_sample_steps)
