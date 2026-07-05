"""
Fixtures partagées pour les tests unitaires de nop_analysis.

Le jeu de données `sample_df` / `sample_ops` est construit à la main avec
des durées connues à l'avance, pour pouvoir vérifier des valeurs exactes
plutôt que de simples propriétés générales (formes, types...).

Plateaux (5 opérations, 30 échantillons à la minute) :
    value=100 : minutes  0- 4  (5 échantillons)  -> durée 5 min
    value=101 : minutes  5- 9  (5 échantillons)  -> durée 5 min
    value=102 : minutes 10-19  (10 échantillons) -> durée 10 min
    value=103 : minutes 20-24  (5 échantillons)  -> durée 5 min
    value=104 : minutes 25-29  (5 échantillons)  -> durée 4 min (dernière
                opération : sa "fin" est le dernier timestamp du jeu de
                données, pas le début d'une opération suivante)
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_index():
    return pd.date_range("2026-01-01", periods=30, freq="1min", tz="UTC")


@pytest.fixture
def sample_df(sample_index):
    values = [100] * 5 + [101] * 5 + [102] * 10 + [103] * 5 + [104] * 5
    return pd.DataFrame({"value_nop": values}, index=sample_index)


@pytest.fixture
def sample_df_with_noise(sample_index):
    """Même jeu de données, avec un NaN et une valeur négative injectés."""
    values = [100] * 5 + [101] * 5 + [102] * 10 + [103] * 5 + [104] * 5
    values = [float(v) for v in values]
    values[2] = np.nan
    values[7] = -1.0
    return pd.DataFrame({"value_nop": values}, index=sample_index)


@pytest.fixture
def sample_ops(sample_df):
    from nop_analysis import extract_operations
    return extract_operations(sample_df)


@pytest.fixture
def larger_ops():
    """
    Jeu de données plus volumineux (100 opérations, durées aléatoires mais
    graine fixée) pour les tests nécessitant plusieurs mois de données
    (scoring mensuel, golden run sur fenêtre 35, etc.).
    """
    from nop_analysis import extract_operations

    rng = np.random.default_rng(42)
    n_ops = 100
    durations_min = rng.gamma(shape=4, scale=90, size=n_ops)  # ~6h en moyenne

    start = pd.Timestamp("2026-01-01", tz="UTC")
    starts = [start]
    for d in durations_min[:-1]:
        starts.append(starts[-1] + pd.Timedelta(minutes=float(d)))

    rows = []
    for i, s in enumerate(starts):
        end = s + pd.Timedelta(minutes=float(durations_min[i]))
        idx = pd.date_range(s, end, freq="1min", inclusive="left", tz="UTC")
        rows.append(pd.Series(1774 + i, index=idx))

    df = pd.concat(rows).to_frame("value_nop")
    return extract_operations(df)
