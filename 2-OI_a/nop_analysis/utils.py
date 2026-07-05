"""
Fonctions utilitaires génériques : validation d'entrées et primitives
vectorisées réutilisées par plusieurs fonctions métier de nop_analysis.py.

Aucune logique métier ici — uniquement des briques génériques.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def ensure_datetime_index(df: pd.DataFrame) -> None:
    """Vérifie que df est indexé par un DatetimeIndex, lève TypeError sinon."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"Le DataFrame doit être indexé par un DatetimeIndex, reçu : {type(df.index)}"
        )


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Vérifie que toutes les colonnes attendues sont présentes, lève KeyError sinon."""
    missing = set(columns) - set(df.columns)
    if missing:
        raise KeyError(f"Colonnes manquantes dans le DataFrame : {sorted(missing)}")


def rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    """
    Somme glissante vectorisée (remplace une boucle Python + slicing manuel).

    Retourne un tableau vide si `window` dépasse la taille de `values`,
    plutôt que de lever une exception, pour permettre aux appelants de
    tester plusieurs tailles de fenêtre dans une boucle sans se soucier
    des cas où il n'y a pas assez de données.
    """
    if window <= 0:
        raise ValueError("window doit être strictement positif")
    if window > len(values):
        return np.array([])
    return np.convolve(values, np.ones(window), mode="valid")


def zscore(values: np.ndarray) -> np.ndarray:
    """Z-score standard ; retourne des zéros si l'écart-type est nul (évite division par 0)."""
    mu = values.mean()
    sigma = values.std()
    if sigma == 0:
        return np.zeros_like(values, dtype=float)
    return (values - mu) / sigma


def to_business_unit(value: float, factor: float) -> float:
    """Convertit une valeur exprimée en 'opérations' vers une unité métier (ex: tonnes)."""
    return value * factor
