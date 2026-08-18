"""
Cache disque pour OI_DataProcessor.

Évite de refaire l'appel réseau à OIA quand les données ont déjà été
récupérées une fois pour la même période/liste de tags : `save()` sérialise
l'état utile d'un processor après `merge()` dans un pickle compressé gzip,
`load()` le relit. Pas de nouvelle dépendance (gzip est dans la stdlib,
`pd.to_pickle`/`pd.read_pickle` savent le piloter directement) et le format
pickle préserve exactement dtypes/index — contrairement à un CSV.

Usage typique (notebook) :

    from tools.oi_cache import save, load, restore_into

    cache_path = "documents/confidentiel/cache/att_2026-06_2026-12.pkl.gz"
    cached = load(cache_path)
    if cached is not None:
        restore_into(processor, cached)
    else:
        processor.merge()
        save(processor, cache_path)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save(processor, path: str | Path) -> Path:
    """Sérialise df/data/unit_tags de `processor` dans un pickle gzip."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "df": processor.df,
        "data": processor.data,
        "unit_tags": processor.unit_tags,
        "rename_mapping": processor.rename_mapping,
        "agg_mapping": processor.agg_mapping,
        "start": processor.start,
        "end": processor.end,
        "interval": processor.interval,
    }
    pd.to_pickle(payload, path, compression="gzip")
    return path


def load(path: str | Path) -> dict | None:
    """Retourne le payload mis en cache, ou None si le fichier n'existe pas."""
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_pickle(path, compression="gzip")


def restore_into(processor, cached: dict) -> None:
    """Réinjecte un cache chargé par `load()` dans `processor` (pas de requête OIA)."""
    processor.df = cached["df"]
    processor.data = cached["data"]
    processor.unit_tags = cached["unit_tags"]
