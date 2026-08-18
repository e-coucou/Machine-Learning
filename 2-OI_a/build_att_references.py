"""
Job batch (hors ligne, sans graphique) — étape 2 du flux en 2 temps décrit
dans `documents/confidentiel/OPERATION.md` :

    1. analyse du document opératoire OPxxxxVA.doc -> OPxxxx_pas_reference.csv
       (manuel/assisté par LLM, cf. OPERATION.md — une fois par PU)
    2. CE SCRIPT : charge les données réelles depuis OIAnalytics pour tous
       les tags de `att_tags_config.TAGS`, reconstruit les opérations et
       exporte un bundle de référence JSON **par opération** (statistiques,
       limites SPC, Pareto des défauts, enchaînements inter-opérations) —
       automatisable, à relancer périodiquement pour tenir les références
       à jour.

Le notebook `OI_ATT_v0.ipynb` couvre le même calcul de façon interactive
(avec les graphiques d'exploration) pour un tag à la fois ; ce script fait
tourner exactement le même pipeline (`att_analysis`) sur tous les tags d'un
coup, sans supervision.

Usage :
    python build_att_references.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.OI_class_OP import OI_DataProcessor
import att_analysis as atta
from att_tags_config import TAGS, has_pas_reference, is_batch

CHAINING_WINDOW = 5
OUTPUT_DIR = "documents/confidentiel"
URL_BASE = "https://oianalytics-100.optimistik.fr/api/oianalytics/time-values/query?"
START, END = "2023-01-01", "2026-12-31"


def process_tag(tag_def: dict, data: pd.DataFrame) -> dict:
    """
    Construit le bundle de référence pour un tag, à partir de la série Att
    brute déjà chargée (`data`, DataFrame indexé par DatetimeIndex, une
    colonne par tag renommée via `nom`).

    Fonction pure (aucun accès réseau) — isolée de `main()` pour rester
    testable sans OIAnalytics : le pipeline lui-même (nettoyage,
    reconstruction, temps de cycle, bundle de référence) est entièrement
    délégué à `att_analysis`, déjà couvert par ses propres tests unitaires.
    """
    nom = tag_def["nom"]
    anchor = tag_def["repere"]

    pas_reference = atta.load_pas_reference(tag_def["pas_reference"])
    cleaned = atta.clean_dataframe(data[[nom]], value_col=nom)
    steps = atta.extract_steps(cleaned, value_col=nom)
    labeled = atta.label_with_pas(steps, pas_reference)
    operations, labeled_steps = atta.reconstruct_operations(labeled, pas_reference)
    defauts = atta.defaut_frequency_table(labeled_steps)
    cycle_times_df = atta.cycle_times(labeled_steps, anchor_repere=anchor)

    return atta.build_reference(
        pas_reference, operations, labeled_steps, defauts, cycle_times_df,
        tag=tag_def["tag"], anchor_repere=anchor, chaining_window=CHAINING_WINDOW,
        description=tag_def.get("description", ""),
        operation=tag_def.get("operation", ""),
        produit=tag_def.get("produit", ""),
    )


def main(tags: list = TAGS, output_dir: str = OUTPUT_DIR) -> None:
    processor = OI_DataProcessor(
        url_base=URL_BASE,
        start=START,
        end=END,
        tags_selected=tags,
        tags_other=[],
        interval="PT01M",
        verbose=True,
        agg="FIRST",
    )
    processor.merge()

    for tag_def in tags:
        label, nom = tag_def["tag"], tag_def["nom"]

        if nom not in processor.data.columns:
            print(f"[SKIP] {label} : donnée absente de processor.data")
            continue
        if not is_batch(tag_def):
            print(f"[SKIP] {label} : type 'continu' (pas de cycle discret, reconstruction non applicable)")
            continue
        if not has_pas_reference(tag_def):
            print(f"[SKIP] {label} : pas de table de référence (étape 1 non faite ou fichier absent)")
            continue

        print(f"[RUN]  {label} ...")
        try:
            reference = process_tag(tag_def, processor.data)
        except Exception as exc:  # un tag en échec ne doit pas bloquer les autres
            print(f"[FAIL] {label} : {exc}")
            continue

        out_path = Path(output_dir) / f"{label}_reference.json"
        atta.save_reference(reference, str(out_path))
        print(f"[OK]   {label} -> {out_path} ({reference['n_operations']} opérations)")


if __name__ == "__main__":
    main()
