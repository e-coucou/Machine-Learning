"""
Construit `documents/confidentiel/operations_network.json` : le réseau des
liaisons inter-opérations (nodes/edges), à partir des tables process déjà
construites (étape 1, cf. `documents/confidentiel/OPERATION.md`).

Sert de base à deux visualisations :
    - un **graphe de dépendances** ("network") : nodes = opérations, edges =
      liaisons (`operation_source` -> `operation_cible`, avec `sens` :
      ATTEND/ENVOIE/BIDIRECTIONNEL) ;
    - un **chronogramme** : chaque opération porte sa `sequence_produit`
      (position dans l'enchaînement du produit, telle que listée dans
      `att_operations_config.json`) et, si son `{tag}_reference.json` existe
      déjà (étape 2 faite), ses durées historiques (moyenne/médiane) et son
      nombre d'opérations observées.

N'accède à aucune donnée OIAnalytics — tourne uniquement sur les
`OPxxxx_pas_reference.csv` et les `{tag}_reference.json` déjà présents sur
disque. À relancer après chaque nouvelle table process construite (étape 1)
ou chaque nouveau `{tag}_reference.json` généré (étape 2).

Usage :
    python build_operations_network.py
"""

from __future__ import annotations

import json
from pathlib import Path

import att_analysis as atta
from att_tags_config import TAGS, has_pas_reference

OUTPUT_PATH = "documents/confidentiel/operations_network.json"


def _sequence_positions(tags: list) -> dict:
    """Position (1-indexée) de chaque opération dans son groupe produit, dans l'ordre du catalogue."""
    positions = {}
    counters = {}
    for t in tags:
        produit = t.get("produit")
        counters[produit] = counters.get(produit, 0) + 1
        positions[t["operation"]] = counters[produit]
    return positions


def main(tags: list = TAGS, output_path: str = OUTPUT_PATH) -> dict:
    entries = []
    for tag_def in tags:
        if not has_pas_reference(tag_def):
            continue
        entries.append({
            "operation": tag_def["operation"],
            "nom": tag_def["nom"],
            "tag": tag_def["tag"],
            "produit": tag_def.get("produit"),
            "description": tag_def.get("description", ""),
            "pas_reference": atta.load_pas_reference(tag_def["pas_reference"]),
        })

    network = atta.build_operations_network(entries)

    sequence_positions = _sequence_positions(tags)
    tag_by_operation = {t["operation"]: t for t in tags}

    for op_entry in network["operations"]:
        op = op_entry["operation"]
        op_entry["sequence_produit"] = sequence_positions.get(op)

        tag_def = tag_by_operation.get(op)
        ref_json_path = Path("documents/confidentiel") / f"{tag_def['tag']}_reference.json" if tag_def else None
        if ref_json_path is not None and ref_json_path.exists():
            ref = atta.load_reference(str(ref_json_path))
            op_entry["duree_mediane_min"] = ref["duration_min"]["median"]
            op_entry["duree_moyenne_min"] = ref["duration_min"]["mean"]
            op_entry["n_operations_historique"] = ref["n_operations"]
        else:
            op_entry["duree_mediane_min"] = None
            op_entry["duree_moyenne_min"] = None
            op_entry["n_operations_historique"] = None

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(network, f, indent=2, ensure_ascii=False)

    print(f"[OK] {len(network['operations'])} operations, {len(network['liaisons'])} liaisons -> {output_path}")
    return network


if __name__ == "__main__":
    main()
