"""
Construit `documents/confidentiel/operations_graph.json` : export dénormalisé
et prêt à afficher de `operations_network.json`, pour reconstruire le graphe
des opérations dans une application web/mobile (React/Next) sans redépendre
de la pipeline Python.

Combine :
    - `operations_network.json` (nodes + liaisons, cf. `build_operations_network.py`)
    - `att_operations_config.json` (type batch/continu)
    - `documents/confidentiel/recap_operations.csv` (nb d'opérations, durées,
      ratio vs référence — produit par `OI_ATT_v3.ipynb` section 5, absent
      tant que le notebook n'a pas tourné : les champs stats restent `null`)

Le schéma de sortie et sa documentation de consommation sont dans
`documents/confidentiel/operations_graph.md` — à relire avant d'intégrer ce
fichier côté frontend.

Usage :
    python build_operations_graph_export.py
"""

from __future__ import annotations

import json
import csv
from pathlib import Path

NETWORK_PATH = "documents/confidentiel/operations_network.json"
CONFIG_PATH = "documents/confidentiel/att_operations_config.json"
RECAP_PATH = "documents/confidentiel/recap_operations.csv"
OUTPUT_PATH = "documents/confidentiel/operations_graph.json"

SCHEMA_VERSION = "1.0.0"


def _load_recap(path: str) -> dict:
    """{nom -> stats}, ou {} si le récap n'a pas encore été exporté par le notebook."""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        return {
            row["tag"]: {
                "nb_operations": int(row["nb_ope"]),
                "duree_min_minutes": float(row["duree_minutes_min"]),
                "duree_mediane_minutes": float(row["duree_minutes_median"]),
                "duree_max_minutes": float(row["duree_minutes_max"]),
                "temps_reference_minutes": float(row["temps_reference_min"]),
                "ratio_mediane_reference": round(float(row["ratio_mediane_reference"]), 3),
            }
            for row in csv.DictReader(f)
        }


def main(
    network_path: str = NETWORK_PATH,
    config_path: str = CONFIG_PATH,
    recap_path: str = RECAP_PATH,
    output_path: str = OUTPUT_PATH,
) -> dict:
    with open(network_path, encoding="utf-8") as f:
        network = json.load(f)
    with open(config_path, encoding="utf-8") as f:
        catalog = json.load(f)

    type_by_code = {d["operation"]: d.get("type") for d in catalog}
    recap = _load_recap(recap_path)

    involved = set()
    neighbors = {}
    for l in network["liaisons"]:
        a, b = l["operation_source"], l["operation_cible"]
        involved.add(a)
        involved.add(b)
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    produit_by_code = {
        op["operation"]: op["produit"]
        for op in network["operations"]
        if op.get("in_catalog") and op.get("produit")
    }

    def infer_lane(code: str) -> str | None:
        """Ligne de produit à utiliser pour le regroupement visuel : la ligne
        catalogue si l'opération y est suivie, sinon la ligne la plus
        fréquente parmi ses voisins de graphe catalogués (ex: OP3935, hors
        catalogue, n'a que OP3110 pour voisin -> RHQ). Distinct de `produit`
        (qui reste `null` pour toute opération hors catalogue) : `lane` est
        une déduction pour l'affichage, pas une donnée déclarative."""
        if code in produit_by_code:
            return produit_by_code[code]
        neighbor_produits = [produit_by_code[n] for n in neighbors.get(code, []) if n in produit_by_code]
        if not neighbor_produits:
            return None
        return max(set(neighbor_produits), key=neighbor_produits.count)

    nodes = []
    for op in network["operations"]:
        code = op["operation"]
        stats = recap.get(op.get("nom")) if op.get("nom") else None
        nodes.append({
            "id": code,
            "tagOia": op.get("nom"),
            "produit": op.get("produit") if op.get("in_catalog") else None,
            "lane": infer_lane(code),
            "description": op.get("description") or None,
            "sequenceProduit": op.get("sequence_produit"),
            "type": type_by_code.get(code),
            "inCatalog": op.get("in_catalog", False),
            "isolated": code not in involved,
            "stats": stats,
        })

    edges = [
        {
            "source": l["operation_source"],
            "target": l["operation_cible"],
            "sens": l["sens"],
            "pas": l["pas_source"],
            "codeCourt": l["code_court_source"],
            "detail": l.get("detail"),
        }
        for l in network["liaisons"]
    ]

    produits = sorted({n["produit"] for n in nodes if n["produit"]})

    graph = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedFrom": {
            "network": network_path,
            "catalog": config_path,
            "recap": recap_path if recap else None,
        },
        "produits": produits,
        "nodes": nodes,
        "edges": edges,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    n_stats = sum(1 for n in nodes if n["stats"])
    print(f"[OK] {len(nodes)} nodes ({n_stats} avec stats), {len(edges)} edges -> {output_path}")
    return graph


if __name__ == "__main__":
    main()
