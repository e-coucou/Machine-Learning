"""
Configuration partagée des tags Att suivis (un dict par PU/opération).

Charge `documents/confidentiel/att_operations_config.json` — le **catalogue
source** de toutes les opérations suivies (nom, description, tag, code
opération, produit), utilisé pour lancer aussi bien l'étape 1 (quels
`OPxxxx_pas_reference.csv` reste-t-il à construire ?) que l'étape 2 (quels
tags a-t-on les données pour reconstruire, cf.
`documents/confidentiel/OPERATION.md`). Modifier ce JSON, pas ce fichier,
pour ajouter/retirer une opération suivie.

Champs de chaque entrée (cf. le JSON pour le détail) :
    nom            colonne renommée par OI_DataProcessor (cf. rename_mapping)
    tag            référence du point OIAnalytics (data-reference), ex: "PU1410VA_Att"
    tag_           référence OIAnalytics de remplacement, ex: "PU1410VA" — utilisée
                   à la place de `tag` via `TAGS_` (cf. plus bas), le temps de la
                   bascule vers cette nouvelle référence
    description    libellé de l'opération (affichage)
    operation      code opération (doc process), ex: "OP1410"
    produit        produit fabriqué par cette opération, ex: "AOIP"
    repere         repère de lancement confirmé (cf. doc process, ex: 110 pour OP1410VA) —
                   ancre utilisée par att_analysis.cycle_times ; None tant que
                   l'étape 1 n'a pas été faite pour cette opération
    pas_reference  chemin vers la table process du pas (étape 1) ; None ou
                   fichier absent => le tag n'est exploré qu'empiriquement
                   (section 3 du notebook), sans reconstruction d'opérations
                   (section 4+) — utiliser `has_pas_reference()` pour tester
                   ça de façon fiable (un chemin renseigné ne garantit pas
                   que le fichier existe déjà).
    type           "batch" ou "continu" — un PU en marche continue n'a pas
                   de cycle discret pas1->pasN->redémarrage : le modèle de
                   reconstruction d'opérations (`att_analysis.reconstruct_operations`)
                   ne s'applique pas et produit des résultats sans sens sur
                   ce type de tag. Utiliser `is_batch()` pour filtrer les
                   tags à exclure de la reconstruction (section 4+ du
                   notebook, `build_att_references.py`) — un tag "continu"
                   reste exploré empiriquement (section 3) si besoin.
    temps_reference_min  temps standard minimum connu de l'exploitant, en
                   minutes (saisi à la main, cf. tableau des opérations) —
                   sert de référence pour valider le "temps standard —
                   minimum/médian" calculé par `reconstruct_operations` ;
                   `None` pour les opérations "continu" (non applicable).
    pas_attente    numéro du pas qui sert de "porte d'attente/lancement"
                   (temps à exclure du temps de l'opération), quand ce
                   n'est PAS le pas 1 documenté -- cas d'`OP2340`, dont le
                   chaînage saute toujours directement au pas 4 (les pas
                   1-3 ne sont jamais observés en pratique) : `pas_attente=4`
                   pour cette opération. Absent du catalogue => utiliser le
                   pas 1 (comportement par défaut).
"""

from __future__ import annotations

import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "documents" / "confidentiel" / "att_operations_config.json"


def _load_tags() -> list:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def _tags_view(tags: list, field: str) -> list:
    """
    Vue de `tags` où la clé "tag" de chaque entrée est remplacée par la
    valeur de `field`. Permet de brancher OI_DataProcessor (et tout code
    qui lit `tag_def["tag"]`, ex. build_att_references.py,
    build_operations_network.py) sur une autre référence OIAnalytics sans
    modifier ce code.
    """
    return [{**t, "tag": t[field]} for t in tags]


def has_pas_reference(tag_def: dict) -> bool:
    """
    True si l'étape 1 (analyse du document opératoire) a été faite pour ce
    tag : `pas_reference` renseigné **et** le fichier existe réellement sur
    disque (un chemin renseigné dans le catalogue documente une intention,
    pas forcément un fichier déjà construit).
    """
    ref_path = tag_def.get("pas_reference")
    return bool(ref_path) and Path(ref_path).exists()


def is_batch(tag_def: dict) -> bool:
    """
    True si le tag est une opération en mode "batch" (cycles discrets,
    pas1->pasN->redémarrage). Un PU en marche "continu" n'a pas cette
    structure — la reconstruction d'opérations ne doit pas lui être
    appliquée. Par défaut True (rétrocompatible) si `type` est absent du
    catalogue.
    """
    return tag_def.get("type", "batch") == "batch"


TAGS = _load_tags()

# Vue de TAGS lisant depuis OIA via le champ "tag_" plutôt que "tag" —
# import à privilégier dans les nouveaux scripts/notebooks ; TAGS reste
# inchangé pour ne pas casser le code existant.
TAGS_ = _tags_view(TAGS, "tag_")
