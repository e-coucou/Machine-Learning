"""
Package att_analysis : exploration des repères d'attente/process (Att)
échantillonnés à la minute sur les PU.

Organisation (miroir de nop_analysis/) :
    config.py         constantes (valeur de repos, nombre de repères affichés, ...)
    att_analysis.py    fonctions métier (extraction des pas, fréquence, transitions)
    plots.py           visualisations
"""

from . import config
from .att_analysis import (
    best_chaining_window,
    build_operations_network,
    build_reference,
    candidate_end_reperes,
    candidate_start_reperes,
    clean_dataframe,
    compare_operation_to_reference,
    cycle_times,
    defaut_frequency_table,
    exclude_boundary_operations,
    extract_enchainements,
    extract_steps,
    find_restart_reference,
    label_with_pas,
    load_pas_reference,
    load_reference,
    pas_duration_stats,
    reference_cycle_time,
    reconstruct_operations,
    repere_frequency_table,
    save_reference,
    transition_counts,
)

__all__ = [
    "config",
    "clean_dataframe",
    "extract_steps",
    "repere_frequency_table",
    "transition_counts",
    "candidate_start_reperes",
    "candidate_end_reperes",
    "load_pas_reference",
    "label_with_pas",
    "find_restart_reference",
    "reconstruct_operations",
    "exclude_boundary_operations",
    "defaut_frequency_table",
    "extract_enchainements",
    "cycle_times",
    "best_chaining_window",
    "pas_duration_stats",
    "reference_cycle_time",
    "build_reference",
    "save_reference",
    "load_reference",
    "compare_operation_to_reference",
    "build_operations_network",
]
