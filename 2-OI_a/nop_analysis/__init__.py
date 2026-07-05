"""
Package nop_analysis : analyse de performance des opérations à partir d'un
compteur cumulatif value_nop.

Organisation :
    config.py         constantes (seuils, tailles de fenêtres, ...)
    utils.py          fonctions utilitaires génériques
    nop_analysis.py   fonctions métier (nettoyage, extraction, KPI)
    plots.py          toutes les visualisations
"""

from . import config
from .nop_analysis import (
    add_score,
    analyze_best_runs,
    build_sliding_windows,
    build_summary_report,
    clean_dataframe,
    compute_control_limits,
    compute_monthly_cv,
    compute_monthly_scores,
    compute_monthly_zscore,
    compute_zscore,
    duration_threshold_report,
    evaluate_golden_run,
    extract_operations,
    filter_operations_by_duration,
    flag_out_of_control,
    get_reference_duration,
    golden_run,
    golden_run_dashboard,
    short_operations,
)

__all__ = [
    "config",
    "clean_dataframe",
    "extract_operations",
    "filter_operations_by_duration",
    "analyze_best_runs",
    "get_reference_duration",
    "build_sliding_windows",
    "add_score",
    "compute_monthly_scores",
    "compute_zscore",
    "compute_monthly_zscore",
    "compute_monthly_cv",
    "golden_run",
    "golden_run_dashboard",
    "build_summary_report",
    "evaluate_golden_run",
    "duration_threshold_report",
    "short_operations",
    "compute_control_limits",
    "flag_out_of_control",
]
