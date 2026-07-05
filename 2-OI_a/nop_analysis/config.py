"""
Constantes de configuration pour l'analyse des opérations NOP.

Centraliser ces valeurs ici évite les "nombres magiques" dispersés dans le
notebook (4, 24, 35, 15, 2.71, ...) et permet de changer un seuil à un seul
endroit.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Données source
# ---------------------------------------------------------------------------
VALUE_COLUMN: str = "value_nop"

# ---------------------------------------------------------------------------
# Filtrage des opérations
# ---------------------------------------------------------------------------
# Une opération dont la durée sort de cet intervalle est considérée comme
# non représentative (typiquement : arrêt de ligne prolongé au-delà de 24h).
# Le seuil bas est fixé à 3h par défaut (filtrage des opérations courtes) ;
# ajuster MIN_OPERATION_DURATION si un bruit capteur spécifique
# doit être exclu.
MIN_OPERATION_DURATION: pd.Timedelta = pd.Timedelta(hours=2)
MAX_OPERATION_DURATION: pd.Timedelta = pd.Timedelta(hours=24)

# ---------------------------------------------------------------------------
# Analyse des meilleures séries (runs)
# ---------------------------------------------------------------------------
# Tailles de fenêtres (nombre d'opérations consécutives) analysées par défaut.
DEFAULT_WINDOW_SIZES: tuple[int, ...] = (9, 15, 25, 35, 50)

# Fenêtre de référence retenue pour le "golden run" et pour le scoring
# mensuel lorsque aucune fenêtre n'est précisée explicitement.
GOLDEN_RUN_WINDOW: int = 35

# ---------------------------------------------------------------------------
# Suivi mensuel
# ---------------------------------------------------------------------------
# Taille de la fenêtre glissante utilisée pour le score / z-score mensuel.
MONTHLY_ROLLING_WINDOW: int = 10

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
# Nombre maximal de fenêtres affichées sur la heatmap globale (lisibilité).
GLOBAL_HEATMAP_MAX_WINDOWS: int = 300

# ---------------------------------------------------------------------------
# Conversion métier
# ---------------------------------------------------------------------------
# Facteur de conversion "opération -> unité métier" (ex: tonnes/opération).
# À ajuster selon le procédé réel.
OP_UNIT_FACTOR: float = 2.71
