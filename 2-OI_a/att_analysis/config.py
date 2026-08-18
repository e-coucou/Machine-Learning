"""
Constantes de configuration pour l'analyse des repères Att (attente/process).
"""

# Valeurs de repère non significatives, à exclure avant toute analyse.
# Confirmé par la doc process (OP1410.DOC) : Att=0 ne correspond à aucun pas
# du séquenceur — ce n'est pas un état de repos réel, c'est un signal
# invalide (capteur au repos / absence de lecture), à ne pas interpréter.
INVALID_REPERE_VALUES: tuple = (0,)

# Valeur de repère utilisée par défaut par les fonctions candidate_start/end
# (utilitaires génériques, pour un tag sans table de référence process — cf.
# reconstruct_operations pour l'approche privilégiée quand la doc process
# est disponible). À ne PAS confondre avec un état de repos réel.
IDLE_VALUE: int = 0

# Marge de tolérance appliquée à la valeur de repère de départ d'une
# opération pour détecter un nouveau départ (cf. reconstruct_operations) :
# une régression du numéro de pas est un nouveau départ si sa valeur
# d'arrivée est <= valeur de départ de l'opération en cours * ce facteur
# (1.1 = 10% de marge, pour absorber le bruit d'échantillonnage à la
# minute), un défaut du pas précédent sinon.
RESTART_TOLERANCE: float = 1.1

# Nombre de repères affichés sur le Pareto / la heatmap de transition
# (au-delà, le graphe devient illisible et les repères rares n'apportent
# rien à la lecture).
TOP_N_REPERES_DISPLAY: int = 40
