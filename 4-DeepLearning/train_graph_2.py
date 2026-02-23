import tools.ai_graph as g
import argparse

def main():
    # 1. Configuration des arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Poussin-v2 Graph Plotter")
    
    parser.add_argument("--ymin", type=float, default=2.6, help="Limite basse de la Loss")
    parser.add_argument("--ymax", type=float, default=3.3, help="Limite haute de la Loss")
    parser.add_argument("--xmax", type=int, default=20000, help="Limite max de steps (axe X)")
    parser.add_argument("--xmin", type=int, default=0, help="Limite basse de steps (axe X)")
    parser.add_argument("--smooth", type=int, default=5, help="Facteur de lissage des courbes")
    parser.add_argument("--horizon", type=int, default=30000, help="Horizon pour le calcul de la cible Loss")
    parser.add_argument("--proj", type=int, default=10, help="Pour le calcul de la projection masque premiers step")
    parser.add_argument("--store", action="store_true", help="Activer la sauvegarde Image")
    parser.add_argument("--compare", type=int, default=None, help="Comparaison des courbes en un point")
    parser.add_argument("--target", type=float, default=None, help="Valeur Cible de Loss")
    parser.add_argument("--speed", type=float, default=None, help="Step par seconde")
    parser.add_argument("--raw", type=bool, default=False, help="Raw data")
    parser.add_argument("--titre", type=str, default="Training Mac-M1", help="Titre du graphique")
    
    # Chemins des fichiers (avec tes valeurs actuelles par défaut)
    parser.add_argument("--log1", type=str, default="model/my_wiky_history.json")
    parser.add_argument("--log2", type=str, default="save/model_4/my_wiky_history.json")
    parser.add_argument("--log3", type=str, default="save/model_3/my_wiky_history.json")

    args = parser.parse_args()

    # 2. Définition de tes événements (Historique du projet)
    mes_evenements = [
        {"step": 2000, "label": "End warmup", "color": "gray", "lw": 0.7},
        {"step": 3400, "label": "Introduction de CulturaX à 10%", "color": "gray", "lw": 1.},
        {"step": 4200, "label": "CulturaX à 20%", "color": "gray", "lw": 1.},
        {"step": 5000, "label": "CulturaX à 35%", "color": "gray", "lw": 1.},
        {"step": 6000, "label": "Introduction de 10% de Littéraire (35/55)", "color": "gray", "lw": 1.},
        {"step": 8100, "label": "Passage EBS à 384 ( 8 x 48)", "color": "gray", "lw": 1.},
        {"step": 10200, "label": "Mixte 60/35/5", "color": "gray", "lw": 1.},
        {"step": 12300, "label": "Mixte 80/15/5", "color": "gray", "lw": 1.},
#        {"step": 7200, "label": "Culturax Mix 50/50 / batch 8x16", "color": "magenta", "lw": 1.},
        {"step": 13700, "label": "Palier 2.8 - Model v4", "color": "orange", "lw": 1.},
        {"step": 18700, "label": "Palier 2.7 - Atteint", "color": "orange", "lw": 1.},
#        {"step": 19600, "label": "CulturaX Mix 40/60", "color": "magenta", "lw": 1.},
#        {"step": 25400, "label": "CulturaX Mix 35/65", "color": "magenta", "lw": 1.},
        {"step": 27200, "label": "Palier 2.6 - Atteint", "color": "orange", "lw": 1.},
#        {"step": 34800, "label": "CulturaX Mix 30/70", "color": "magenta", "lw": 1.},
#        {"step": 39800, "label": "CulturaX Mix 35/75", "color": "magenta", "lw": 1.},
#        {"step": 44800, "label": "Litteraire Mix 50/30/20", "color": "blue", "lw": 1.},
        {"step": 45000, "label": "Palier 2.5 - Atteint", "color": "orange", "lw": 1.},
#        {"step": 46200, "label": "Litteraire Mix 20/30/50", "color": "blue", "lw": 1.},
#        {"step": 48600, "label": "Litteraire Mix 05/30/65 + wiki Clean", "color": "blue", "lw": 1.},
 #       {"step": 53400, "label": "EBS=192, Dropout 0.10", "color": "red", "lw": 1.},
        {"step": 60000, "label": "Cible 2.45", "color": "red", "lw": 1.5},
    ]

    # 3. Appel de la fonction de graphisme
    print(f"📊 Génération du graphique : {args.log1} ...")
    
    g.plot_poussin_gap(
        y_min=args.ymin,
        y_max=args.ymax,
        x_max=args.xmax,
        x_min=args.xmin,
        smooth=args.smooth,
        log_path=args.log1,
        second=args.log2,
        third=args.log3,
        compare=args.compare,
        target=args.target,
        speed=args.speed,
        annot_event=mes_evenements,
        raw=args.raw,
        titre=args.titre,
        horizon=args.horizon,
        proj=args.proj
    )

if __name__ == "__main__":
    main()
