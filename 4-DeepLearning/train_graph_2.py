import tools.ai_graph as g
import argparse

def main():
    # 1. Configuration des arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Poussin-v2 Graph Plotter")
    
    parser.add_argument("--ymin", type=float, default=2.6, help="Limite basse de la Loss")
    parser.add_argument("--ymax", type=float, default=3.3, help="Limite haute de la Loss")
    parser.add_argument("--xmax", type=int, default=20000, help="Limite max de steps (axe X)")
    parser.add_argument("--smooth", type=int, default=5, help="Facteur de lissage des courbes")
    parser.add_argument("--strore", action="store_true", help="Activer la sauvegarde Image")
    parser.add_argument("--compare", type=int, default=None, help="Comparaison des courbes en un point")
    parser.add_argument("--target", type=float, default=None, help="Valeur Cible de Loss")
    parser.add_argument("--speed", type=float, default=None, help="Step par seconde")
    
    # Chemins des fichiers (avec tes valeurs actuelles par défaut)
    parser.add_argument("--log1", type=str, default="model/my_wiky_history.json")
    parser.add_argument("--log2", type=str, default="save/model_3/my_wiky_history.json")
    parser.add_argument("--log3", type=str, default="save/model_1/my_wiky_history.json")

    args = parser.parse_args()

    # 2. Définition de tes événements (Historique du projet)
    mes_evenements = [
        {"step": 2000, "label": "End warmup", "color": "gray", "lw": 0.7},
        {"step": 7200, "label": "Culturax Mix 0.5 / batch 8x16", "color": "blue", "lw": 1.2},
        {"step": 12000, "label": "Cible 2.8", "color": "gray", "lw": 0.7},
        {"step": 21500, "label": "Cible 2.7", "color": "cyan", "lw": 1.7},
        {"step": 35000, "label": "Cible 2.6", "color": "cyan", "lw": 1.7},
#        {"step": 19500, "label": "Dropout 0.15", "color": "gray", "lw": 0.7},
#        {"step": 39000, "label": "LR_Decay Phase", "color": "orange", "lw": 0.7},
#        {"step": 46589, "label": "Epoch 2", "color": "red", "lw": 1.1},
    ]

    # 3. Appel de la fonction de graphisme
    print(f"📊 Génération du graphique : {args.log1} ...")
    
    g.plot_poussin_gap(
        y_min=args.ymin,
        y_max=args.ymax,
        x_max=args.xmax,
        smooth=args.smooth,
        log_path=args.log1,
        second=args.log2,
        third=args.log3,
        compare=args.compare,
        target=args.target,
        speed=args.speed,
        annot_event=mes_evenements,
    )

if __name__ == "__main__":
    main()
