import tools.ai_graph as g
import argparse

def main():
    # 1. Configuration des arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Poussin-v2 Graph Plotter")
    
    parser.add_argument("--ymin", type=float, default=2.6, help="Limite basse de la Loss")
    parser.add_argument("--ymax", type=float, default=3.3, help="Limite haute de la Loss")
    parser.add_argument("--xmax", type=int, default=20000, help="Limite max de steps (axe X)")
    parser.add_argument("--smooth", type=int, default=5, help="Facteur de lissage des courbes")
    parser.add_argument("--store", action="store_true", help="Activer la sauvegarde Image")
    parser.add_argument("--compare", type=int, default=None, help="Comparaison des courbes en un point")
    parser.add_argument("--target", type=float, default=None, help="Valeur Cible de Loss")
    parser.add_argument("--speed", type=float, default=None, help="Step par seconde")
    parser.add_argument("--raw", type=bool, default=False, help="Raw data")
    parser.add_argument("--titre", type=str, default="Training Mac-M1", help="Titre du graphique")
    
    # Chemins des fichiers (avec tes valeurs actuelles par défaut)
    parser.add_argument("--log1", type=str, default="model/my_wiky_history.json")
    parser.add_argument("--log2", type=str, default="save/model_3/my_wiky_history.json")
    parser.add_argument("--log3", type=str, default="save/model_1/my_wiky_history.json")

    args = parser.parse_args()

    # 2. Définition de tes événements (Historique du projet)
    mes_evenements = [
        {"step": 2000, "label": "End warmup", "color": "gray", "lw": 0.7},
        {"step": 7200, "label": "Culturax Mix à 0.5 / batch 8x16", "color": "blue", "lw": 1.2},
        {"step": 13700, "label": "Cible 2.8 - Atteinte", "color": "gray", "lw": 1.},
        {"step": 18700, "label": "Cible 2.7 - Atteinte", "color": "gray", "lw": 1.},
        {"step": 19600, "label": "CulturaX Mix à 0.4", "color": "blue", "lw": 1.2},
        {"step": 32500, "label": "Cible 2.6", "color": "magenta", "lw": 1.5},
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
        raw=args.raw,
        titre=args.titre,
    )

if __name__ == "__main__":
    main()
