import matplotlib
# matplotlib.use('Agg') # pour éviter d'ouvri une fenetre
import time, json, os, argparse
import numpy as np
import matplotlib.pyplot as plt
import plotille

class TrainingVisualizer:
    def __init__(self, history_path='model/my_wiky_history.json'):
        self.history_path = history_path

    def plot_silent(self, window_size=10, lim_y=None, save_path='model/monitor.png'):
        data = self.load_data()
        if not data or 'train_loss' not in data:
            print("⌛ En attente de données valides...")
            return
        
        train_loss = data.get('train_loss', [])
        val_loss = data.get('val_loss', [])
        steps = data.get('steps', [])

        if len(train_loss) == 0: return

        # Création de la figure
        plt.figure(figsize=(12, 6))
        plt.style.use('bmh')

        if lim_y:
            plt.ylim(lim_y[0], lim_y[1])
        
        # 1. Tracé des données brutes
        plt.plot(steps, train_loss, color='blue', alpha=0.3, label='Train (brut)')
        plt.plot(steps, val_loss, color='red', alpha=0.3, label='Val (brut)')

        # 2. Lissage (Ta logique np.convolve)
        if len(train_loss) >= window_size:
            train_smooth = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
            val_smooth = np.convolve(val_loss, np.ones(window_size)/window_size, mode='valid')
            smooth_steps = np.array(steps)[window_size - 1:]
            
            plt.plot(smooth_steps, train_smooth, color='blue', linewidth=2, label=f'Train (smooth {window_size})')
            plt.plot(smooth_steps, val_smooth, color='red', linewidth=2, label=f'Val (smooth {window_size})')

        # Configuration du graphique
        plt.title(f'Évolution Loss - Step: {steps[-1]}', fontsize=14)
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Stats box
        stats_text = f"Dernier Train: {train_loss[-1]:.4f}\nDernier Val: {val_loss[-1]:.4f}"
        plt.annotate(stats_text, xy=(0.05, 0.05), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        # SAUVEGARDE SILENCIEUSE
        plt.savefig(save_path)
        plt.close() # Libère la mémoire
        print(f"📸 Graphique mis à jour à {time.strftime('%H:%M:%S')} -> {save_path}")

    def load_data(self):
        if not os.path.exists(self.history_path):
            return None
        try:
            with open(self.history_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception):
            return None

    def plot_live(self, window_size=10, lim_y=(4.0, 5.5)):
        plt.ion()  # Mode interactif ON
        fig = plt.figure(figsize=(12, 6))
        plt.style.use('bmh') # Style propre

        print(f"📈 Monitoring live sur {self.history_path}...")

        while True:
            data = self.load_data()
            if data and 'train_loss' in data and len(data['train_loss']) > 0:
                train_loss = data.get('train_loss', [])
                val_loss = data.get('val_loss', [])
                steps = data.get('steps', [])

                plt.clf() # Efface le graphique précédent
                
                # Gestion des axes
                if lim_y:
                    plt.ylim(lim_y[0], lim_y[1])

                # 1. Tracé des données brutes
                plt.plot(steps, train_loss, color='blue', alpha=0.3, label='Train (brut)')
                plt.plot(steps, val_loss, color='red', alpha=0.3, label='Val (brut)')

                # 2. Moyennes mobiles (Ton code NP)
                if len(train_loss) >= window_size:
                    train_smooth = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
                    val_smooth = np.convolve(val_loss, np.ones(window_size)/window_size, mode='valid')
                    steps_array = np.array(steps)
                    smooth_steps = steps_array[window_size - 1:]
                    
                    plt.plot(smooth_steps, train_smooth, color='blue', linewidth=2, label=f'Train (smooth {window_size})')
                    plt.plot(smooth_steps, val_smooth, color='red', linewidth=2, label=f'Val (smooth {window_size})')

                # Décoration
                plt.title(f'Monitoring Live - Step: {steps[-1]}', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend(loc='upper right')
                
                # Stats box
                stats_text = f"Step: {steps[-1]}\nTrain Loss: {train_loss[-1]:.4f}\nVal Loss: {val_loss[-1]:.4f}"
                plt.annotate(stats_text, xy=(0.02, 0.05), xycoords='axes fraction', 
                             bbox=dict(boxstyle="round", fc="white", alpha=0.8))

                plt.draw()
                plt.pause(60) # Rafraîchissement toutes les minutes
            else:
                print("⌛ En attente de données valides...")
                time.sleep(10)
    
    def update_image_only(self):
        # Au lieu de plt.show(), tu fais :
        plt.savefig('model/live_progress.png')
        plt.close()
        print(f"📸 Image mise à jour à {time.strftime('%H:%M:%S')}")

def terminal_monitor(vMin = 3.5, vMax=5.5):
    history_path = 'model/my_wiky_history.json'
    print("⌨️ Monitoring Terminal démarré...")
    yMin = vMin
    yMax = vMax
    while True:
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    data = json.load(f)
                
                steps = data.get('steps', [])
                val_loss = data.get('val_loss', [])
                val = val_loss[-1]
                # On prépare la ligne d'objectif (ex: 2.61 target court terme et la cible 2.5)
                target_val = 2.61
                target_line = [target_val] * len(steps)
                obj_val = 2.50
                obj_line = [obj_val] * len(steps)
                if (vMin == -1):
                    losses = val_loss[-50:]
                    yMin = min(losses) * 0.97
                    yMax = max(losses) * 1.07
                if len(steps) > 5:
                    # On nettoie le terminal
                    os.system('clear')
                    print(f"--- Live Status | Step: {steps[-1]} | Val Loss: {val_loss[-1]:.4f} ---")
                    
                    # On trace la courbe en ASCII
                    fig = plotille.Figure()
                    fig.width = 80
                    fig.height = 20
                    # Force l'affichage en entier pour l'axe X (les steps)
                    fig.x_label_fmt = "{:.0f}".format
                    fig.set_x_limits(min_=steps[0], max_=steps[-1])
                    fig.set_y_limits(min_=yMin, max_=yMax)

                    fig.plot(steps, val_loss, lc='red', label='Val Loss')
                    # La ligne d'objectif (en bleu ou blanc pour différencier)
                    fig.plot(steps, target_line, lc='yellow', label=f'Target {target_val}')
                    fig.plot(steps, obj_line, lc='blue', label=f'Target {obj_val}')
                    print(fig.show())
                    print("\nMis à jour à :", time.strftime("%H:%M:%S"))
                
            except Exception as e:
                pass
        time.sleep(60)

if __name__ == "__main__":
    # 1. Définition des arguments
    parser = argparse.ArgumentParser(description="Live Monitor pour GPT")
    parser.add_argument('--min', type=float, default=3.5, help='Limite minimale de Y')
    parser.add_argument('--max', type=float, default=6.0, help='Limite maximale de Y')
    parser.add_argument('--window', type=int, default=10, help='Taille du lissage')
    parser.add_argument('--mode', type=int, default=0, help='mode 0:Ascii | 1:Live | 2:Silent')
    
    args = parser.parse_args()

    if (args.mode == 0):
        terminal_monitor(args.min, args.max)
    elif (args.mode == 1):
        # On pointe vers ton fichier d'historique
        viz = TrainingVisualizer('model/my_wiky_history.json')
        # On lance avec tes paramètres de lissage
        viz.plot_live(window_size=10, lim_y=(args.min, args.max))
    elif (args.mode == 2):
        viz = TrainingVisualizer()
        while True:
            viz.plot_silent(window_size=args.window, lim_y=(args.min, args.max))
            time.sleep(60) # Rafraîchissement toutes les 60 secondes
    else:
        print('Choisissez un mode valide 0|1|2')       
