import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class PoussinVisualizer:
    def __init__(self, monitor_path, info_path, model_name="model_5"):
        self.monitor_path = monitor_path
        self.model_name = model_name
        self.config = self.load_json(info_path)
        
        # Initialisation des vecteurs de données
        self.steps = []
        self.tokens = []
        self.train_loss = []
        self.val_loss = []
        self.gap = []
        self.lrs = []

    def load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def step_to_tokens(self, step):
        """Convertit un step en Millions de Tokens selon les phases EBS/BlockSize."""
        phases = self.config.get(self.model_name, [])
        total_tokens = 0
        for p in phases:
            if step > p['start_step']:
                steps_in_phase = min(step, p['end_step']) - p['start_step']
                total_tokens += steps_in_phase * p['ebs'] * p['block_size']
        return total_tokens / 1_000_000

    def parse_logs(self):
        """Lit le monitor.log et extrait les métriques."""
        if not os.path.exists(self.monitor_path):
            print(f"❌ Erreur : {self.monitor_path} introuvable.")
            return

        with open(self.monitor_path, 'r') as f:
            for line in f:
                if "step" not in line or "loss" not in line:
                    continue
                try:
                    # Extraction basée sur ton format : step 11500: train loss 2.87, val loss 3.01, lr 2.92e-04
                    main_part = line.split('|')[0]
                    parts = main_part.split(',')
                    
                    s_val = int(parts[0].split('step')[1].split(':')[0].strip())
                    t_loss = float(parts[0].split('loss')[1].strip())
                    v_loss = float(parts[1].split('loss')[1].strip())
                    lr = float(parts[2].split('lr')[1].strip())

                    self.steps.append(s_val)
                    self.tokens.append(self.step_to_tokens(s_val))
                    self.train_loss.append(t_loss)
                    self.val_loss.append(v_loss)
                    self.gap.append(v_loss - t_loss)
                    self.lrs.append(lr)
                except Exception as e:
                    continue

    def calculate_ema(self, data, alpha=0.1):
        """Lissage Exponentiel Mobile."""
        if len(data) == 0: return np.array([])
        ema = [data[0]]
        for i in range(1, len(data)):
            ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
        return np.array(ema)

    def exponential_model(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def run_analysis(self, use_tokens=True):
        """Génère le graphique complet avec encart stats et projection."""
        x = np.array(self.tokens if use_tokens else self.steps)
        x_label = "Millions de Tokens" if use_tokens else "Steps d'entraînement"
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(15, 9), constrained_layout=True)
        
        # 1. Courbes de base
        ax.plot(x, self.train_loss, color='gray', alpha=0.2, label="Train Loss (Raw)")
        ema_val = self.calculate_ema(self.val_loss, alpha=0.15)
        ax.plot(x, ema_val, color='#00FFCC', lw=2.5, label="Validation Loss (EMA)")
        
        # 2. Cible (Target)
        target = 2.80
        ax.axhline(y=target, color='red', linestyle='--', alpha=0.6, label=f"Objectif {target}")

        # 3. Événements (Marqueurs verticaux)
        # On définit en steps, la classe convertit en X (tokens ou steps)
        events = {
            10100: "Switch Ratio 60/35/5",
            8100: "EBS Jump 192->384"
        }
        for s_ev, label in events.items():
            if s_ev in self.steps:
                x_ev = self.step_to_tokens(s_ev) if use_tokens else s_ev
                ax.axvline(x=x_ev, color='orange', linestyle=':', alpha=0.7)
                ax.text(x_ev, ax.get_ylim()[1]*0.95, f" {label}", color='orange', fontsize=9)

        # 4. Projection mathématique
        try:
            popt, _ = curve_fit(self.exponential_model, x, ema_val, p0=(1, 0.0001, 2.5))
            x_future = np.linspace(x[0], x[-1] * 1.5, 100)
            y_future = self.exponential_model(x_future, *popt)
            ax.plot(x_future, y_future, 'y:', alpha=0.5, label="Projection Horizon")
            horizon_loss = y_future[-1]
        except:
            horizon_loss = 0

        # 5. L'Encart de Statistiques (Info Box)
        last_step = self.steps[-1]
        last_val = self.val_loss[-1]
        last_gap = self.gap[-1]
        
        info_text = (
            f"📊 STATS POUSSIN-V5\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Dernier Step : {last_step}\n"
            f"Tokens vus   : {x[-1]:.1f}M\n"
            f"Loss Val     : {last_val:.4f}\n"
            f"Gap T/V      : {last_gap:.4f}\n"
            f"Projection   : {horizon_loss:.3f}\n"
            f"Learning Rate: {self.lrs[-1]:.2e}"
        )
        
        plt.text(0.98, 0.75, info_text, transform=ax.transAxes, 
                 fontsize=11, verticalalignment='top', horizontalalignment='right',
                 family='monospace', bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.2))

        # Cosmétique
        ax.set_title(f"Analyse de Convergence : {self.model_name}", fontsize=16, color='white', pad=20)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Loss (Cross-Entropy)", fontsize=12)
        ax.grid(True, which='both', linestyle='--', alpha=0.1)
        ax.legend(loc='upper left')
        
        # Sauvegarde
        os.makedirs("model/screenshot", exist_ok=True)
        plt.savefig("model/screenshot/analyse_tokens_v5.png", dpi=150)
        plt.show()

# --- EXÉCUTION DU SCRIPT ---
if __name__ == "__main__":
    viz = PoussinVisualizer(
        monitor_path='model/monitor.log', 
        info_path='model/train_info.json',
        model_name="model_5"
    )
    viz.parse_logs()
    # On affiche en tokens pour une analyse de densité réelle
    viz.run_analysis(use_tokens=True)
