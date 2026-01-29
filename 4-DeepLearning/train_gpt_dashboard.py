import json, re, time, os
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# --- CONFIGURATION (À vérifier dans ton script d'entraînement) ---
VERSION = 'v3.3'
LOG_FILE = 'model/training.log'
BATCH_SIZE = 128   # au réel 32 Batch_size x 4 grad_accum 
BLOCK_SIZE = 256   
TARGET_STEP = 46589 # Nombre total de steps pour 1 epoch : 5963390 block de 256 /(32*4) = 46589
TARGET_TRAIN = 75000
LIGNE_LEN = 67

#---
# Palette de couleurs ANSI pour GPT-Monitor
UI = {
    "RESET": "\033[0m",
    "BOLD":  "\033[1m",
    "DIM":   "\033[2m",
    "UNDER": "\033[4m",
    # Couleurs de texte
    "GREEN":  "\033[92m",
    "YELLOW": "\033[93m",
    "RED":    "\033[91m",
    "BLUE":   "\033[94m",
    "MAGENTA":"\033[95m",
    "CYAN":   "\033[96m",
    "WHITE":  "\033[97m",
    "GRAY":   "\033[90m",   
    # Couleurs de fond (si besoin pour des étiquettes)
    "BG_RED": "\033[41m",
    "BG_GREEN": "\033[42m",
}
# Modèle de décroissance : a est l'amplitude, b la vitesse, c l'asymptote (le plancher)
# def loss_model(x, a, b, c):
#     return a * np.exp(-b * (x / 10000)) + c
def loss_model_old(x, a, b, c):
    return a * np.exp(-b * np.log(x + 1)) + c

def get_forecast_old(steps, losses, horizons):
    try:
        # On normalise les steps pour aider l'optimiseur
        x_data = steps
        y_data = losses
        
        # Estimation initiale : a=amplitude, b=décroissance, c=valeur finale visée
        p0 = [losses[0] - 2.5, 0.5, 2.5] 
        popt, _ = curve_fit(loss_model, x_data, y_data, p0=p0, maxfev=2000)
        
        predictions = {}
        for h in horizons:
            future_step = steps[-1] + h
            pred = loss_model(future_step, *popt)
            predictions[h] = max(1.8, pred) # On met un "plancher" réaliste à 1.8
        return predictions, popt[2] # popt[2] est l'asymptote théorique
    except:
        # Backup si Scipy échoue ou n'est pas là
        return None, None

# Nouvelle prédictions ..
def loss_model(x, a, b, c):
    # Forme classique des Scaling Laws: L(step) = a * (step^-b) + c
    return a * np.power(x + 1, -b) + c

def get_forecast(steps, losses, horizons):
    try:
        # On filtre les valeurs aberrantes ou les zéros
        mask = steps > 0
        x_data = steps[mask]
        y_data = losses[mask]
        
        # Initialisation intelligente (p0) et BORNES (bounds)
        # a > 0, b entre 0 et 2 (décroissance), c entre 1.0 et 3.0 (plancher réaliste)
        p0 = [10.0, 0.1, 2.5]
        bounds = ([0.1, 0.001, 1.0], [100.0, 1.0, 3.5])
        
        popt, _ = curve_fit(loss_model, x_data, y_data, p0=p0, bounds=bounds, maxfev=5000)
        
        predictions = {}
        for h in horizons:
            future_step = steps[-1] + h
            pred = loss_model(future_step, *popt)
            predictions[h] = pred
        return predictions, popt[2]
    except Exception as e:
        return None, None

def parse_logs(current_log):
    steps, losses, times, lr= [], [], [], []
    
    # 1. Liste des fichiers historiques à scanner (dans l'ordre)
    # On cherche training_01.log, training_02.log, etc.
    history_files = sorted([f for f in os.listdir('model/') if re.match(r'training_\d+\.log', f)])
    all_log_files = ['model/' + f for f in history_files] + [current_log]
    
    # pattern = r"step (\d+): .*val loss ([\d.]+), .* \(([\d.]+)s\)"
    # pattern = r"step (\d+): train loss ([\d.]+), val loss ([\d.]+), lr ([\d.e-]+) \(([\d.]+)s\) \| .* Po: ([\d.]+)%"
    pattern = r"step (\d+): train loss ([\d.]+), val loss ([\d.]+), lr ([\d.e-]+) \(([\d.]+)s\)" # \| .* Po: ([\d.]+)%"
    
    for file_path in all_log_files:
        if not os.path.exists(file_path): continue
        with open(file_path, 'r') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    s = int(match.group(1))
                    # On évite les doublons si un step est présent dans deux fichiers
                    if not steps or s > steps[-1]:
                        steps.append(s)
                        losses.append(float(match.group(3)))
                        lr.append(float(match.group(4)))
                        times.append(float(match.group(5)))
                        
    return np.array(steps), np.array(losses), np.array(times), np.array(lr)

def parse_logs_single(file_path):
    pattern = r"step (\d+): .*val loss ([\d.]+), .* \(([\d.]+)s\)"
    steps, losses, times = [], [], []
    if not os.path.exists(file_path): return steps, losses, times
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                steps.append(int(match.group(1))); losses.append(float(match.group(2))); times.append(float(match.group(3)))
    return np.array(steps), np.array(losses), np.array(times)

def get_intel_level(ema_loss):
    if ema_loss > 5.0: return "CHAOS.    ", "🔴" , "\033[91m"
    if ema_loss > 4.0: return "PHONÉTIQUE", "🟠", "\033[91m"
    if ema_loss > 3.2: return "SYNTAXIQUE", "🟡", "\033[93m"
    if ema_loss > 2.8: return "PERROQUET ", "🟢", "\033[92m"
    if ema_loss > 2.6: return "ÉTUDIANT. ", "🔵", "\033[94m"
    if ema_loss > 2.4: return "EXPERT.   ", "🟣", "\033[95m"
    return "MÉMORISATION", "🔥", "\033[1;38;5;208m"

def get_speed_level(s_step):
    if s_step < 6.0:
        return "🟣", UI["MAGENTA"], "-----"
    elif s_step < 6.5:
        return "🔵", UI["BLUE"], "GLACE"
    elif s_step < 7.0:
        return "🟢", UI["GREEN"], "FROID"
    elif s_step < 8.0:
        return "🟡", UI["YELLOW"], "CHAUD"
    else:
        return "🔴", UI["RED"], "SURCHAUFFE"

def calculate_saturation_score(steps, losses, lr, window=5000):
    """
    Calcule la pente réelle de la perte et le score d'efficience.
    """
    # On ne garde que les données récentes pour éviter le biais du début
    mask = steps > (steps[-1] - window)
    x = steps[mask].reshape(-1, 1)
    y = losses[mask]
    
    if len(x) < 10: return 0, 0, 2.6  # Sécurité si pas assez de données

    # Régression linéaire pour trouver la pente réelle (Trend)
    model = LinearRegression().fit(x, y)
    slope_per_step = model.coef_[0]
    theoretical_floor = model.intercept_ + (slope_per_step * 150000) # Projection à 150k steps
    
    # Calcul de l'efficience : (Baisse de Loss pour 1k steps) / LR actuel
    trend_1k = abs(slope_per_step * 1000)
    efficiency_score = sum(losses[-10:-1])/sum(lr[-10:-1]) * 5e-4/3.8   # Normalisation empirique
    
    return efficiency_score, theoretical_floor, trend_1k

def get_dashboard():
    steps, losses, times, lr = parse_logs(LOG_FILE)
    if len(steps) < 5: 
        print('waiting pour 5 iter')
        return
    current_interval = steps[-1] - steps[-2]
    # Performances et vitesses
    sec_per_step = times[-1] / current_interval
    tok_s = (BATCH_SIZE * BLOCK_SIZE) / sec_per_step
    
    # Tendance & Projection (Moyenne glissante pour éviter les sauts)
    window = 10
    slope = np.polyfit(steps[-window:], losses[-window:], 1)[0]
    
    # Progression de l'entrainement epoch et steps total
    pct = (steps[-1] / TARGET_STEP) * 100 
    epoch = f"[epoch {int(pct/100 + 1)}]"
    if pct > 100:
        pct = pct % 100
    bar = "▬" * (int(pct/4)-1)+ f"{UI['RED']}▬" +"\033[0m"+ "┅" * (25 - int(pct/4)) #. ─
    # bar = "█" * int(pct/2) + "░" * (50 - int(pct/2))
    pct_train = (steps[-1] / TARGET_TRAIN) * 100
    bar_train = "▬" * (int(pct_train/4)-1)+ f"{UI['RED']}▬" +"\033[0m"+ "┅" * (25 - int(pct_train/4)) #. "▄"

    # Alerte Thermique (basée sur le temps de cycle)
    # Si le Mac met plus de 195s pour 25 steps, on affiche en Orange/Rouge
    # temp_icon = "🌡️ "
    temp_icon, temp_color, temp_text = get_speed_level(sec_per_step)
    
    # --- AFFICHAGE DU DASHBOARD ---
    os.system('clear')
    Titre = f"🚀 M1 GPT-MONITOR {VERSION}. - {steps[-1]}/{current_interval} - {epoch}"
    timestamp = datetime.now().strftime('%H:%M:%S')
    header_right = f"{temp_color}{temp_icon} \033[1m{timestamp}\033[0m" 

    # Calcul de l'espace (sans compter les codes ANSI invisibles)
    visible_right_len = len("🌡️ ") + len(timestamp)
    padding = LIGNE_LEN - len(Titre) - visible_right_len-1
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    print(f"{temp_color}\033[1m{Titre}\033[0m" + (" " * padding) + header_right)
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    print(f"E: {pct:.0f}% {UI['GREEN']}┣{bar}┫{UI['GREEN']}┣{bar_train}┫ T: {pct_train:.0f}%")
    print(f"{UI['GRAY']}"+f"━" * (LIGNE_LEN) + f"{UI['RESET']}")
    
    remaining_steps = TARGET_TRAIN - steps[-1]
    eta_seconds = remaining_steps * sec_per_step
    eta_str = str(timedelta(seconds=int(eta_seconds)))

    # --- CALCUL DE LA LOSS LISSÉE (EMA) ---
    # On initialise l'EMA avec la première valeur de la liste
    ema_loss = losses[0]
    for l in losses:
        ema_loss = 0.1 * l + (1 - 0.1) * ema_loss
    level_name, level_icon, level_color = get_intel_level(ema_loss)
    
    # --- CALCUL DU TREND SUR LA LOSS LISSÉE ---
    # On prend les 15 derniers points lissés pour une tendance stable
    window = 15
    if len(steps) >= window:
        # On recalcule une petite série lissée pour la pente
        # Cela évite les "Trend: +0.059" quand un seul point remonte
        y_segment = losses[-window:]
        x_segment = steps[-window:]
        slope = np.polyfit(x_segment, y_segment, 1)[0]
    else:
        slope = 0

    # Calcul d'Efficience du model (v. simple sinon ajouter Delta Loss/Delta LR)
    efficiency, asymptote_new, trend_1k = calculate_saturation_score(steps, losses, lr, window=5000)
    status = f"⚡ {UI['BLUE']}PRODUCTIF {UI['RESET']}" if efficiency > 0.5 else f"🐢 {UI['RED']}SATURATION{UI['RESET']}"
    level_name_new, _,color_asympt_new = get_intel_level(asymptote_new)
    # --- AFFICHAGE MIS À JOUR ---
    color_l = "\033[92m" if ema_loss < 2.90 else "\033[94m"
    color_t = "\033[92m" if slope < 0 else "\033[91m"

    # Status Row
    color_loss = "\033[92m" if losses[-1] < 2.95 else "\033[93m"
    color_trend = "\033[92m" if slope < 0 else "\033[91m"
    total_tokens = steps[-1] * BATCH_SIZE * BLOCK_SIZE

    print(f"  {level_icon} {level_color}{level_name}{UI['RESET']} | 📚 SAVOIR ABSORBÉ : {total_tokens / 1e6:.2f} Millions de tokens")        
    print(f"  {status} | LOSS: {color_loss}{losses[-1]:.3f}\033[0m | EMA: {color_l}{ema_loss:.3f}\033[0m | TREND: {color_trend}{trend_1k:+.3f}/k\033[0m | {efficiency:.2f}")
    print(f"  SPD : {sec_per_step:.2f} s/st | DATA: {tok_s:,.0f} tok/s | ETA {TARGET_TRAIN}: {eta_str}")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    
    # Predictions
    print(f"  🔮 PRÉDICTIONS ",end="")
    for h in [1000, 10000]:
        target = losses[-1] + (slope * h)
        print(f" | +{h:5} st ➔ \033[1m{max(2.0, target):.3f}\033[0m", end="")
    print(f"\n{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")

    # Prédictions améliorées ...
    horizons = [1000, 5000, 10000]
    # On passe les 200 derniers points pour donner plus de contexte à Scipy
    #preds, asymptote = get_forecast(steps[-200:], losses[-200:], horizons)
    # finalement on envoie tout !
    preds, asymptote = get_forecast(steps, losses, horizons)
    _, _, color_asympt = get_intel_level(asymptote)
    asymptote_str = f"{asymptote:.4f}"
    target_txt = ['🎯', 'PLANCHER', 'THÉORIQUE', asymptote_str]
    # Affichage de l'asymptote avec couleur
    print(f"  🔮 PRÉVISIONS (Scaling Law - Scipy)              |       🎯")
    if preds:
        i=1
        for h, val in preds.items():
            eta = timedelta(seconds=int(h * sec_per_step))
            txt = f"      +{h:5} steps ➔  \033[1m{val:.4f}\033[0m  (dans {eta})"
            pad = LIGNE_LEN - 16 - (len(txt)-8) 
            pad2 = (16 - len(target_txt[i])) // 2
            txt += (" "*pad) + "|" +(" "*pad2) + f"{color_asympt}" +target_txt[i]+ f"{UI['RESET']}"
            print(txt)
            i += 1
        print(f"   🎯 Nouvelle prévision modèle final : {UI['BOLD']}{color_asympt_new}{asymptote_new:.2f} {level_name_new}{UI['RESET']}")
        print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    else:
        print("  (Collecte de données pour Scipy...)")

    # Graphique amélioré
    mini = losses[-40:] # On prend plus de points pour le graph
    m_min, m_max = min(mini), max(mini)
    print(f"  HISTORY (Last 40): ", end="")
    for v in mini:
        # On utilise des caractères de blocs pour simuler une courbe
        levels = [" "," ","▂","▃","▄","▅","▆","▇"]
        idx = int(((v - m_min) / (m_max - m_min + 1e-6)) * 7)
        print(levels[7-idx], end="")
    print(f"\n{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")

while True:
    try: get_dashboard()
    except: 
        print('error:')
    time.sleep(30)