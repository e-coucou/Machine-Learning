import json, re, time, os, psutil, subprocess, json, math
import numpy as np
import torch, gc
from datetime import timedelta, datetime
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# --- CONFIGURATION (À vérifier dans ton script d'entraînement) ---
VERSION = 'v5.9.5'
LOG_FILE = 'model/training.log'
MONITOR_FILE = 'model/monitor.log'
HISTORY_FILE = 'model/my_wiky_history.json'
MODEL_FILE = 'model/my_wiki.pth'
BATCH_SIZE = 128   # au réel 32 Batch_size x 4 grad_accum 
BLOCK_SIZE = 256   
TARGET_STEP = 46589 # Nombre total de steps pour 1 epoch : 5963390 block de 256 /(32*4) = 46589
TARGET_TRAIN = 80000
LIGNE_LEN = 74
TARGET_BLOCK_DS1 = 3367466 #5051199 # old avec wiki_raw 5963390
TARGET_BLOCK_DS2 = 1619593 #2429390
TARGET_BLOCK_DS3 = 116091 #174137

timeout = 0
first = True
model_ema = False
n_params = 0
update = {'status': 0, 'step': 0, 'elapse':244}

#---
# Palette de couleurs ANSI pour GPT-Monitor
#  construction du escape \033[1 gras ; 38 foreground 48 background ; 5 mode 256 couleurs ; 201 code couleur m
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
    "ORANGE": "\033[38;5;208m",
    "B_OR":   "\033[1;38;5;208m",
    "B_MAG":  "\033[1;38;5;201m",
    # Couleurs de fond (si besoin pour des étiquettes)
    "BG_RED": "\033[41m",
    "BG_GREEN": "\033[42m",
}

color_rank = ["\033[91m","\033[38;5;208m","\033[93m","\033[92m","\033[94m","\033[95m"]

def f_num(n):
    # Formate avec des virgules puis change en points
    return f"{int(n):,}".replace(',', '.')

def calcul_epoch(step, target, mix, batch, grad, t_step):
#    epoch_ds1 = calcul_epoch(data_monitor[-1]['dataset1'] , TARGET_BLOCK_DS1, (1-params.get('mixed_ratio',0.4)), batch_size, grad_accum_steps, data_monitor[-1]['step'])
    epoch = 0
    if mix>0:
        epoch =  int((target/batch - step ) / ( mix * grad) + t_step)
    return epoch
    
def calcul_ema(data):
    # Calcul de la moyenne mobile exponentielle
    ema_loss=data[0]
    for l in data:
        ema_loss = 0.1 * l + (1 - 0.1) * ema_loss
    return ema_loss
    
def calcul_lr(it,params):
    # Utilisation des paramètres passés à l'init
    warmup = params.get('warmup_iters', 500)
    max_iters = params.get('lr_decay_iters', 80000)
    lr_max = params.get('learning_rate', 0.0003)
    lr_min = params.get('min_lr', lr_max * 0.1)
    # 1) Phase de warmup
    if it < warmup:
        return lr_max * it / warmup
    # 2) Phase de plateau bas
    if it > max_iters:
        return lr_min
    # 3) Phase de Cosine Decay
    decay_ratio = (it - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return lr_min + coeff * (lr_max - lr_min)
     

def get_checkpoint(path):
    global first, n_params, model_ema
    ckpt = torch.load(path, map_location='cpu', mmap=True)
    wiki = ckpt.get('total_step_wiki', 0)
    cult = ckpt.get('total_step_cult', 0)
    litt = ckpt.get('total_step_litt', 0)
    step = ckpt.get('total_steps_done', 0)
    config = ckpt.get('config',None)
    params = ckpt.get('params',None)
    if first:
        model_ema = ckpt.get('model_ema',None)
        n_params = sum(t.numel() for t in ckpt['model'].values() if isinstance(t, torch.Tensor))
        first = False
    del ckpt
    return step, wiki, cult, litt, config, params

def get_memory_status():
    # RAM système
    vm = psutil.virtual_memory()
    # Pression mémoire (en %) - c'est l'indicateur le plus important sur macOS
    # Sur Mac, psutil ne donne pas la "pression" exacte d'Apple, mais on l'estime ainsi :
    pression = vm.percent 
    
    # Mémoire spécifique à ton process Python
    process = psutil.Process(os.getpid())
    mem_rss = process.memory_info().rss / (1024**2) # En Mo
    swap = psutil.swap_memory().used / (1024**2) # en Mo
#    mps = torch.mps.current_allocated_memory() / 1_048_576 # en Mo
    
    status = "🟣" if pression < 70 else "🟢" if pression < 80 else "🟡" if  pression<85 else "🟠" if pression < 90 else f"\U0001F534"

    return status, pression, mem_rss, swap

def get_cpu_usage():
    # Utilisation globale du CPU (tous les cœurs)
    cpu = psutil.cpu_percent(interval=None)
    status = "🟣" if cpu <10 else "🟢" if cpu < 50 else "🟡" if cpu < 85 else "🔴"
    color =  "\033[95m" if cpu<10 else "\033[92m" if cpu < 50 else "\033[93m" if cpu < 85 else "\033[91m"
    return cpu, status, color

def get_gpu_usage():
    """Récupère l'utilisation GPU via powermetrics (macOS)"""
    try:
        # On lance une mesure rapide (échantillon de 10ms)
        cmd = ["sudo", "powermetrics", "-n", "1", "--samplers", "gpu_power"]
        result = subprocess.check_output(cmd).decode("utf-8")

        gpu = 0.0
        # On cherche la ligne "GPU Active residency"  GPU HW active residency
        match = re.search(r"GPU HW active residency:\s+([\d.]+)%", result)
        if match:
            gpu = float(match.group(1))
        # 2. Extraction de la puissance (mW)
        gpu_watts = 0.0
        match_pwr = re.search(r"GPU Power:\s+(\d+) mW", result)
        if match_pwr:
            gpu_watts = float(match_pwr.group(1)) / 1000.0 # On convertit en Watts

        status = "🔴" if gpu < 50 else "🟡" if gpu < 90 else "🟣"
        color = "\033[91m" if gpu < 50 else "\033[93m" if gpu < 90 else "\033[95m"
        return gpu, status, color, gpu_watts
        
    except Exception:
        return 0.0, "❌", "\033[96m" ,0.0

    return 0.0, "?", "\033[96m", 0.0

def get_ssd_usage():
    try:
#        result = subprocess.run(['diskutil','info','disk0'], capture_output=True, text=True)
        result = subprocess.run(['iostat','-d','-c','2','disk0'], capture_output=True, text=True, timeout=3)
#        for line in result.stdout.split('\n'):
#            print(line.strip())
        # On ne récupère que la dernière ligne
        lines = result.stdout.strip().split('\n')
        last_line = lines[-1].split()
        
        if len(last_line) >= 3:
            kbt = float(last_line[0]) # KiloByte par transfert
            tps = float(last_line[1]) # Transfet Par Seconde
            mbs = float(last_line[2]) # MegaByte par seconde le débit ...
            
            # Indicateur de santé (0 à 100)
            # On considère que 2000 tps est le seuil de saturation sur M1
            stress_score = min(100, (tps / 2000) * 100)
            status = "🟣" if stress_score < 15 else "🟡" if stress_score < 50 else "🔴"
            color = "\033[95m" if stress_score < 15 else "\033[93m" if stress_score < 50 else "\033[91m"

            return round(stress_score, 1), status, color
#            return { "tps": tps, "mbs": mbs, "stress_pct": round(stress_score, 1) }
    except Exception as e:
        print(e)
        return 0.0, "❌", "\033[96m"

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

def parse_monitor(monitor_log):
#    steps, losses, times, lr, trains = [], [], [], [], []
    """
    data = []
    if not os.path.exists(monitor_log): return data
    with open(monitor_log, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    """
    if not os.path.exists(monitor_log): 
        return []
    
    with open(monitor_log, 'r') as f:
        # On lit tout d'un coup, c'est plus rapide que de boucler sur le fichier
        lines = f.readlines()
        
    # Utilisation d'une "list comprehension" : c'est 2x plus rapide qu'une boucle .append()
    data = [json.loads(line) for line in lines if line.strip()]
    
    return data


def parse_history(history_log):
    if not os.path.exists(history_log):
        print(f"❌ Fichier principal introuvable : {history_log}")
        return np.empty(0)
        
    with open(history_log, "r") as f:
        data = json.load(f)

    steps = np.array(data["steps"])
    train_loss = np.array(data["train_loss"])
    val_loss = np.array(data["val_loss"])
    time_run = np.array(data["time_elapse"])

    return steps, train_loss, val_loss, time_run

def parse_logs(current_log):
    steps, losses, times, lr, trains = [], [], [], [], []
    
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
                        trains.append(float(match.group(2)))
                        losses.append(float(match.group(3)))
                        lr.append(float(match.group(4)))
                        times.append(float(match.group(5)))
                        
    return np.array(steps), np.array(losses), np.array(times), np.array(lr), np.array(trains)

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
    if ema_loss > 4.00: return "CHAOS.    ", "🔴" , "\033[91m"
    if ema_loss > 3.25: return "PHONÉTIQUE", "🟠", "\033[38;5;208m"
    if ema_loss > 2.90: return "SYNTAXIQUE", "🟡", "\033[93m"
    if ema_loss > 2.60: return "PERROQUET ", "🟢", "\033[92m"
    if ema_loss > 2.45: return "ÉTUDIANT. ", "🔵", "\033[94m"
    if ema_loss > 2.35: return "EXPERT.   ", "🟣", "\033[95m"
    return "MÉMORISATION", "🔥", "\033[1;38;5;201m"

def get_speed_level(tok_s): # ce sont désormais des tokens/s
    if tok_s < 1000:
        return "🔴", UI["RED"], "SURCHAUFFE"
    elif tok_s < 1200:
        return "🟠", UI["ORANGE"], "BOUILLANT"
    elif tok_s < 1400:
        return "🟡", UI["YELLOW"], "CHAUD"
    elif tok_s < 1600:
        return "🟢", UI["GREEN"], "FROID"
    elif tok_s < 1800:
        return "🔵", UI["BLUE"], "GLACE"
    else:
        return "🟣", UI["MAGENTA"], "-----"
def get_green_level(level):
    # Calcul de l'indicateur visuel
    if level < 5:
        leaf = "🌱" # Exceptionnel (ton cas actuel)
    elif level < 8:
        leaf = "🍀" # Très bon
    elif level < 12:
        leaf = "🌿" # Très bon
    elif level < 15:
        leaf = "🍂" # Le swap ou la chauffe ralentit tout
    else:
        leaf = "☢️" # c'est la panique ...
    return leaf

def get_micro_graph(data,m_min=0,m_max=1,seuil_l=0.05, seuil_h=0.12):
    # Graphique amélioré
    if m_min == -1:
        m_min, m_max = min(data), max(data)
    levels = [" ","_","▂","▃","▄","▅","▆","▇"]
    graph = ""
    for v in data:
        # On utilise des caractères de blocs pour simuler une courbe
        c = UI['GREEN'] if v < seuil_l else (UI['YELLOW'] if v < seuil_h else UI['RED'])
        idx = int(((v - m_min) / (m_max - m_min + 1e-6)) * 7)
        g = (levels[7-idx])
        graph += f"{c}{g}"
    return graph
 
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

def predict_loss_projection(steps, losses, target_step, start_idx=-1):
    """
    Projection robuste de la Loss pour LLM.
    On bloque les coefficients dans une zone 'réelle'
    """
    # Modèle : L(s) = a * s^-b + c
    def power_law(s, a, b, c):
        return a * np.power(s, -b) + c

    # On garde une fenêtre glissante si start_idx n'est pas défini
    if start_idx == -1:
        start_idx = len(steps) // 10 
    
    fit_steps = np.array(steps[start_idx:])
    fit_losses = np.array(losses[start_idx:])

    try:
        # --- ASTUCE 1 : Les Bornes (Bounds) ---
        # a > 0
        # b (vitesse) : entre 0.01 (lent) et 0.8 (rapide). Un LLM dépasse rarement 0.5.
        # c (asymptote) : la loss finale théorique. Pour ton modèle, elle sera entre 1.5 et 3.5.
        lower_bounds = [0.1, 0.01, 1.5]
        upper_bounds = [1000, 0.8, 4.0] 
        
        # --- ASTUCE 2 : Poids dégressifs (Sigma) ---
        # On donne plus d'importance aux points RÉCENTS pour capturer la pente actuelle,
        # tout en gardant les anciens pour la courbure globale.
        sigmas = np.ones(len(fit_steps))
        # On divise l'erreur autorisée par 2 sur les 10% derniers points
        sigmas[-max(1, len(sigmas)//10):] = 0.5 

        # Estimation initiale réaliste
        p0 = [10, 0.2, 2.8] 
        
        popt, _ = curve_fit(
            power_law, fit_steps, fit_losses, 
            p0=p0, 
            bounds=(lower_bounds, upper_bounds),
            sigma=sigmas,
            maxfev=20000
        )
        
        # Génération de la projection
        """ Uniquement pour les courbes
        future_steps = np.arange(steps[-1], target_step, 100)
        projection = power_law(future_steps, *popt)
        
        return future_steps, projection, popt
        """
        return popt
    except Exception as e:
        print(f"Erreur d'extrapolation : {e}")
        print(losses[100:110])
        print('dimension du loss:',len(losses), len(steps))
        return None # Uniquement cour, None, None

def check_efficiency(popt, current_step, target_step=60000):
    """
    Analyse l'efficacité de l'apprentissage à un horizon donné.
    """
    a, b, c = popt
    
    # 1. Calcul de la perte projetée à l'horizon (ex: 60k)
    # Formule : L(s) = a * s^-b + c
    loss_at_horizon = a * np.power(target_step, -b) + c
    
    # 2. Calcul de la pente (dérivée) au step actuel
    # La dérivée de a*s^-b + c est : -a * b * s^-(b+1)
    # Elle représente de combien la loss descend par step.
    current_slope = -a * b * np.power(current_step, -(b + 1))
    
    # 3. Calcul de la pente à l'horizon
    # Pour savoir si le modèle sera "mort" (plateau) ou s'il descendra encore
    future_slope = -a * b * np.power(target_step, -(b + 1))
    
    return {
        "loss_horizon": round(loss_at_horizon, 4),
        "current_slope_per_1000": round(current_slope * 1000, 5), # Gain pour 1000 steps
        "future_slope_per_1000": round(future_slope * 1000, 5),
        "is_plateau": abs(future_slope * 1000) < 0.005 # Seuil arbitraire de stagnation
    }

def get_gn_status(norms):
    if not norms:
        return "N/A", ""
    
    avg_gn = sum(norms) / len(norms)
    max_gn = max(norms)
    
    # Détermination du statut et de la couleur (codes ANSI)
    if max_gn >= 0.95:
        # Zone de Clipping : le modèle force trop
        status = f"{UI['RED']}\u26a1 CRITIQUE (Max: {max_gn:.3f}){UI['RESET']}" 
    elif max_gn > 0.70:
        # Zone de turbulence : attention au dataset
        status = f"{UI['ORANGE']}\u26a0INSTABLE (Max: {max_gn:.3f}){UI['RESET']}"
    elif avg_gn < 0.01:
        # Zone de gel : le modèle n'apprend plus
        status = f"{UI['CYAN']}\u2744GELÉ (Avg: {avg_gn:.3f}){UI['RESET']}"
    else:
        # Zone nominale
        status = f"{UI['GREEN']}\u2714 OK ({avg_gn:.3f}){UI['RESET']}"
        
    return status, avg_gn, max_gn

def get_dashboard():
    global update
    steps, losses, times, lr, trains = parse_logs(LOG_FILE)
    last_update = os.path.getmtime(LOG_FILE)
    steps_h, trains_h, vals_h, times_h = parse_history(HISTORY_FILE)
    gap_h = vals_h - trains_h
    data_monitor = parse_monitor(MONITOR_FILE)

    file = MODEL_FILE
    status, pression, mem_rss, swap = get_memory_status()
    step, wiki, cult, litt, config, params = get_checkpoint(file)
    batch_size = params.get('batch_size',32)
    block_size = config.get('block_size',BLOCK_SIZE)
    target_train = params.get('lr_decay_iters',TARGET_TRAIN)
    grad_accum_steps = params['grad_accum_steps']
    EBS = batch_size * grad_accum_steps
    
    if len(steps) < 5: 
        print('waiting pour 5 iter')
        return
    current_interval = steps[-1] - steps[-2]
  
    # Performances et vitesses
#    sec_per_step = times[-1] / current_interval
    sec_per_step = data_monitor[-1]['elapse'] / data_monitor[-1]['inter']
    tok_s = (EBS * block_size) / sec_per_step
    
    # Tendance & Projection (Moyenne glissante pour éviter les sauts)
    window = 10
    slope = np.polyfit(steps[-window:], losses[-window:], 1)[0]

    # Progression de l'entrainement epoch et steps total
#    pct_ds1 = (data_monitor[-1]['dataset1'] * batch_size/TARGET_BLOCK_DS1) * 100
#    pct = (steps[-1] / TARGET_STEP) * 100 
#    epoch = f"[epoch {int(pct/100 + 1)}]"
#    if pct > 100:
#        pct = pct % 100
#    bar = "▬" * (int(pct_ds1/4)-1)+ f"{UI['RED']}▬" +UI['DIM']+UI['WHITE']+ "┅" * (25 - int(pct_ds1/4)) #. ─
    # bar = "█" * int(pct/2) + "░" * (50 - int(pct/2))
    pct_train = (steps[-1] / target_train) * 100
    bar_train = "▬" * (int(pct_train/4)-1)+ f"{UI['RED']}\u2719" +UI['DIM']+UI['WHITE']+ "┅" * (25 - int(pct_train/4)) #. "▄"

    # Alerte Thermique (basée sur le temps de cycle)
    # Si le Mac met plus de 195s pour 25 steps, on affiche en Orange/Rouge
    # temp_icon = "🌡️ "
    temp_icon, temp_color, temp_text = get_speed_level(tok_s) # on passe en token par seconde

    # Calcul du progrès interne
    elapsed = time.time() - last_update
    steps_since_log = elapsed / sec_per_step
    next_step = steps[-1] + current_interval
    current_step_est = steps[-1] + int(steps_since_log)
    p_step = 1 - (next_step - current_step_est)/current_interval

    ratio_wiki = wiki/(wiki+cult+litt)*100
    ratio_cult = cult/(wiki+cult+litt)*100
    ratio_litt = litt/(wiki+cult+litt)*100
    remaining_steps = target_train - steps[-1]
    eta_seconds = remaining_steps * sec_per_step
    eta_str = str(timedelta(seconds=int(eta_seconds)))

    ssd_usage, ssd_status, ssd_color = get_ssd_usage()        
    cpu_usage, cpu_status, cpu_color = get_cpu_usage()
    gpu_usage, gpu_status, gpu_color, gpu_watts = get_gpu_usage()

    # Indicateur visuel "Green"
    monitor_inter = data_monitor[-1]['inter']
    tok_s_live = (EBS * block_size) / (data_monitor[-1]['elapse']) * monitor_inter
    mJ_tok = (gpu_watts / tok_s_live) * 1000. if tok_s_live > 0 else 0
    green_status = get_green_level(mJ_tok)

    lr_h = []
    for it in steps_h:
        lr_h.append(calcul_lr(it, params))
    lr_h = np.array(lr_h)

    monitor_steps = [s['step'] for s in data_monitor]
    monitor_losses = [l['loss'] for l in data_monitor]

  # Progression de l'entrainement epoch et steps total
    epoch_1 = int((data_monitor[-1]['dataset1'] * batch_size/TARGET_BLOCK_DS1))
    pct_ds1 = (data_monitor[-1]['dataset1'] * batch_size/TARGET_BLOCK_DS1 - epoch_1) * 100
    epoch_2 = int((data_monitor[-1]['dataset2'] * batch_size/TARGET_BLOCK_DS2))
    pct_ds2 = (data_monitor[-1]['dataset2'] * batch_size/TARGET_BLOCK_DS2 - epoch_2) * 100
    epoch_3 = int((data_monitor[-1]['dataset3'] * batch_size/TARGET_BLOCK_DS3))
    pct_ds3 = (data_monitor[-1]['dataset3'] * batch_size/TARGET_BLOCK_DS3 - epoch_3) * 100
    pct = (steps[-1] / TARGET_STEP) * 100 
    epoch = f"[epoch {int(pct/100 + 1)}]"
    if pct > 100:
        pct = pct % 100
    bar_ds1 = "▬" * (int(pct_ds1/4)-1)+ f"{UI['RED']}\u2719" +UI['DIM']+UI['CYAN']+ "┅" * (25 + (-1 if int(pct_ds1/4)==0 else 0) - int(pct_ds1/4))
    bar_ds2 = "▬" * (int(pct_ds2/4)-1)+ f"{UI['RED']}\u2719" +UI['DIM']+UI['CYAN']+ "┅" * (25 + (-1 if int(pct_ds2/4)==0 else 0) - int(pct_ds2/4))
    bar_ds3 = "▬" * (int(pct_ds3/4)-1)+ f"{UI['RED']}\u2719" +UI['DIM']+UI['CYAN']+ "┅" * (25 + (-1 if int(pct_ds3/4)==0 else 0) - int(pct_ds3/4))

    batch_ds1 = data_monitor[-1]['dataset1']
    batch_ds2 = data_monitor[-1]['dataset2']
    batch_ds3 = data_monitor[-1]['dataset3']
    
    ratio_ds1 = ( batch_ds1 - data_monitor[-4]['dataset1']) / grad_accum_steps / 3 / monitor_inter # vs :  (1-params.get('cult_ratio',0.)-params.get('litt_ratio',0.))
    ratio_ds2 = (data_monitor[-1]['dataset2'] - data_monitor[-4]['dataset2']) / grad_accum_steps / 3 / monitor_inter # vs : (params.get('cult_ratio',0.))
    ratio_ds3 = (data_monitor[-1]['dataset3'] - data_monitor[-4]['dataset3']) / grad_accum_steps / 3 / monitor_inter # vs :  (params.get('litt_ratio',0.))

    ratio_ds1 = ratio_ds1 if abs(ratio_ds1 -  (1-params.get('cult_ratio',0.)-params.get('litt_ratio',0.))) > 0.06 else  (1-params.get('cult_ratio',0.)-params.get('litt_ratio',0.))
    ratio_ds2 = ratio_ds2 if abs(ratio_ds2 - params.get('cult_ratio',0.)) > 0.06 else  params.get('cult_ratio',0.)
    ratio_ds3 = ratio_ds3 if abs(ratio_ds3 - params.get('litt_ratio',0.)) > 0.06 else  params.get('litt_ratio',0.)
    
    epoch_ds1 = calcul_epoch(data_monitor[-1]['dataset1'] , TARGET_BLOCK_DS1, ratio_ds1, batch_size, grad_accum_steps, data_monitor[-1]['step'])
    epoch_ds2 = calcul_epoch(data_monitor[-1]['dataset2'] , TARGET_BLOCK_DS2, ratio_ds2, batch_size, grad_accum_steps, data_monitor[-1]['step'])
    epoch_ds3 = calcul_epoch(data_monitor[-1]['dataset3'] , TARGET_BLOCK_DS3, ratio_ds3, batch_size, grad_accum_steps, data_monitor[-1]['step'])
    gn_status, gn_avg, gn_max = get_gn_status(data_monitor[-1]['grad_norm'])
    timeout_pattern = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']

    if (update['status'] == 1):
        update['status'] = 0
    else:
        if (update['step'] != data_monitor[-1]['step']):
            update['step'] = data_monitor[-1]['step']
            update['status']=1

    
    # --- CALCUL DE LA LOSS LISSÉE (EMA) ---
    # On initialise l'EMA avec la première valeur de la liste
    ema_loss = calcul_ema(vals_h)
    ema_name, ema_icon, ema_color = get_intel_level(ema_loss)
    loss_name, loss_icon, loss_color = get_intel_level(losses[-1])
    
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
    

    #--------------------------------
    # --- AFFICHAGE DU DASHBOARD --- ---------------------------------------------------------------------------
    #--------------------------------
    
    os.system('clear')
    Titre = f"  🚀   M1 MONITOR {VERSION}. - {steps[-1]}/{current_interval}"
    timestamp = datetime.now().strftime('%H:%M:%S')
    header_right = f"  \033[1m{timestamp}\033[0m" 
#    header_right = f"{temp_color}{timeout_pattern[timeout]}  \033[1m{timestamp}\033[0m" 
    visible_right_len = len("🌡️ ") + len(timestamp)
    padding = LIGNE_LEN - len(Titre) - visible_right_len-1
#    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    print(f"{loss_color}\033[1m{Titre}\033[0m" + (" " * padding) + header_right)

    # Affichage de l'entrainement en cours ...
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    print(f"  {status} RAM Système: {pression}% /({data_monitor[-1]['ram']}%) | Process: {mem_rss:.0f}Mo | Swap: {swap:.0f}Mo")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    print(f"  {cpu_status} CPU : {cpu_color}{cpu_usage:.1f}% {UI['RESET']} | {gpu_status} GPU : {gpu_color}{gpu_usage:.1f}% {UI['RESET']} ⚡ {gpu_watts:.1f}W | {ssd_status} SSD : {ssd_color}{ssd_usage:.1f}% | {green_status} {mJ_tok:.1f}mJ/tk")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
#    print(f"  Epoch: {pct:4.1f}% {UI['CYAN']}┣{bar}┫ {UI['RESET']}pour {TARGET_STEP} steps")
    print(f"     Train:    {pct_train:4.1f}% {UI['CYAN']}  ┣{bar_train}┫ {UI['RESET']}pour {f_num(target_train).rjust(10)} steps")
    print(f"     Wiki ({epoch_1:1d}): {pct_ds1:4.1f}% {UI['CYAN']}  ┣{bar_ds1}┫ {UI['RESET']}pour {f_num(TARGET_BLOCK_DS1).rjust(10)} blocks")
    print(f"     Cult ({epoch_2:1d}): {pct_ds2:4.1f}% {UI['CYAN']}  ┣{bar_ds2}┫ {UI['RESET']}pour {f_num(TARGET_BLOCK_DS2).rjust(10)} blocks")
    print(f"     Litt ({epoch_3:1d}): {pct_ds3:4.1f}% {UI['CYAN']}  ┣{bar_ds3}┫ {UI['RESET']}pour {f_num(TARGET_BLOCK_DS3).rjust(10)} blocks")
    print(f"     Wikipédia  [{ratio_ds1:.2f}] | batch: {batch_ds1:8} | end: {epoch_ds1:6} | {batch_ds1*block_size*batch_size/1000**2:6.1f} Mtoken")
    print(f"     CulturaX   [{ratio_ds2:.2f}] | batch: {batch_ds2:8} | end: {epoch_ds2:6} | {batch_ds2*block_size*batch_size/1000**2:6.1f} Mtoken")
    print(f"     Litteraire [{ratio_ds3:.2f}] | batch: {batch_ds3:8} | end: {epoch_ds3:6} | {batch_ds3*block_size*batch_size/1000**2:6.1f} Mtoken")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    print(f"  {temp_icon} PERFORMANCES |", end="")
    print(f" {temp_color}SPD : {sec_per_step:.2f} s/st{UI['RESET']} | {temp_color}DATA: {tok_s:,.0f} tok/s{UI['RESET']}")
    print(f"     ETA {target_train}: {eta_str}")
    print(f"{UI['GRAY']}"+f"━" * (LIGNE_LEN) + f"{UI['RESET']}")
    
    # Calcul d'Efficience du model (v. simple sinon ajouter Delta Loss/Delta LR)
#    efficiency, asymptote_new, trend_1k = calculate_saturation_score(steps, losses, lr, window=5000)
    efficiency, asymptote_new, trend_1k = calculate_saturation_score(steps_h, vals_h, lr_h, window=5000)
    status = f"⚡ {UI['BLUE']}PRODUCTIF {UI['RESET']}" if efficiency > 0.5 else f"🐢 {UI['RED']}SATURATION{UI['RESET']}"
    level_name_new, _,color_asympt_new = get_intel_level(asymptote_new)
    total_tokens = (batch_ds1 + batch_ds2 + batch_ds3) * block_size * batch_size   #steps[-1] * EBS * block_size
    pct = ratio_wiki
    w=UI['BLUE']
    c=UI['ORANGE']
    l=UI['GREEN']
    bar = f"{w}" +"▬" * (int(pct/6)-1) +f"{c}"+ "▬" * (int(ratio_cult/6))+f"{l}"+ "▬" * (int(ratio_litt/6)) #
    # Calcul du GAP (Sur-apprentissage)
    gap = losses[-1] - trains[-1]
    gap_h_last = gap_h[-1]
    gap_color = UI['GREEN'] if gap < 0.05 else (UI['YELLOW'] if gap < 0.12 else UI['RED'])
    graph = get_micro_graph(gap_h[-15:],-1)

    print(f"  {loss_icon} {loss_color}{loss_name}{UI['RESET']}      | LOSS: {loss_color}{losses[-1]:.3f}{UI['RESET']} | EMA: {ema_color}{ema_loss:.3f}{UI['RESET']}")
    print(f"  📚 SAVOIR ABSORBÉ : {total_tokens / 1e6:.2f} Millions de tokens")
    print(f"     {w}Wikipédia/{c}CulturaX/{l}Litteraire : {w}{ratio_wiki:2.0f}% / {c}{ratio_cult:2.0f}% / {l}{ratio_litt:2.0f}% | {w}{bar}{UI['RESET']}")  
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    print(f"  {status} | efficience: {efficiency:.2f} ", end="")
    print(f"| {gap_color}GAP: {gap:.3f}{UI['RESET']}  >  {graph}")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    
    # Predictions
    print(f"  🔮 PRÉDICTIONS ",end="")
    print(f" | TREND: {trend_1k:+.3f}/1k")
    print("    ",end="")
    for h in [100, 1000,  10000]:
        target = losses[-1] + (slope * h)
        print(f" | +{h:5} st ➔ \033[1m{max(2.0, target):.3f}\033[0m", end="")
    print(f"\n{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")

    # Prédictions améliorées ...
    # anciennes prédictions ...
    horizons = [100, 500]
    preds, asymptote = get_forecast(steps_h, vals_h, horizons)

    # NOUVELLE STATISTIQUE DE PROJECTION ...
    horizon = target_train
    proj = 10;
#    popt = predict_loss_projection(monitor_steps, monitor_losses, 0, proj*1)
    popt = predict_loss_projection(steps_h, vals_h, 0, proj)
    stats = check_efficiency(popt, steps[-1], horizon)
    status_plateau = f"{UI['RED']}\u2715 Non" if stats["is_plateau"] else f"{UI['GREEN']}\u2714 Yes !"
    asymptote = stats['loss_horizon']
    _, _, color_asympt = get_intel_level(asymptote)
    asymptote_str = f"{asymptote:.3f}"
    target_txt = ['🎯', 'ASYMPTOTE', asymptote_str]

    # Affichage de l'asymptote avec couleur
    print(f"  🎯 PRÉVISIONS : {horizon} | Slope 1k : {stats['current_slope_per_1000']:.3f} | Slope After : {UI['BOLD']}{status_plateau} {UI['RESET']}")
    if preds:
        i=1
        for h, val in preds.items():
            eta = timedelta(seconds=int(h * sec_per_step))
            txt = f"      +{h:5} steps ➔  \033[1m{val:.2f}\033[0m  (-> {eta})"
            pad = LIGNE_LEN - 16 - (len(txt)-8) 
            pad2 = (16 - len(target_txt[i])) // 2
            txt += (" "*pad) + "|" +(" "*pad2) + f"{color_asympt}" +target_txt[i]+ f"{UI['RESET']}"
            print(txt)
            i += 1
#        print(f"   🎯 Nouvelle prévision modèle final : {UI['BOLD']}{color_asympt_new}{asymptote_new:.2f} {level_name_new}{UI['RESET']}")
        print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    else:
        print("  (Collecte de données pour Scipy...)")

    # Graphique amélioré
    mini = losses[-40:] # On prend plus de points pour le graph
    m_min, m_max = min(mini), max(mini)
    print(f"  📉 LOSS History > ", end="")
    for v in mini:
        # On utilise des caractères de blocs pour simuler une courbe
        levels = [" "," ","▂","▃","▄","▅","▆","▇"]
        idx = int(((v - m_min) / (m_max - m_min + 1e-6)) * 7)
        print(levels[7-idx], end="")
    print(f"\n{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")

    print(f"  📦 CONFIG du training")
    params_ = (list(params))
    for  p1, p2, p3 in zip(params_[:5], params_[5:10], params_[10:]):
        txt = f"     {p1}: {params[p1]} "
        txt2= f"| {p2}: {params[p2]}"
        print(f"{txt}"+" "*(27 - len(txt))+ f"{txt2}" +" "*(25-len(txt2)) + f"| {p3}: {params[p3]}")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")

    print(f"  🔄 Effective Batch Size (EBS) : {UI['BOLD']}{EBS}{UI['RESET']}            | ",end="")
#    ema_model_status = '✅' if model_ema is not None else '❌'
    ema_model_status = f"{UI['GREEN']}\u2714" if model_ema is not None else f"{UI['RED']}\u2718"
    print(f"  {UI['BOLD']}{ema_model_status}{UI['RESET']} Model EMA")
    print(f"  📥 Taille du Model : {UI['BOLD']}{n_params/1e6:.1f}M{UI['RESET']} Paramètres")
    print(f"    ",end="")
    for i, (c,v) in enumerate(config.items()):
        txt = f" {c}: {v}"
        print(txt+" "*(22 - len(txt))+"|",end="")
        if i == 2: print("\n    ",end="")
    print(f"\n{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")

    #-----------  AFFICHE de Train/Run en cours -------------

#    {'status': 'RUNNING', 'last_update': '23:08:49', 'step': 10150, 'loss': 3.2715, 'lr': '2.93e-04', 'inter': 10, 'elapse': 161.98, 'ram': 81.2, 'swap': 0.6}
    d = data_monitor[-1]
    star = f"{UI['MAGENTA']}{UI['BOLD']}\u2605 {UI['RESET']}{UI['CYAN']}" if d['purg'] == 1 else ""
    update_icon = f"{UI['MAGENTA']}{UI['BOLD']}🔊 {UI['CYAN']}" if update['status'] == 1 else ""
    h, m , s = map(int,d['last_update'].split(':')) # - time.time()
    elapse = (( datetime.now() - datetime.now().replace(hour=h, minute=m,second=s) ).total_seconds())/ (d['elapse'])
    update['elapse'] = d['elapse']
    pattern = [' ','⡀','⡄','⡆','⡇']
    val = round(elapse*5)
    cpt = f"{pattern[val%5]}"*1
#    cpt = get_micro_graph([elapse],-0.05,1.05,-1,2)
#    elapse_time = datetime.strptime(d['last_update'],"%HH:MM:SS").time() # - time.time()
#    print( cpt)
    loss_avg = np.mean([d['loss'] for d in data_monitor[-100:]]) # *(1 - config['dropout'])
    loss_avg = calcul_ema([ v['loss'] for v in data_monitor ])
#    arrow = f"{UI['B_OR']}\u2191{UI['CYAN']}" if (d['loss'] > loss_avg) else f"{UI['B_MAG']}\u2193{UI['CYAN']}"
#    arrow = f"{UI['B_OR']}\u2934{UI['CYAN']}" if (d['loss'] > loss_avg) else f"{UI['B_MAG']}\u2935{UI['CYAN']}"
    arrow = f"{UI['B_OR']}\u2b06{UI['CYAN']}" if (d['loss'] > loss_avg) else f"{UI['B_MAG']}\u2b07{UI['CYAN']}"
#    print(f"{UI['CYAN']}   {star}{update_icon}STEP: {d['step']} | SPD : {(d['elapse']/d['inter']):.2f}/st | LOSS train : {d['loss']:.3f} {arrow} | EMA : {loss_avg:.3f}{UI['RESET']} {cpt}")
    print(f"{UI['CYAN']}   {star}STEP: {d['step']}    | SPD : {(d['elapse']/d['inter']):.2f}/st | LOSS train : {d['loss']:.3f} {arrow} | EMA : {loss_avg:.3f}{UI['RESET']}")
    # Gard Norm en live ...
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    print(f"   {gn_status}{UI['CYAN']}  | Evolution du Grad Norm. valeur Maxi  : [\u2191 {gn_max:.3f}]")    

    # Step dynamique (incrémente en temps réel)
    pct = p_step * 100.
    bar = "▬" * (int(pct/4)-1)+ f"{UI['RED']}▬" +f"{UI['GRAY']}"+ "┅" * (25 - int(pct/4))
    running_step = current_interval // data_monitor[-1]['inter'] 
    next_run_ecart = ((current_interval * (-sec_per_step + np.mean([d['elapse'] for d in data_monitor[-running_step:]])/data_monitor[-1]['inter'])))
    
    run_ecart = f"{UI['MAGENTA']}(-" if next_run_ecart > 0 else f"{UI['ORANGE']}(+"
    run_ecart += f"{datetime.fromtimestamp(abs(next_run_ecart)).strftime('%M:%S')}"
    next_run = datetime.fromtimestamp(last_update + current_interval * sec_per_step)

    print(f"{UI['CYAN']}   {current_interval:3d} stp/run  {UI['RED'] if p_step > 0.95 else UI['CYAN']} ┣{bar}┫ {UI['RESET']}{UI['CYAN']}   {next_run.strftime('%H:%M')} {run_ecart}){UI['RESET']}")  # ┣{bar}┫ {UI['DIM']")
    # print(f"{U['D']}Log: {curr['step']} (+{int(steps_since_log)}{U['RE']}")
    
#    padding = (LIGNE_LEN - 5)//2
#    print(f" "*padding+f"{next_run.strftime('%H:%M')}{UI['RESET']}")
    # print(f"{U['D']}"+f" "*padding+f"{datetime.now().strftime('%H:%M:%S')}{U['RE']}")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
#    print(d)
#    print(update)
#    print(ratio_ds1, ratio_ds2, ratio_ds3, EBS)
#    print(stats)

#-----------------------

def get_last_modified(file):
    try:
        return os.path.getmtime(file)
    except OSError:
        return 0

last_time = get_last_modified(MONITOR_FILE)
last_time_log = get_last_modified(LOG_FILE)

print(f"📡 Monitoring réactif activé sur {MONITOR_FILE}")
timeout_pattern = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
timeout = 0;
while True:
    current_time_M = get_last_modified(MONITOR_FILE)
    current_time_L = get_last_modified(LOG_FILE)
    current_time = max(current_time_M, current_time_L)
    seconds_since_update = time.time() - current_time
    progression = seconds_since_update / max(1, update['elapse'])
    timeout = (timeout + 1) % len(timeout_pattern) 
#    print(f"⏳ Progression estimée du step : {min(100, progression*100):.1f}%",end="\r")
    print(f"   {timeout_pattern[timeout]}  Progression estimée du step : {UI['RED'] if progression>0.95 else UI['RESET']}{min(100, progression*100):.1f}%     ",end="\r")
    # Si le fichier a été modifié depuis la dernière vérification
    if ( (current_time > last_time) | (first==True) ):
        try:
            time.sleep(3)
            print("updating data in progress ...                         ")
            get_dashboard()
            last_time = current_time # On met à jour le marqueur
        except Exception as e:
            print(f'error: {e}')
        
        gc.collect()
    
    # On garde un petit sleep très court pour ne pas saturer le CPU
    # 1 seconde suffit pour être quasi instantané par rapport à l'entraînement
    time.sleep(1)
