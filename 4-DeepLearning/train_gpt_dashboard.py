import json, re, time, os, psutil, subprocess, json, math
import numpy as np
import torch, gc
from datetime import timedelta, datetime
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

# --- CONFIGURATION (À vérifier dans ton script d'entraînement) ---
VERSION = 'v5.0'
LOG_FILE = 'model/training.log'
MONITOR_FILE = 'model/monitor.log'
HISTORY_FILE = 'model/my_wiky_history.json'
MODEL_FILE = 'model/my_wiki.pth'
BATCH_SIZE = 128   # au réel 32 Batch_size x 4 grad_accum 
BLOCK_SIZE = 256   
TARGET_STEP = 46589 # Nombre total de steps pour 1 epoch : 5963390 block de 256 /(32*4) = 46589
TARGET_TRAIN = 80000
LIGNE_LEN = 67
TARGET_BLOCK_DS1 = 5963390
TARGET_BLOCK_DS2 = 2429390

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

def calcul_epoch(step, target, mix, batch, grad, t_step):
    return int((target - step * batch) / (batch * mix * grad) + t_step)
    
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
    ckpt = torch.load(path, map_location='cpu', mmap=True)
    wiki = ckpt.get('total_step_wiki', 0)
    cult = ckpt.get('total_step_cult', 0)
    step = ckpt.get('total_steps_done', 0)
    config = ckpt.get('config',None)
    params = ckpt.get('params',None)
    model_ema = ckpt.get('model_ema',None)
    n_params = sum(t.numel() for t in ckpt['model'].values() if isinstance(t, torch.Tensor))
    del ckpt
    return step, wiki, cult, config, model_ema, n_params, params

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
    
    status = "🟢" if pression < 70 else "🟡" if pression < 85 else "🔴"

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
        
        # On cherche la ligne "GPU Active residency"  GPU HW active residency
        match = re.search(r"GPU HW active residency:\s+([\d.]+)%", result)
        if match:
            gpu = float(match.group(1))
            status = "🔴" if gpu < 50 else "🟡" if gpu < 90 else "🟣"
            color = "\033[91m" if gpu < 50 else "\033[93m" if gpu < 90 else "\033[95m"
            return gpu, status, color
    except Exception:
        return 0.0, "❌", "\033[96m"
    return 0.0, "❌", "\033[96m"

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
    steps, losses, times, lr, trains = [], [], [], [], []
    data = []
    if not os.path.exists(monitor_log): return data
    with open(monitor_log, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
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
    if ema_loss > 4.0: return "CHAOS.    ", "🔴" , "\033[91m"
    if ema_loss > 3.2: return "PHONÉTIQUE", "🟠", "\033[38;5;208m"
    if ema_loss > 2.90: return "SYNTAXIQUE", "🟡", "\033[93m"
    if ema_loss > 2.60: return "PERROQUET ", "🟢", "\033[92m"
    if ema_loss > 2.45: return "ÉTUDIANT. ", "🔵", "\033[94m"
    if ema_loss > 2.35: return "EXPERT.   ", "🟣", "\033[95m"
    return "MÉMORISATION", "🔥", "\033[1;38;5;201m"

def get_speed_level(s_step):
    if s_step < 14.0:
        return "🟣", UI["MAGENTA"], "-----"
    elif s_step < 15.0:
        return "🔵", UI["BLUE"], "GLACE"
    elif s_step < 16.0:
        return "🟢", UI["GREEN"], "FROID"
    elif s_step < 17.0:
        return "🟡", UI["YELLOW"], "CHAUD"
    elif s_step < 18.0:
        return "🟠", UI["ORANGE"], "BOUILLANT"
    else:
        return "🔴", UI["RED"], "SURCHAUFFE"

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

def get_dashboard():
    steps, losses, times, lr, trains = parse_logs(LOG_FILE)
    last_update = os.path.getmtime(LOG_FILE)
    steps_h, trains_h, vals_h, times_h = parse_history(HISTORY_FILE)
    gap_h = vals_h - trains_h
    data_monitor = parse_monitor(MONITOR_FILE)
   
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
#    pct_ds1 = (data_monitor[-1]['dataset1'] * batch_size/TARGET_BLOCK_DS1) * 100
#    pct = (steps[-1] / TARGET_STEP) * 100 
#    epoch = f"[epoch {int(pct/100 + 1)}]"
#    if pct > 100:
#        pct = pct % 100
#    bar = "▬" * (int(pct_ds1/4)-1)+ f"{UI['RED']}▬" +UI['DIM']+UI['WHITE']+ "┅" * (25 - int(pct_ds1/4)) #. ─
    # bar = "█" * int(pct/2) + "░" * (50 - int(pct/2))
    pct_train = (steps[-1] / TARGET_TRAIN) * 100
    bar_train = "▬" * (int(pct_train/4)-1)+ f"{UI['RED']}▬" +UI['DIM']+UI['WHITE']+ "┅" * (25 - int(pct_train/4)) #. "▄"

    # Alerte Thermique (basée sur le temps de cycle)
    # Si le Mac met plus de 195s pour 25 steps, on affiche en Orange/Rouge
    # temp_icon = "🌡️ "
    temp_icon, temp_color, temp_text = get_speed_level(sec_per_step)

    # Calcul du progrès interne
    elapsed = time.time() - last_update
    steps_since_log = elapsed / sec_per_step
    next_step = steps[-1] + current_interval
    current_step_est = steps[-1] + int(steps_since_log)
    p_step = 1 - (next_step - current_step_est)/current_interval

    file = MODEL_FILE
    status, pression, mem_rss, swap = get_memory_status()
    step, wiki, cult, config, model_ema, n_params, params = get_checkpoint(file)
    batch_size = params.get('batch_size',32)
    block_size = params.get('block_size',BLOCK_SIZE)
    grad_accum_steps = params['grad_accum_steps']
    EBS = batch_size * grad_accum_steps
    ratio_wiki = wiki/(wiki+cult)*100
    ratio_cult = cult/(wiki+cult)*100
    remaining_steps = TARGET_TRAIN - steps[-1]
    eta_seconds = remaining_steps * sec_per_step
    eta_str = str(timedelta(seconds=int(eta_seconds)))

    ssd_usage, ssd_status, ssd_color = get_ssd_usage()        
    cpu_usage, cpu_status, cpu_color = get_cpu_usage()
    gpu_usage, gpu_status, gpu_color = get_gpu_usage()

    lr_h = []
    for it in steps_h:
        lr_h.append(calcul_lr(it, params))
    lr_h = np.array(lr_h)

 
  # Progression de l'entrainement epoch et steps total
    pct_ds1 = (data_monitor[-1]['dataset1'] * batch_size/TARGET_BLOCK_DS1) * 100
    pct_ds2 = (data_monitor[-1]['dataset2'] * batch_size/TARGET_BLOCK_DS2) * 100
    pct = (steps[-1] / TARGET_STEP) * 100 
    epoch = f"[epoch {int(pct/100 + 1)}]"
    if pct > 100:
        pct = pct % 100
    bar_ds1 = "▬" * (int(pct_ds1/4)-1)+ f"{UI['RED']}▬" +UI['DIM']+UI['CYAN']+ "┅" * (25 - int(pct_ds1/4))
    bar_ds2 = "▬" * (int(pct_ds2/4)-1)+ f"{UI['RED']}▬" +UI['DIM']+UI['CYAN']+ "┅" * (25 - int(pct_ds2/4))

    epoch_ds1 = calcul_epoch(data_monitor[-1]['dataset1'] , TARGET_BLOCK_DS1, (1-params.get('mixed_ratio',0.4)), batch_size, grad_accum_steps, data_monitor[-1]['step'])
    epoch_ds2 = calcul_epoch(data_monitor[-1]['dataset2'] , TARGET_BLOCK_DS2, (params.get('mixed_ratio',0.4)), batch_size, grad_accum_steps, data_monitor[-1]['step'])

    #--------------------------------
    # --- AFFICHAGE DU DASHBOARD --- ---------------------------------------------------------------------------
    #--------------------------------
    
    os.system('clear')
    Titre = f"🚀 M1 GPT-MONITOR {VERSION}. - {steps[-1]}/{current_interval} - {epoch}"
    timestamp = datetime.now().strftime('%H:%M:%S')
    header_right = f"{temp_color}{temp_icon} \033[1m{timestamp}\033[0m" 
    visible_right_len = len("🌡️ ") + len(timestamp)
    padding = LIGNE_LEN - len(Titre) - visible_right_len-1
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    print(f"{temp_color}\033[1m{Titre}\033[0m" + (" " * padding) + header_right)

#    get_gpu_usage()
    # Affichage de l'entrainement en cours ...
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    print(f"  {status} RAM Système: {pression}% | Process: {mem_rss:.0f}Mo | Swap: {swap:.0f}Mo")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    print(f"  {cpu_status} CPU Usage: {cpu_color}{cpu_usage:.1f}% {UI['RESET']} | {gpu_status} GPU Usage: {gpu_color}{gpu_usage:.1f}% {UI['RESET']} | {ssd_status} SSD Usage {ssd_color}{ssd_usage:.1f}% ")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
#    print(f"  Epoch: {pct:4.1f}% {UI['CYAN']}┣{bar}┫ {UI['RESET']}pour {TARGET_STEP} steps")
    print(f"     Train: {pct_train:4.1f}% {UI['CYAN']}┣{bar_train}┫ {UI['RESET']}pour {TARGET_TRAIN} steps")
    print(f"     Wiki : {pct_ds1:4.1f}% {UI['CYAN']}┣{bar_ds1}┫ {UI['RESET']}pour {TARGET_BLOCK_DS1:7d} blocks")
    print(f"     Cult : {pct_ds2:4.1f}% {UI['CYAN']}┣{bar_ds2}┫ {UI['RESET']}pour {TARGET_BLOCK_DS2:7d} blocks")
    print(f"     Wikipédia  | batch: {wiki} | end: {epoch_ds1} | {wiki*BLOCK_SIZE/1000**2:.1f} Mtoken")
    print(f"     CulturaX   | batch: {cult} | end: {epoch_ds2} | {cult*BLOCK_SIZE/1000**2:.1f} Mtoken")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    print(f"  {temp_icon} PERFORMANCES")
    print(f"     {temp_color}SPD : {sec_per_step:.2f} s/st{UI['RESET']} | DATA: {tok_s:,.0f} tok/s")
    print(f"     ETA {TARGET_TRAIN}: {eta_str}")
    print(f"{UI['GRAY']}"+f"━" * (LIGNE_LEN) + f"{UI['RESET']}")
    

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

    # Calcul d'Efficience du model (v. simple sinon ajouter Delta Loss/Delta LR)
#    efficiency, asymptote_new, trend_1k = calculate_saturation_score(steps, losses, lr, window=5000)
    efficiency, asymptote_new, trend_1k = calculate_saturation_score(steps_h, vals_h, lr_h, window=5000)
    status = f"⚡ {UI['BLUE']}PRODUCTIF {UI['RESET']}" if efficiency > 0.5 else f"🐢 {UI['RED']}SATURATION{UI['RESET']}"
    level_name_new, _,color_asympt_new = get_intel_level(asymptote_new)
    total_tokens = steps[-1] * EBS * BLOCK_SIZE
    pct = ratio_wiki
    w=UI['BLUE']
    c=UI['ORANGE']
    bar = f"{w}" +"▬" * (int(pct/5)-1) +f"{c}"+ "▬" * (20 - int(pct/5)) #
    # Calcul du GAP (Sur-apprentissage)
    gap = losses[-1] - trains[-1]
    gap_h_last = gap_h[-1]
    gap_color = UI['GREEN'] if gap < 0.05 else (UI['YELLOW'] if gap < 0.12 else UI['RED'])
    graph = get_micro_graph(gap_h[-30:],-1)

    print(f"  {loss_icon} {loss_color}{loss_name}{UI['RESET']}    | LOSS: {loss_color}{losses[-1]:.3f}{UI['RESET']} | EMA: {ema_color}{ema_loss:.3f}{UI['RESET']}")
    print(f"  📚 SAVOIR ABSORBÉ : {total_tokens / 1e6:.2f} Millions de tokens")
    print(f"     Ratio {w}Wikipédia/{c}CulturaX : {w}{ratio_wiki:.1f}% / {c}{ratio_cult:.1f}% | {w}{bar}{UI['RESET']}")  
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    print(f"  {status} | ratio efficience: {efficiency:.2f}")
    print(f"     {gap_color}GAP: {gap:.3f}{UI['RESET']} | {graph}")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    
    # Predictions
    print(f"  🔮 PRÉDICTIONS ",end="")
    print(f" | TREND: {trend_1k:+.3f}/1k")
    print("    ",end="")
    for h in [1000, 5000,  10000]:
        target = losses[-1] + (slope * h)
        print(f" | +{h:5} st ➔ \033[1m{max(2.0, target):.3f}\033[0m", end="")
    print(f"\n{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")

    # Prédictions améliorées ...
    horizons = [1000, 5000, 10000]
    # On passe les 200 derniers points pour donner plus de contexte à Scipy
    #preds, asymptote = get_forecast(steps[-200:], losses[-200:], horizons)
    # finalement on envoie tout !
#    preds, asymptote = get_forecast(steps, losses, horizons)
    preds, asymptote = get_forecast(steps_h, vals_h, horizons)
    _, _, color_asympt = get_intel_level(asymptote)
    asymptote_str = f"{asymptote:.3f}"
    target_txt = ['🎯', 'PLANCHER', 'THÉORIQUE', asymptote_str]
    # Affichage de l'asymptote avec couleur
    print(f"  🔮 PRÉVISIONS (Scaling Law - Scipy)              |       🎯")
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
        print(f"   🎯 Nouvelle prévision modèle final : {UI['BOLD']}{color_asympt_new}{asymptote_new:.2f} {level_name_new}{UI['RESET']}")
        print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
    else:
        print("  (Collecte de données pour Scipy...)")

    # Graphique amélioré
    mini = losses[-40:] # On prend plus de points pour le graph
    m_min, m_max = min(mini), max(mini)
    print(f"  📉 HISTORY (Last 40): ", end="")
    for v in mini:
        # On utilise des caractères de blocs pour simuler une courbe
        levels = [" "," ","▂","▃","▄","▅","▆","▇"]
        idx = int(((v - m_min) / (m_max - m_min + 1e-6)) * 7)
        print(levels[7-idx], end="")
    print(f"\n{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")

    print(f"  📦 CONFIG du training")
    params_ = (list(params))
    for  p, p2, c in zip(params_[:6], params_[6:], config):
        txt = f"     {c}: {config[c]} "
        txt2= f"| {p}: {params[p]}"
        print(f"{txt}"+" "*(21 - len(txt))+ f"{txt2}" +" "*(23-len(txt2)) + f"| {p2}: {params[p2]}")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")

    print(f"  🔄 Effective Batch Size (EBS) : {UI['BOLD']}{EBS}{UI['RESET']}")
    print(f"  📥 Taille du Model : {n_params/1e6:.2f}M Paramètres")
    ema_model_status = '✅' if model_ema is not None else '❌'
    print(f"  {ema_model_status} Model EMA")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")

    #-----------  AFFICHE de Train/Run en cours -------------

#    {'status': 'RUNNING', 'last_update': '23:08:49', 'step': 10150, 'loss': 3.2715, 'lr': '2.93e-04', 'inter': 10, 'elapse': 161.98, 'ram': 81.2, 'swap': 0.6}
    d = data_monitor[-1]
    star = f"{UI['MAGENTA']}{UI['BOLD']}\u2605 {UI['CYAN']}" if d['purg'] == 1 else ""
    h, m , s = map(int,d['last_update'].split(':')) # - time.time()
    elapse = (( datetime.now() - datetime.now().replace(hour=h, minute=m,second=s) ).total_seconds())/ (d['elapse'])
    cpt = get_micro_graph([elapse],-0.05,1.05,-1,2)
#    elapse_time = datetime.strptime(d['last_update'],"%HH:MM:SS").time() # - time.time()
#    print( cpt)
    loss_avg = np.mean([d['loss'] for d in data_monitor[-100:]]) # *(1 - config['dropout'])
    loss_avg = calcul_ema([ v['loss'] for v in data_monitor ])
    arrow = f"{UI['B_OR']}\u2191{UI['CYAN']}" if (d['loss'] > loss_avg) else f"{UI['B_MAG']}\u2193{UI['CYAN']}"
    print(f"{UI['CYAN']}   {star}STEP: {d['step']} | SPD : {(d['elapse']/d['inter']):.2f}/st | LOSS : {d['loss']:.4f} {arrow} | AVG : {loss_avg:.4f}{UI['RESET']} {cpt}")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")

    # Step dynamique (incrémente en temps réel)
    pct = p_step * 100.
    bar = "▬" * (int(pct/4)-1)+ f"{UI['RED']}▬" +f"{UI['GRAY']}"+ "┅" * (25 - int(pct/4))
    running_step = current_interval // data_monitor[-1]['inter'] 
    next_run_ecart = ((current_interval * (-sec_per_step + np.mean([d['elapse'] for d in data_monitor[-running_step:]])/data_monitor[-1]['inter'])))
    
    run_ecart = f"{UI['MAGENTA']}(-" if next_run_ecart < 0 else f"{UI['ORANGE']}(+"
    run_ecart += f"{datetime.fromtimestamp(abs(next_run_ecart)).strftime('%M:%S')}"
    next_run = datetime.fromtimestamp(last_update + current_interval * sec_per_step)

    print(f"{UI['CYAN']}   Run de {current_interval} steps  {UI['RED'] if p_step > 0.95 else UI['CYAN']} ┣{bar}┫ {UI['RESET']}{UI['CYAN']}   {next_run.strftime('%H:%M')} {run_ecart}){UI['RESET']}")  # ┣{bar}┫ {UI['DIM']")
    # print(f"{U['D']}Log: {curr['step']} (+{int(steps_since_log)}{U['RE']}")
    
#    padding = (LIGNE_LEN - 5)//2
#    print(f" "*padding+f"{next_run.strftime('%H:%M')}{UI['RESET']}")
    # print(f"{U['D']}"+f" "*padding+f"{datetime.now().strftime('%H:%M:%S')}{U['RE']}")
    print(f"{UI['GRAY']}"+f"─" * LIGNE_LEN+f"{UI['RESET']}")
#    print(d)

while True:
    try: get_dashboard()
    except: 
        print('error:')
    gc.collect()
    time.sleep(30)
