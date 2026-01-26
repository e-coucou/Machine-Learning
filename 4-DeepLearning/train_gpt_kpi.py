import os, time, json, re
import numpy as np
import datetime

# --- CONFIG ---
LOG_FILE = "model/training.log"
HISTORY_FILE = "model/my_wiki_history.json"
REFRESH_RATE = 5
FINAL_STEP = 75000
SIZE = 20

U = {"G": "\033[92m", "Y": "\033[93m", "R": "\033[91m", "C": "\033[96m", 
     "B": "\033[1m", "RE": "\033[0m", "D": "\033[2m", "W": "\033[97m"}
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

def get_gauge(value, min_val, max_val, bars=10):
    # Inverse : plus la valeur est petite, plus la barre est pleine (pour la Loss)
    percent = max(0, min(1, (max_val - value) / (max_val - min_val)))
    filled = int(percent * bars)
    return f"[{'■'*filled}{' '*(bars-filled)}]"

def parse_logs():
    steps, losses, times = [], [], []
    
    # 1. Liste des fichiers historiques à scanner (dans l'ordre)
    # On cherche training_01.log, training_02.log, etc.
    history_files = sorted([f for f in os.listdir('model/') if re.match(r'training_\d+\.log', f)])
    all_log_files = ['model/' + f for f in history_files] + [LOG_FILE]
    
    pattern = r"step (\d+): .*val loss ([\d.]+), .* \(([\d.]+)s\)"
    
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
                        losses.append(float(match.group(2)))
                        times.append(float(match.group(3)))
                        
    return np.array(steps), np.array(losses), np.array(times)

def get_dual_log_data():
    if not os.path.exists(LOG_FILE): return None
    try:
        mtime = os.path.getmtime(LOG_FILE)
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            results = []
            for line in reversed(lines):
                # m = re.search(r"step (\d+): .*loss ([\d.]+), .* \(([\d.]+)s\)", line)
                m = re.search(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+), .* \(([\d.]+)s\)", line)
                if m:
                    results.append({
                        'step': int(m.group(1)), 
                        'train': float(m.group(2)), 
                        'val': float(m.group(3)), 
                        'time': float(m.group(4))
                    })
                if len(results) == 2: break
            return results, mtime
    except: pass
    return None

def get_latest_data():
    if not os.path.exists(LOG_FILE): return None
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            for line in reversed(lines):
                m = re.search(r"step (\d+): .*loss ([\d.]+), .* \(([\d.]+)s\)", line)
                if m: return int(m.group(1)), float(m.group(2)), float(m.group(3))
    except: pass
    return None

def get_color(idx,a,b,c,d,e):
    if idx < a:
        return "🟣", UI["MAGENTA"], "-----"
    elif idx < b:
        return "🔵", UI["BLUE"], "GLACE"
    elif idx < c:
        return "🟢", UI["GREEN"], "FROID"
    elif idx < d:
        return "🟡", UI["YELLOW"], "CHAUD"
    elif idx < e:
        return "🟠", UI["RED"], "CHAUD"
    else:
        return "🔴", UI["RED"], "SURCHAUFFE"

def run_monitor():
    while True:
        data = get_latest_data()
        result, last_update = get_dual_log_data()
        if data == None:
            steps_all, losses_all, times_all = parse_logs()
            data = steps_all[-1], losses_all[-1], times_all[-1]
            result = [{'step':steps_all[-1], 'train':0.,'val':losses_all[-1],'time':times_all[-1] }]
            last_update = os.path.getmtime(LOG_FILE)
        if data:
            curr = result[0]
            # Calcul dynamique de l'intervalle (ex: 100) et du SPS réel
            if len(result) == 2:
                step_interval = curr['step'] - result[-1]['step']
                sps = curr['time'] / step_interval
            else:
                step_interval = 100 # Fallback
                sps = curr['time'] / 100            
            step, loss, duration = data
            # sps = duration / step_interval
            
            # Calcul du GAP (Sur-apprentissage)
            gap = curr['val'] - curr['train']
            
            # Calcul du progrès interne
            elapsed = time.time() - last_update
            steps_since_log = elapsed / sps
            next_step = curr['step'] + step_interval
            current_step_est = curr['step'] + int(steps_since_log)
            p_step = 1 - (next_step - current_step_est)/step_interval
            
            os.system('clear' if os.name == 'posix' else 'cls')

            # --- RENDU VISUEL SIZE CARACTÈRES ---
            print("-" * SIZE)
            print(f"{U['C']} 🚀 GPT live{U['RE']}")
            print("-" * SIZE)
            print(f"STEP: {U['B']}{step}{U['RE']}")
            
            # Jauge de Loss (Cerveau) - Cible 2.4 à 3.5
            print(f"{U['C']}BRAIN (Loss){U['RE']}")
            print(f"{loss:.4f}")
            _,color_loss,_ = get_color(loss,2.4,2.6,2.8,3.2,3.5)
            print(f"{color_loss}{get_gauge(loss, 1.8, 4.0,bars=SIZE-2)}{U['RE']}")
            
            # Jauge de Vitesse (SPS) - Cible 6.0 à 9.0
            print(f"{U['C']}SPEED (Sps){U['RE']}")
            print(f"{sps:.2f} s/st")
            _,color_sps,_ = get_color(sps,6,6.5,7,7.5,8)
            print(f"{color_sps}{get_gauge(sps, 6.0, 9.0,bars=SIZE-2)}{U['RE']}")
            
            # -GAP Val Loss vs Train Loss
            # Vert si < 0.05, Jaune si < 0.12, Rouge si > 0.12
            gap_color = U['G'] if gap < 0.05 else (U['Y'] if gap < 0.12 else U['R'])
            print(f"{U['B']}GAP (Overfit){U['RE']}")
            print(f"{gap_color}{gap:+.4f}{U['RE']}")            
            
            # Barre de progression globale
            print("-" * SIZE)
            prog_p = (step / FINAL_STEP)
            print(f"TRAINING: {prog_p*100:.1f}%")
            # print(f"{U['B']}[{'#'*int(prog_p*11)}{' '*(SIZE-int(prog_p*SIZE))}]{U['RE']}")
            print(f"{U['B'] if prog_p < 0.9 else UI['BLUE']}{get_gauge(1-prog_p, 0, 1.0,bars=SIZE-2)}{U['RE']}")
            print("-" * SIZE)
            
            # Step dynamique (incrémente en temps réel)
            print(f"{U['R'] if p_step > 0.90 else U['D']}{get_gauge(p_step,0,1.01,bars=SIZE-2)}") #{U['RE']}")
            # print(f"{U['D']}Log: {curr['step']} (+{int(steps_since_log)}{U['RE']}")
            
            padding = (SIZE - 5)//2
            next_run = datetime.datetime.fromtimestamp(last_update + step_interval * sps)
            print(f" "*padding+f"{next_run.strftime('%H:%M')}{U['RE']}")
            # print(f"{U['D']}"+f" "*padding+f"{datetime.now().strftime('%H:%M:%S')}{U['RE']}")

        time.sleep(REFRESH_RATE)

if __name__ == "__main__":
    try: run_monitor()
    except KeyboardInterrupt: print("\nFin.")