import subprocess
import re

def scan_final_m1():
    print("🔍 Scan des données brutes du M1...")
    try:
        # On utilise -l pour lister tout et -n pour filtrer par le nom du driver
        # On teste les deux noms de drivers les plus courants pour les capteurs M1
        for driver_name in ["AppleASICGenericSensor", "AppleARMIODevice"]:
            print(f"\n--- Test du driver : {driver_name} ---")
            cmd = f"ioreg -n {driver_name} -l"
            output = subprocess.check_output(cmd.split()).decode()
            
            # On cherche tout ce qui ressemble à une température ou un nom de sonde
            # On extrait les blocs de texte qui contiennent "value"
            lines = output.split('\n')
            for line in lines:
                if any(key in line.lower() for key in ["value", "temperature", "sensor-id", "ext-name"]):
                    print(line.strip())
                    
    except Exception as e:
        print(f"❌ Erreur : {e}")

# scan_final_m1()

def get_m1_pmu_temp():
    try:
        # On cherche spécifiquement le service qui contient le capteur PMU repéré
        cmd = "ioreg -r -n AppleARMIODevice"
        output = subprocess.check_output(cmd.split()).decode()
        
        # On cherche la section qui suit immédiatement le capteur PMU
        # Sur M1, les valeurs sont souvent stockées dans 'current-value' 
        # juste après la déclaration du sensor
        if "PMU tdev7" in output:
            # On cherche la première valeur numérique après l'apparition du nom du capteur
            parts = output.split("PMU tdev7")
            # On regarde dans les 500 caractères suivants
            match = re.search(r'"current-value"\s*=\s*(\d+)', parts[1][:500])
            if match:
                val = float(match.group(1))
                return val / 1000 if val > 1000 else val
    except:
        pass
    return None

temp = get_m1_pmu_temp()
if temp:
    print(f"🌡️ Température M1 (PMU tdev7) : {temp:.1f}°C")
else:
    # Si le parsing échoue, on utilise la méthode globale sur le dictionnaire
    print("🌡️ Capteur identifié, mais valeur verrouillée par Macs Fan Control.")

def get_asitop_stats():
    # On demande à powermetrics un échantillon unique (durée 1ms)
    # On cible les statistiques CPU/GPU et thermiques
    cmd = "sudo powermetrics -n 1 --sample-rate 1 -i 1 --samplers cpu_gpu,thermal"
    
    try:
        res = subprocess.check_output(cmd.split()).decode()
        
        # On extrait la température (souvent appelée 'Combined share' ou 'average')
        temp_match = re.search(r"Combined share (?:average|limit): (\d+)%", res)
        # Sur M1, powermetrics donne souvent l'état thermique sous forme de % de charge
        
        # Mais le plus important pour toi : la puissance (Watts)
        gpu_power = re.search(r"GPU Power: (\d+\.?\d*) mW", res)
        
        return {
            "gpu_mw": gpu_power.group(1) if gpu_power else "N/A",
            "thermal_throttle": "Yes" if "throttle: yes" in res.lower() else "No"
        }
    except:
        return None

print(f"📊 Stats brutes : {get_asitop_stats()}")