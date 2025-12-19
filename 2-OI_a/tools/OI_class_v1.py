import pandas as pd
import numpy as np
from urllib.parse import quote
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from functools import wraps

class DataProcessor_v1:
    def __init__(self, url_base, tags_other, tags_selected, start, end, 
                 interval='PT20M', hS='00', hF='23', cred_file="../../../cred.txt", 
                 verbose=True):
        self.url_base = url_base
        self.tags_other = tags_other
        self.tags_selected = tags_selected
        self.start, self.end = start, end
        self.interval = interval
        self.hS, self.hF = hS, hF
        self.verbose = verbose
        
        # Pipeline pour stocker les étapes de calcul
        self.pipeline = [] # Liste de dictionnaires {'func': nom_methode, 'args': [], 'kwargs': {}}

        # Gestion des credentials
        try:
            with open(cred_file, "r") as f:
                self.credentials = f.read().strip()
        except FileNotFoundError:
            print(f"[ERREUR] Fichier credentials introuvable : {cred_file}")
            self.credentials = ""

        self.rename_mapping = {item['tag']: item['nom'] for item in tags_selected}
        self.df = None    
        self.data = None

    # --- MÉCANISME DE PIPELINE ---
    def register_step(func):
        """Décorateur pour enregistrer une action dans le pipeline."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # On exécute la fonction
            result = func(self, *args, **kwargs)
            # On enregistre l'étape si elle n'est pas déjà dans le pipeline (évite les doublons au recalcul)
            # Note: on n'enregistre que si on n'est pas déjà en train de "recalculer"
            if not getattr(self, '_is_recalculating', False):
                self.pipeline.append({'func': func.__name__, 'args': args, 'kwargs': kwargs})
                self._log(f"Étape enregistrée : {func.__name__}")
            return result
        return wrapper

    def recalculate(self):
        """Recharge les données brutes et réapplique tout le pipeline de traitement."""
        self._log("=== DÉBUT DU RECALCUL GLOBAL ===")
        self._is_recalculating = True # Drapeau pour éviter d'enregistrer les étapes en double
        
        try:
            # 1. Nouveau Merge
            self.merge()
            
            # 2. Réapplication des étapes enregistrées
            for step in self.pipeline:
                func_name = step['func']
                args = step['args']
                kwargs = step['kwargs']
                
                method = getattr(self, func_name)
                method(*args, **kwargs)
                
            self._log(f"=== RECALCUL TERMINÉ ({len(self.pipeline)} étapes appliquées) ===")
        finally:
            self._is_recalculating = False
        return self

    def clear_pipeline(self):
        """Efface toutes les étapes enregistrées."""
        self.pipeline = []
        self._log("Pipeline vidé.")

    # --- MÉTHODES EXISTANTES MODIFIÉES ---

    def _log(self, message):
        if self.verbose: print(f"[INFO] {message}")

    def merge(self):
        self._log(f"Récupération : {self.start} au {self.end} (Intervalle: {self.interval})")
        tags = self.tags_other + list(self.rename_mapping.keys())
        df_list = []

        for tag in tags:
            encoded_tag = quote(tag, safe=':/?#[]@!$&\'()*+,;=')
            url = (f"{self.url_base}data-reference={encoded_tag}&aggregation=TIME"
                   f"&aggregation-function=MEAN&from={self.start}T{self.hS}%3A00%3A00.000Z"
                   f"&to={self.end}T{self.hF}%3A59%3A59.000Z&aggregation-period={self.interval}")
            try:
                d_data = pd.read_json(url, storage_options={'Authorization': f'basic {self.credentials}'})
                if 'values' in d_data and len(d_data['values']) > 0:
                    temp_df = pd.DataFrame(d_data['values'][0])
                    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
                    temp_df.set_index('timestamp', inplace=True)
                    temp_df.rename(columns={'value': tag}, inplace=True)
                    df_list.append(temp_df)
            except Exception as e:
                print(f"[ATTENTION] Erreur tag '{tag}': {e}")

        if df_list:
            self.df = pd.concat(df_list, axis=1)
            self.data = self.df.rename(columns=self.rename_mapping)
            self._log(f"Fusion terminée : {self.data.shape}")
        return self

    @register_step
    def filtering(self, tag, min_val, max_val, na=None):
        """Filtre les données (gère les listes de tags et de valeurs)."""
        if self.data is None: 
            return self
        
        # Conversion systématique en listes pour pouvoir itérer avec zip
        tags = [tag] if isinstance(tag, str) else tag
        mins = [min_val] * len(tags) if not isinstance(min_val, list) else min_val
        maxs = [max_val] * len(tags) if not isinstance(max_val, list) else max_val

        # Vérification de sécurité
        if len(tags) != len(mins) or len(tags) != len(maxs):
            print("[ERREUR] Le nombre de tags ne correspond pas au nombre de valeurs min/max.")
            return self

        len_before = len(self.data)

        # Itération synchronisée sur les tags, les mins et les maxs
        for t, mi, ma in zip(tags, mins, maxs):
            if t in self.data.columns:
                # La comparaison se fait maintenant valeur par valeur (mi et ma sont des scalaires ici)
                self.data = self.data[(self.data[t] >= mi) & (self.data[t] <= ma)]
                self._log(f"Filtre appliqué sur '{t}': [{mi} - {ma}]")
            else:
                self._log(f"Attention : Colonne '{t}' introuvable pour le filtrage.")
        
        if na:
            cols_na = [na] if isinstance(na, str) else na
            self.data.dropna(subset=cols_na, how="all", inplace=True)
            
        len_after = len(self.data)
        self._log(f"Lignes restantes : {len_after} (supprimées : {len_before - len_after})")
        return self

    @register_step
    def ajoute_cumul(self, col_poids, col_valeur, ratio, nom):
        if {col_poids, col_valeur}.issubset(self.data.columns):
            self.data[nom] = (self.data[col_poids] * self.data[col_valeur]) / ratio
        return self

    @register_step
    def ajouter_moyennes_glissantes(self, col_poids, col_valeur, nom, window=10):
        if {col_poids, col_valeur}.issubset(self.data.columns):
            prod = (self.data[col_poids] * self.data[col_valeur]).rolling(window).sum()
            poids_sum = self.data[col_poids].rolling(window).sum()
            self.data[nom] = prod / poids_sum
        return self

    # --- SETTERS ---
    def set_start(self, val): self.start = val; self._log(f"Start set to {val}")
    def set_end(self, val): self.end = val; self._log(f"End set to {val}")
    def set_interval(self, val): self.interval = val; self._log(f"Interval set to {val}")

    def show_pipeline(self):
        """Affiche les étapes de traitement enregistrées dans le pipeline."""
        if not self.pipeline:
            print("\n[PIPELINE] Vide. Aucune étape enregistrée.")
            return

        print("\n" + "="*50)
        print(f"{'ORDRE':<7} | {'FONCTION':<25} | {'PARAMÈTRES'}")
        print("-" * 50)
        
        for i, step in enumerate(self.pipeline, 1):
            # Formatage des arguments pour un affichage propre
            args_str = ", ".join([str(a) for a in step['args']])
            kwargs_str = ", ".join([f"{k}={v}" for k, v in step['kwargs'].items()])
            params = f"{args_str}{', ' if args_str and kwargs_str else ''}{kwargs_str}"
            
            print(f"{i:<7} | {step['func']:<25} | {params}")
        
        print("="*50 + "\n")

    def remove_last_step(self):
        """Supprime la dernière étape enregistrée dans le pipeline."""
        if self.pipeline:
            removed = self.pipeline.pop()
            self._log(f"Étape supprimée : {removed['func']}")
        return self

    # --- SETTERS AMÉLIORÉS ---
    # Vous pouvez maintenant passer 'autorecalc=True' pour gagner du temps
    def update_config(self, start=None, end=None, interval=None, autorecalc=False):
        """Met à jour plusieurs paramètres à la fois et relance le calcul si besoin."""
        if start: self.start = start
        if end: self.end = end
        if interval: self.interval = interval
        
        self._log(f"Configuration mise à jour (autorecalc={autorecalc})")
        
        if autorecalc:
            return self.recalculate()
        return self