import pandas as pd
import numpy as np
from urllib.parse import quote
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from functools import wraps
import logging

class OI_DataProcessor:
    """
    Classe complète pour la gestion, le traitement (Pipeline) 
    et la visualisation de données API.
    """
    
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
        self.pipeline = [] 
        self._is_recalculating = False

        # Gestion des credentials
        try:
            with open(cred_file, "r") as f:
                self.credentials = f.read().strip()
        except FileNotFoundError:
            print(f"[ERREUR] Fichier credentials introuvable : {cred_file}")
            self.credentials = ""

        # Mapping initial
        self.rename_mapping = {item['tag']: item['nom'] for item in tags_selected}
        self.df = None    # Données brutes (API)
        self.data = None  # Données de travail (Renommées/Traitées)

    # --- MÉCANISME DE LOG ET PIPELINE ---
    def _log(self, message):
        """Affiche un message si verbose est True."""
        if self.verbose:
            print(f"[INFO] {message}")
    def register_step(func):
        """Décorateur pour enregistrer automatiquement les actions dans le pipeline."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if not self._is_recalculating:
                self.pipeline.append({'func': func.__name__, 'args': args, 'kwargs': kwargs})
                self._log(f"Étape ajoutée au pipeline : {func.__name__}")
            return result
        return wrapper
    # --- RÉCUPÉRATION ET FUSION ---
    def get_OI(self, tag):
        """Récupère les données brutes d'un tag spécifique."""
        url = (f"{self.url_base}data-reference={tag}&aggregation=TIME"
               f"&aggregation-function=MEAN&from={self.start}T{self.hS}%3A00%3A00.000Z"
               f"&to={self.end}T{self.hF}%3A59%3A59.000Z&aggregation-period={self.interval}")
        headers = {'Authorization': f'basic {self.credentials}'}
        try:
            d_data = pd.read_json(url, storage_options=headers)
            if 'values' in d_data and len(d_data['values']) > 0:
                return d_data['values'][0],d_data['unit'][0]
            return []
        except Exception as e:
            print(f"[ATTENTION] Erreur récupération tag '{tag}': {e}")
            print(f"📋 [CURL] Commande à copier-coller dans le terminal :")
            print("-" * 80)
            # Générer la commande curl
            curl_cmd = f"curl -X GET '{url}'"
            for key, value in headers.items():
                curl_cmd += f" \\\n  -H '{key}: {value}'"
            print(curl_cmd)
            print("-" * 80)
            return []
    def merge(self):
        """Récupère tous les tags et initialise self.data."""
        self._log(f"Début fusion ({self.start} au {self.end})")
        tags_api = self.tags_other + list(self.rename_mapping.keys())
        df_list = []
        self.unit_tags = []
        for tag in tags_api:
            encoded_tag = quote(tag, safe=':/?#[]@!$&\'()*+;-=')
            #encoded_tag = quote(tag)
            raw_data, unit = self.get_OI(encoded_tag)            
            if raw_data:
                temp_df = pd.DataFrame(raw_data)
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
                temp_df.set_index('timestamp', inplace=True)
                temp_df.rename(columns={'value': tag}, inplace=True)
                df_list.append(temp_df)
                self.unit_tags.append({'tag': tag, 'unit': unit, 'nom': self.rename_mapping.get(tag, tag)})
        if not df_list:
            print("[ERREUR] Aucune donnée récupérée.")
            return self
        self.df = pd.concat(df_list, axis=1)
        self.data = self.df.rename(columns=self.rename_mapping)
        self._log(f"Fusion terminée : {self.data.shape}")
        return self
    # --- TRAITEMENTS ENREGISTRÉS DANS LE PIPELINE ---
    @register_step
    def filtering(self, tag, min_val, max_val, na=None):
        """Filtre les données. Gère les listes de tags et de valeurs min/max."""
        if self.data is None: return self
        self.data.dropna(how='all', inplace=True)
        
        tags = [tag] if isinstance(tag, str) else tag
        mins = [min_val] * len(tags) if not isinstance(min_val, list) else min_val
        maxs = [max_val] * len(tags) if not isinstance(max_val, list) else max_val

        for t, mi, ma in zip(tags, mins, maxs):
            if t in self.data.columns:
                self.data = self.data[(self.data[t].between(mi, ma)) | (self.data[t].isna())]
            else:
                self._log(f"Tag '{t}' introuvable pour filtrage.")
        
        if na:
            cols_na = [na] if isinstance(na, str) else na
            self.data.dropna(subset=cols_na, how="all", inplace=True)
        
        self._log(f"Filtrage effectué. Lignes restantes : {len(self.data)}")
        return self

    @register_step
    def ajoute_cumul(self, col_poids, col_valeur, ratio, nom):
        """Calcule une nouvelle colonne (Poids * Valeur / Ratio)."""
        if self.data is not None and {col_poids, col_valeur}.issubset(self.data.columns):
            self.data[nom] = (self.data[col_poids] * self.data[col_valeur]) / ratio
            unit_ = next((p for p in self.unit_tags if p['nom'] == col_poids), None) if self.unit_tags else None
            unit1 = unit_['unit'] if unit_ else 'NA'
            unit_ = next((p for p in self.unit_tags if p['nom'] == col_valeur), None) if self.unit_tags else None
            unit2 = unit_['unit'] if unit_ else 'NA'
            unit = '('+unit1 + 'x' + unit2 +')/'+str(ratio)
            self.unit_tags.append({'tag': nom, 'unit': unit, 'nom': nom})
            self._log(f"Colonne cumulée '{nom}' ajoutée. [Unit: {unit}]")
        return self

    @register_step
    def ajoute_calcul(self, value, type, uv_0, uv_1, scale, nom):
        """
        Calcule une colonne cumulative (optimisée vectorisée) :
        cumul(i) = cumul(i-1) + (value(i)-value(i-1)) * (type(i)==0 ? uv_0(i) : uv_1(i)) * scale
        Si value(i) < value(i-1), ajouter 1_000_000 au cumul
        """
        if self.data is not None and {value, type, uv_0, uv_1}.issubset(self.data.columns):
            # Calculer le delta
            delta = self.data[value].diff().fillna(0)
            # Coefficient = uv_0 si type==0, sinon uv_1
            coefficient = np.where(
                self.data[type] == 0,
                self.data[uv_0],
                self.data[uv_1]
            )
            # Correction si débordement (value[i] < value[i-1])
            correction = np.where(
                self.data[value] < self.data[value].shift(1),
                self.data[value].shift(1)-self.data[value],
                0
            )
            correction[0] = 0  # Première ligne pas de correction
            # Calcul cumulatif
            self.data[nom] = (delta * coefficient * scale + correction).cumsum()
            self.unit_tags.append({'tag': nom, 'nom': nom})
            self._log(f"Colonne cumulative '{nom}' ajoutée.")
        else:
            self._log(f"Erreur : colonnes manquantes pour '{nom}'", level='error')
        return self

    @register_step
    def ajoute_all(self, batchs, nom):
        """
        Crée une colonne qui est la SOMME des valeurs calculées pour TOUS les batchs/continus.
        
        Chaque élément de batchs peut être :
        - Un batch (avec 'pu', 'in', 'out', 'value', etc.)
        - Un continu (avec 'tag'/'value', 'min', 'epalage', etc.)
        - Ou un hybride !
        """
        if self.data is None or not batchs:
            self._log(f"Erreur : data est None ou batchs vide")
            return self
        
        n_rows = len(self.data)
        resultats_batchs = []
        
        for i, batch in enumerate(batchs):
            # 1. Extraction des paramètres
            pu = batch.get('pu')
            in_val = batch.get('in', -np.inf)
            out_val = batch.get('out', np.inf)
            
            # Fallback pour value/tag
            value_col = batch.get('value') or batch.get('tag')
            uv_col = batch.get('uv')
            scale = batch.get('scale', 1.0)
            
            # Paramètres d'épalage
            epalage = batch.get('epalage')
            min_val = batch.get('min')
            
            # Conditions et multiplicateurs
            cond = batch.get('cond')
            val_cond = batch.get('val_cond', [1.0, 1.0])
            
            # 2. Gestion de la condition 'pu' (si spécifiée)
            if pu is not None:
                if pu not in self.data.columns:
                    self._log(f"Élément {i}: colonne 'pu' '{pu}' manquante")
                    return self
                pu_vals = self.data[pu].values
                condition_pu = (pu_vals >= in_val) & (pu_vals < out_val)
            else:
                condition_pu = np.ones(n_rows, dtype=bool)
                
            # 3. Récupération de la colonne 'value' / 'tag'
            if value_col is None:
                value_vals = np.ones(n_rows)
            else:
                if value_col not in self.data.columns:
                    self._log(f"Élément {i}: colonne de valeur '{value_col}' manquante")
                    return self
                value_vals = self.data[value_col].values
                
            # 4. Récupération de la colonne 'uv'
            if uv_col is None:
                uv_vals = np.ones(n_rows)
            else:
                if uv_col not in self.data.columns:
                    self._log(f"Élément {i}: colonne 'uv' '{uv_col}' manquante")
                    return self
                uv_vals = self.data[uv_col].fillna(1.0).values
                
            # 5. Gestion du multiplicateur de condition 'cond'
            if cond is not None:
                if cond not in self.data.columns:
                    self._log(f"Élément {i}: colonne de condition '{cond}' manquante")
                    return self
                cond_vals = self.data[cond].values
                cond_multiplicateur = np.where(cond_vals == 0, val_cond[0], val_cond[1])
            else:
                cond_multiplicateur = 1.0
                
            # 6. Calcul de la valeur théorique (avec ou sans épalage)
            if epalage is not None:
                # Épalage polynomial sur les valeurs de value_vals
                poly_low = self._eval_polynomial(epalage[0], value_vals)
                poly_high = self._eval_polynomial(epalage[1], value_vals)
                
                # Le seuil min détermine quel épalage utiliser
                condition_min = value_vals <= min_val
                calc_val = np.where(condition_min, poly_low, poly_high) * uv_vals * scale * cond_multiplicateur
            else:
                # Calcul simple standard
                calc_val = value_vals * uv_vals * scale * cond_multiplicateur
                
            # 7. Application de la condition pu (0 si non respectée)
            batch_result = np.where(condition_pu, calc_val, 0.0)
            
            resultats_batchs.append(batch_result)
            
        # Somme vectorielle de tous les éléments
        self.data[nom] = np.stack(resultats_batchs, axis=0).sum(axis=0)
        self.unit_tags.append({'tag': nom, 'nom': nom})
        self._log(f"Colonne all-sum '{nom}' ajoutée ({len(batchs)} éléments)")
        
        return self

    def _eval_polynomial(self, coeffs, x):
        """
        Fonction interne pour évaluer un polynôme vectorisé :
        P(x) = coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
        
        Args:
            coeffs : list de coefficients [c0, c1, c2, ...]
            x : ndarray des valeurs
        
        Returns:
            ndarray des résultats
        """
        result = np.zeros_like(x, dtype=float)
        for i, c in enumerate(coeffs):
            result += c * (x ** i)
        return result

    @register_step
    def ajouter_moyennes_glissantes_ponderee(self, col_poids, col_valeur, nom, window=10):
        """Calcule une moyenne glissante pondérée."""
        if self.data is not None and {col_poids, col_valeur}.issubset(self.data.columns):
            prod = (self.data[col_poids] * self.data[col_valeur]).rolling(window).sum()
            poids_sum = self.data[col_poids].rolling(window).sum()
            self.data[nom] = prod / poids_sum
            unit_ = next((p for p in self.unit_tags if p['nom'] == col_valeur), None) if self.unit_tags else None
            unit = unit_['unit'] if unit_ else 'NA'
            self.unit_tags.append({'tag': nom, 'unit': unit, 'nom': nom})
            self._log(f"Moyenne glissante pondérée '{nom}' ajoutée.")
        return self

    @register_step
    def ajouter_moyennes_glissantes(self, col_valeur, nom, window=10):
        """Calcule une moyenne glissante."""
        if self.data is not None and { col_valeur}.issubset(self.data.columns):
            prod = (self.data[col_valeur]).rolling(window).sum()
            self.data[nom] = prod / window
            unit_ = next((p for p in self.unit_tags if p['nom'] == col_valeur), None) if self.unit_tags else None
            unit = unit_['unit'] if unit_ else 'NA'
            self.unit_tags.append({'tag': nom, 'unit': unit, 'nom': nom})
            self._log(f"Moyenne glissante '{nom}' ajoutée.")
        return self

    # --- GESTION DU RECALCUL ---

    def recalculate(self):
        """Relance le merge et réapplique tout le pipeline."""
        self._log("=== LANCEMENT DU RECALCUL GÉNÉRAL ===")
        self._is_recalculating = True
        try:
            self.merge()
            for step in self.pipeline:
                func = getattr(self, step['func'])
                func(*step['args'], **step['kwargs'])
            self._log("=== RECALCUL TERMINÉ AVEC SUCCÈS ===")
        finally:
            self._is_recalculating = False
        return self

    def show_pipeline(self):
        """Affiche les étapes du pipeline."""
        print("\n--- PIPELINE ACTUEL ---")
        for i, s in enumerate(self.pipeline, 1):
            print(f"{i}. {s['func']} | Args: {s['args']} | Kwargs: {s['kwargs']}")
        print("-----------------------\n")

    def clear_pipeline(self):
        self.pipeline = []
        self._log("Pipeline effacé.")

    # --- SETTERS ET UTILITAIRES ---

    def set_start(self, val): self.start = val; self._log(f"Date début : {val}")
    def set_end(self, val): self.end = val; self._log(f"Date fin : {val}")
    def set_interval(self, val): self.interval = val; self._log(f"Intervalle : {val}")
    def set_hS(self, val): self.hS = val
    def set_hF(self, val): self.hF = val

    def reset_data(self):
        """Réinitialise data à partir de df (brut)."""
        if self.df is not None:
            self.data = self.df.rename(columns=self.rename_mapping)
            self._log("Données réinitialisées (cache API conservé).")

    # --- VISUALISATION (PLOT_TAG COMPLET) ---

    def plot_simple_tag(self, tag):
        """Analyse graphique complète (Temporel + Distribution + Stats)."""
        if self.data is None: return
        tags = [tag] if isinstance(tag, str) else tag
        
        for t in tags:
            if t not in self.data.columns: continue
            series = self.data[t].dropna()
            if series.empty: continue

            fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3],
                                subplot_titles=(f"Série : {t}", "Distribution"))
            
            # Scatter
            fig.add_trace(go.Scatter(x=series.index, y=series, mode='markers', 
                                     marker=dict(opacity=0.4, size=5), name="Données"), row=1, col=1)
            # Tendance
            x_vals = np.arange(len(series))
            slope, intercept, r_val, p_val, std_err = stats.linregress(x_vals, series.values)
            fig.add_trace(go.Scatter(x=series.index, y=intercept + slope*x_vals, 
                                     line=dict(color='red'), name="Tendance"), row=1, col=1)
            # Histogramme
            fig.add_trace(go.Histogram(x=series, nbinsx=30, histnorm='probability density'), row=1, col=2)
            
            fig.update_layout(height=500, title_text=f"Analyse {t} (R²={r_val**2:.3f})", showlegend=False)
            fig.show()

    def info(self):
        """
        Affiche tous les paramètres de la classe.
        """
        print("="*60)
        print("INFORMATIONS DataProcessor")
        print("="*60)
        print(f"URL de base       : {self.url_base}")
        print(f"Tags autres       : {self.tags_other}")
        print(f"Tags sélectionnés : {self.tags_selected}")
        tags_api = [item['tag'] for item in self.tags_selected]
        print(f"Tous les tags API : {self.tags_other + tags_api}")
        print(f"Mapping renommage : {self.rename_mapping}")
        print(f"Date de début     : {self.start}")
        print(f"Date de fin       : {self.end}")
        print(f"Intervalle        : {self.interval}")
        print(f"Heure de début    : {self.hS}")
        print(f"Heure de fin      : {self.hF}")
        print("-"*60)
        if self.df is not None:
            print(f"DataFrame brut (df)      : {len(self.df)} lignes, {len(self.df.columns)} colonnes")
            print(f"Colonnes df (noms API)   : {list(self.df.columns)}")
        else:
            print("DataFrame brut (df)      : Non chargé")
        
        if self.data is not None:
            print(f"DataFrame travail (data)    : {len(self.data)} lignes, {len(self.data.columns)} colonnes")
            print(f"Colonnes data (noms finaux) : {list(self.data.columns)}")
        else:
            print("DataFrame travail (data)    : Non chargé")
        print("="*60)
 
    def plot_tag(self, tag, index=None):
            """
            Affiche les graphiques temporel et de distribution pour un ou plusieurs tags.
            
            Parameters:
            tag : str or list, nom du/des tag(s) à afficher
            index : str, optionnel, nom d'une courbe à afficher sur un second axe Y (ex: régime moteur)
            """
            if self.data is None:
                print("Erreur : Aucune donnée disponible. Exécutez merge() d'abord.")
                return
            
            tags = [tag] if isinstance(tag, str) else tag
            IDX_ = 0
            
            # Vérification des tags valides
            tags_valides = [t for t in tags if t in self.data.columns]
            if not tags_valides:
                print(f"Erreur : Aucun tag valide trouvé. Colonnes : {list(self.data.columns)}")
                return

            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
            
            for idx, tag_name in enumerate(tags_valides):
                color = colors[idx % len(colors)]
                data_series = self.data[tag_name].dropna()
                
                if len(data_series) == 0:
                    continue
                    
                # Calcul des statistiques (conservé tel quel)
                moyenne = data_series.mean()
                mediane = data_series.median()
                ecart_type = data_series.std()
                nb_valeurs = len(data_series)
                percentile_25 = data_series.quantile(0.25)
                percentile_75 = data_series.quantile(0.75)
                val_min = data_series.min()
                val_max = data_series.max()
                
                # --- MODIFICATION : Activation du second axe Y pour le col 1 ---
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=(f'{tag_name}', f'Distribution'),
                    horizontal_spacing=0.05,
                    column_widths=[0.6, 0.4],
                    specs=[[{"secondary_y": True}, {"secondary_y": True}]]  # Axe secondaire pour le subplot 1 & 2
                )
                
                # === SUBPLOT 1 : Scatter plot (Axe Y Principal) ===
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index, y=self.data[tag_name],
                        mode='markers', name=tag_name,
                        marker=dict(color=color, opacity=0.3, size=6),
                        hovertemplate='Date: %{x}<br>Valeur: %{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1, secondary_y=False
                )
                IDX_ += 1

                # --- MODIFICATION : Ajout de la courbe d'index sur l'axe secondaire ---
                val_max_index = None
                if index and index in self.data.columns:
                    val_max_index = self.data[index].max()
                    fig.add_trace(
                        go.Scatter(
                            x=self.data.index,
                            y=self.data[index],
                            mode='lines',
                            name=f"Index: {index}",
                            line=dict(color='rgba(128, 128, 128, 0.5)', width=1.5, shape='hv'),
                            hovertemplate=f'{index}: %{{y:.2f}}<extra></extra>'
                        ),
                        row=1, col=1, secondary_y=True
                    )
                    IDX_ += 1
#                    fig.update_yaxes(title_text=f"Axe Index ({index})", secondary_y=True, row=1, col=1)
                    # Masquage complet de l'axe Y secondaire (pas de grille, pas de chiffres)
                    fig.update_yaxes(
                        range=[0, val_max_index * 5], # Calage sur les 20% bas
                        showgrid=False,               # Pas de quadrillage
                        zeroline=False,               # Pas de ligne de zéro
                        showticklabels=False,         # Pas de chiffres sur le côté
                        secondary_y=True, row=1, col=1
                    )
                # Ligne de moyenne
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index, y=[moyenne] * len(self.data.index),
                        mode='lines', name=f'Moyenne: {moyenne:.2f}',
                        line=dict(color='red', dash='dash', width=2)
                    ),
                    row=1, col=1, secondary_y=False
                )
                IDX_ += 1
                
                # Ligne de médiane
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index, y=[mediane] * len(self.data.index),
                        mode='lines', name=f'Médiane: {mediane:.2f}',
                        line=dict(color='violet', dash='dash', width=2)
                    ),
                    row=1, col=1, secondary_y=False
                )
                IDX_ += 1
                
                # Courbe de tendance (Régression linéaire)
                x = np.arange(len(self.data.index))
                y = self.data[tag_name].values
                mask = ~np.isnan(y)
                if len(x[mask]) > 1:
                    coeffs = np.polyfit(x[mask], y[mask], 1)
                    line = np.polyval(coeffs, x)
                    fig.add_trace(
                        go.Scatter(
                            x=self.data.index, y=line,
                            mode='lines', name='Tendance',
                            line=dict(color='green', width=2)
                        ),
                        row=1, col=1, secondary_y=False
                    )

                # === SUBPLOT 2 : Histogramme (Distribution) ===
                fig.add_trace(
                    go.Histogram(
                        x=data_series, name='Distribution',
                        marker=dict(color='lightblue', line=dict(color='darkblue', width=1)), opacity=0.7,
                        histnorm='',
                        showlegend=False
                    ),
                    row=1, col=2, secondary_y=False
                )
                IDX_ += 1

                # KDE et lignes stats (Moyenne, Médiane, Percentiles) sur Col 2
                if len(data_series) > 1:
                    kde = stats.gaussian_kde(data_series)
                    x_range = np.linspace(val_min, val_max, 200)
                    y_max = max(kde(x_range)) * 1.1
                    
                    fig.add_trace(
                        go.Scatter(x=x_range, y=kde(x_range), mode='lines', line=dict(color='darkblue', width=2), showlegend=False),
                        row=1, col=2, secondary_y=False
                    )
                    IDX_ += 1
                    # Lignes verticales sur subplot 2
                    for val, name, c, d in [(moyenne, 'Moyenne', 'red', 'dash'), (mediane, 'Médiane', 'violet', 'dash'), 
                                        (percentile_25, 'P25', 'orange', 'dot'), (percentile_75, 'P75', 'brown', 'dot')]:
                        fig.add_trace(
                            go.Scatter(x=[val, val], y=[0, 1], mode='lines', name=name, 
                                    line=dict(color=c, dash=d, width=1.5), showlegend=False),
                            row=1, col=2, secondary_y=True
                        )
                        IDX_ += 1

                # Fixer l'axe secondaire du subplot 2 pour que les lignes fassent toute la hauteur
                fig.update_yaxes(range=[0, 1], visible=False, secondary_y=True, row=1, col=2)                

                # Statistiques en légende (Traces invisibles)
                stats_text = [f"<b>Stats {tag_name}</b>", f"N: {nb_valeurs}", f"Avg: {moyenne:.2f}", f"Std: {ecart_type:.2f}", f"Idx: {val_max_index}", f"[{val_min:.2f} - {val_max:.2f}]"]
                for stat in stats_text:
                    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', name=stat, showlegend=True), row=1, col=1)

                #=== BOUTON TOGGLE (Nombre vs Densité) ===
                fig.update_layout(
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="right",
                            x=1.08, y=1.12,
                            showactive=True,
                            bgcolor="white",
                            bordercolor="lightblue",
                            font=dict(color="darkblue"),
                            active=0,
                            buttons=[
                                dict(label="Nombre",
                                    method="update",
                                    args=[{"histnorm": [""]}, # Update traces
                                        {"yaxis3.title.text": "Nombre", # Update layout axis title
                                        "updatemenus[0].bordercolor": "lightblue"}]), # Update button state]
                                dict(label="Densité",
                                    method="update",
                                    args=[{"histnorm": ["probability density"]}, # Update traces
                                        {"yaxis3.title.text": "Densité",
                                        "updatemenus[0].bordercolor": "lightblue"} ])
                            ]
                        )
                    ]
                )

                # Mise en page finale
                fig.update_layout(
                    #title_text=f"Analyse de {tag_name}",
                    height=600, template="plotly_white",hovermode="x unified",
                    legend=dict(
                        yanchor="top",xanchor="left",
                        y=0.99, x=0.95,
                        bgcolor="rgba(248, 249, 250, 0.9)", # Fond gris très clair
                        bordercolor="rgba(100, 100, 100, 0.5)",
                        borderwidth=1,
                        font=dict(size=11, color="black")
                    ),
                    # Options de zoom et barre d'outils
                    dragmode="zoom" # Zoom par défaut
                )
                unit_ = next((p for p in self.unit_tags if p['nom'] == tag_name), None) if self.unit_tags else None
                unit = unit_['unit'] if unit_ else 'non spécifiée'
                fig.update_xaxes(title_text="Date/Temps", row=1, col=1)

#---------------------
# CLASSE SPÉCIALISÉE POUR LE BILAN DE PRODUCTION
# ---------------------
#                
class OI_ProductionProcessor(OI_DataProcessor):
    """
    Classe spécialisée héritant de OI_DataProcessor pour automatiser
    le calcul et le suivi des bilans de production, consommation et stocks par produit.
    """
    
    def __init__(self, url_base, start, end, tags_metadata, produits,
                 interval='PT20M', hS='00', hF='23', cred_file="../../../cred.txt", 
                 verbose=True):
        """
        Initialise le processeur de production.
        
        Parameters:
        -----------
        url_base : str
            URL de base pour l'API.
        start, end : str
            Dates de début et de fin.
        tags_metadata : list
            Liste des dictionnaires décrivant les tags (metadata).
        produits : list
            Liste unique de dictionnaires décrivant les produits.
            Chaque dictionnaire contient:
            - 'nom': nom du produit (ex: 'Acetate')
            - 'conso': dictionnaire de configuration pour la consommation
            - 'stock': liste des composants (batch/continu) pour le stock
        """
        super().__init__(url_base, [], tags_metadata, start, end, 
                         interval, hS, hF, cred_file, verbose)
        
        self.produits = produits

    def ajoute_conso_produit(self, value, type,  uv_col, scale, nom):
        """
        Calcule le cumul de consommation spécifique à un produit :
        cumul(i) = cumul(i-1) + (value(i)-value(i-1)) * uv(i) * scale (uniquement si type_col(i) == type_val)
        Si value(i) < value(i-1), ajouter le débordement
        """
        if self.data is not None and {value, uv_col[0], uv_col[1]}.issubset(self.data.columns):
            n_rows = len(self.data)
            delta = self.data[value].diff().fillna(0)
            
            # Récupérer les valeurs UV
            if uv_col is None:
                uv_vals = np.ones(n_rows)
            else:
                uv_vals = np.where(
                    self.data[type] == 0,
                    self.data[uv_col[0]].fillna(1.0).values,
                    self.data[uv_col[1]].fillna(1.0).values
                )
                 
            # Actif uniquement si type_col == type_val
#            active = self.data[uv_col] == type_val
            active = 1
            
            # Correction si débordement de la balance
            correction = np.where(
                self.data[value] < self.data[value].shift(1),
                self.data[value].shift(1) - self.data[value],
                0.0
            )
            correction[0] = 0.0
            
            # Calcul cumulatif uniquement sur les périodes actives
            corrected_delta = np.where(active, delta * uv_vals * scale + correction, 0.0)
            
            self.data[nom] = corrected_delta.cumsum()
            self.unit_tags.append({'tag': nom, 'nom': nom})
            self._log(f"Colonne de consommation '{nom}' ajoutée.")
        return self

    @OI_DataProcessor.register_step
    def compute_production_balance(self):
        """
        Exécute et automatise l'ensemble du pipeline pour chaque produit :
        1. Filtrage sur la consommation générale (Ester >= 0).
        2. Pour chaque produit, calcule:
           - consommation_{produit}
           - stock_{produit}
           - conso_delta_{produit} (consommation cumulée depuis le début)
           - delta_stock_{produit} (variation de stock depuis le début)
           - production_{produit} (production totale = conso_delta + delta_stock)
        """
        # Désactivation temporaire de la double-inscription dans le pipeline pour les sous-étapes
        already_recalculating = self._is_recalculating
        self._is_recalculating = True
        
        try:
            # A. Filtrage global
            self.filtering(tag=['Ester'], min_val=[0], max_val=[10_000_000], na=['Ester'])
            
            # B. Calculs par produit
            for prod in self.produits:
                nom = prod['nom']
                conso_conf = prod.get('conso')
                stock_list = prod.get('stock', [])
                
                # 1. Consommation cumulée du produit
                conso_col = f"consommation_{nom}"
                if conso_conf:
                    conso_val = conso_conf.get('value', 'Ester')
                    conso_type = conso_conf.get('type', 'A/P')
                    conso_uv = conso_conf.get('uv')
                    conso_scale = conso_conf.get('scale', 1e-9)
                    
                    self.ajoute_conso_produit(conso_val,conso_type, conso_uv, conso_scale, conso_col)
                else:
                    self.data[conso_col] = 0.0
                
                # 2. Stock unifié du produit
                stock_col = f"stock_{nom}"
                if stock_list:
                    self.ajoute_all(stock_list, stock_col)
                else:
                    self.data[stock_col] = 0.0
                    
        finally:
            self._is_recalculating = already_recalculating
            
        # C. Calcul des deltas et production pour chaque produit
        if self.data is not None:
            for prod in self.produits:
                nom = prod['nom']
                conso_col = f"consommation_{nom}"
                stock_col = f"stock_{nom}"
                
                # Évolution de la consommation depuis t=0
                self.data[f"conso_delta_{nom}"] = self.data[conso_col] - self.data[conso_col].iloc[0]
                
                # Évolution du stock depuis t=0
                self.data[f"delta_stock_{nom}"] = self.data[stock_col] - self.data[stock_col].iloc[0]
                
                # Production totale du produit
                self.data[f"production_{nom}"] = self.data[f"conso_delta_{nom}"] + self.data[f"delta_stock_{nom}"]
                
            self._log("Bilans de production calculés avec succès pour tous les produits.")
            
        return self

    def calcul_cumul_mensuel(self, mois, annee=None, nom_produit=None, std=2):
        """
        Calcule le cumul de consommation, de variation de stock et de production
        pour un mois donné, pour un produit spécifique ou pour tous les produits.
        """
        df = self.data
        if df is None or df.empty:
            print("Erreur : Les données sont vides.")
            return None
            
        if annee is None:
            annee = df.index.year[0]
            
        # Parsing du mois
        month_num = None
        if isinstance(mois, int):
            month_num = mois
        elif isinstance(mois, str):
            mois_clean = mois.strip().lower()
            if '-' in mois_clean:
                try:
                    parts = mois_clean.split('-')
                    annee = int(parts[0])
                    month_num = int(parts[1])
                except ValueError:
                    pass
            else:
                mois_fr = {
                    'janvier': 1, 'jan': 1, 'fevrier': 2, 'février': 2, 'fev': 2, 'fév': 2,
                    'mars': 3, 'mar': 3, 'avril': 4, 'avr': 4, 'mai': 5, 'juin': 6, 'jui': 6,
                    'juillet': 7, 'juil': 7, 'aout': 8, 'août': 8, 'aou': 8, 'aoû': 8,
                    'septembre': 9, 'sept': 9, 'sep': 9, 'octobre': 10, 'oct': 10,
                    'novembre': 11, 'nov': 11, 'decembre': 12, 'décembre': 12, 'dec': 12, 'déc': 12
                }
                if mois_clean in mois_fr:
                    month_num = mois_fr[mois_clean]
                else:
                    try:
                        month_num = int(mois_clean)
                    except ValueError:
                        pass
                        
        if month_num is None or not (1 <= month_num <= 12):
            print(f"Erreur : Mois '{mois}' non valide.")
            return None

        # Dates cibles
        start_dt = pd.Timestamp(year=annee, month=month_num, day=1, hour=std, minute=0, second=0)
        if month_num == 12:
            end_dt = pd.Timestamp(year=annee + 1, month=1, day=1, hour=std, minute=0, second=0)
        else:
            end_dt = pd.Timestamp(year=annee, month=month_num + 1, day=1, hour=std, minute=0, second=0)
            
        if df.index.tz is not None:
            start_dt = start_dt.tz_localize(df.index.tz)
            end_dt = end_dt.tz_localize(df.index.tz)
            
        last_dt = df.index[-1]
        is_finished = end_dt <= last_dt
        actual_end_dt = end_dt if is_finished else last_dt
        
        try:
            # Start row
            if start_dt in df.index:
                row_start = df.loc[start_dt]
            else:
                future_idx = df.index[df.index >= start_dt]
                if len(future_idx) == 0:
                    print(f"Erreur : Pas de données après le début du mois.")
                    return None
                row_start = df.loc[future_idx[0]]
                start_dt = future_idx[0]
                
            # End row
            if actual_end_dt in df.index:
                row_end = df.loc[actual_end_dt]
            else:
                past_idx = df.index[df.index <= actual_end_dt]
                if len(past_idx) == 0:
                    print(f"Erreur : Pas de données avant la fin du mois.")
                    return None
                row_end = df.loc[past_idx[-1]]
                actual_end_dt = past_idx[-1]
        except Exception as e:
            print(f"Erreur recherche dates : {e}")
            return None
            
        # Filtre des produits à calculer
        produits_to_calc = [p['nom'] for p in self.produits]
        if nom_produit is not None:
            if nom_produit not in produits_to_calc:
                print(f"Erreur : Le produit '{nom_produit}' n'est pas défini dans la variable produits.")
                return None
            produits_to_calc = [nom_produit]
            
        # Affichage
        status_str = "Terminé" if is_finished else "À date (En cours)"
        nom_mois = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"][month_num - 1]
        
        print("_" * 60)
        print(f"BILAN MENSUEL - {nom_mois.upper()} {annee} ({status_str})")
        print(f"Période : du {start_dt.strftime('%d/%m/%Y à %H:%M')} au {actual_end_dt.strftime('%d/%m/%Y à %H:%M')}")
        print("_" * 60)
        
        bilan_results = {}
        for nom in produits_to_calc:
            conso_delta_col = f"conso_delta_{nom}"
            delta_stock_col = f"delta_stock_{nom}"
            production_col = f"production_{nom}"
            
            # Le delta de consommation pour ce mois particulier
            delta_conso = row_end[conso_delta_col] - row_start[conso_delta_col]
            # La variation de stock pour ce mois particulier
            delta_stock_val = row_end[delta_stock_col] - row_start[delta_stock_col]
            # La production pour ce mois particulier
            delta_prod = row_end[production_col] - row_start[production_col]
            
            print(f"PRODUIT : {nom.upper()}")
            print("-" * 60)
            print(f"  Consommation (Ester)   : {delta_conso:.4f}")
            print(f"  Variation de Stock     : {delta_stock_val:.4f}")
            print(f"  Production             : {delta_prod:.4f}")
            print("_" * 60)
            
            bilan_results[nom] = {
                'consommation': delta_conso,
                'delta_stock': delta_stock_val,
                'production': delta_prod
            }
        print("_" * 60)
        return bilan_results