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
        try:
            d_data = pd.read_json(url, storage_options={'Authorization': f'basic {self.credentials}'})
            if 'values' in d_data and len(d_data['values']) > 0:
                return d_data['values'][0]
            return []
        except Exception as e:
            print(f"[ATTENTION] Erreur récupération tag '{tag}': {e}")
            return []

    def merge(self):
        """Récupère tous les tags et initialise self.data."""
        self._log(f"Début fusion ({self.start} au {self.end})")
        tags_api = self.tags_other + list(self.rename_mapping.keys())
        df_list = []

        for tag in tags_api:
            encoded_tag = quote(tag, safe=':/?#[]@!$&\'()*+,;=')
            raw_data = self.get_OI(encoded_tag)
            
            if raw_data:
                temp_df = pd.DataFrame(raw_data)
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
                temp_df.set_index('timestamp', inplace=True)
                temp_df.rename(columns={'value': tag}, inplace=True)
                df_list.append(temp_df)

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
        
        tags = [tag] if isinstance(tag, str) else tag
        mins = [min_val] * len(tags) if not isinstance(min_val, list) else min_val
        maxs = [max_val] * len(tags) if not isinstance(max_val, list) else max_val

        for t, mi, ma in zip(tags, mins, maxs):
            if t in self.data.columns:
                self.data = self.data[(self.data[t] >= mi) & (self.data[t] <= ma)]
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
            self._log(f"Colonne cumulée '{nom}' ajoutée.")
        return self

    @register_step
    def ajouter_moyennes_glissantes(self, col_poids, col_valeur, nom, window=10):
        """Calcule une moyenne glissante pondérée."""
        if self.data is not None and {col_poids, col_valeur}.issubset(self.data.columns):
            prod = (self.data[col_poids] * self.data[col_valeur]).rolling(window).sum()
            poids_sum = self.data[col_poids].rolling(window).sum()
            self.data[nom] = prod / poids_sum
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
        
    def plot_tag(self, tag):
        """
        Affiche les graphiques temporel et de distribution pour un ou plusieurs tags.
        IMPORTANT: Utiliser le(s) nom(s) final(aux) (renommé(s)) de(s) colonne(s).
        
        Parameters:
        tag : str or list, nom du/des tag(s) à afficher (nom(s) final(aux))
        """
        if self.data is None:
            print("Erreur : Aucune donnée disponible. Exécutez merge() d'abord.")
            return
        
        # Convertir en liste si c'est une chaîne
        tags = [tag] if isinstance(tag, str) else tag
        
        # Vérifier que tous les tags existent
        tags_valides = []
        for t in tags:
            if t not in self.data.columns:
                print(f"Attention : colonne '{t}' introuvable dans data")
            else:
                tags_valides.append(t)
        
        if not tags_valides:
            print(f"Erreur : Aucun tag valide trouvé")
            print(f"Colonnes disponibles : {list(self.data.columns)}")
            return
        
        # Couleurs pour différencier les tags
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
        for idx, tag_name in enumerate(tags_valides):
            color = colors[idx % len(colors)]
            
            # Calcul des statistiques
            data_series = self.data[tag_name].dropna()
            if len(data_series) == 0:
                print(f"Attention : tag '{tag_name}' ne contient aucune donnée valide")
                continue
                
            moyenne = data_series.mean()
            mediane = data_series.median()
            ecart_type = data_series.std()
            nb_valeurs = len(data_series)
            percentile_25 = data_series.quantile(0.25)
            percentile_75 = data_series.quantile(0.75)
            val_min = data_series.min()
            val_max = data_series.max()
            
            # Création de la figure avec 2 subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=(f'{tag_name} - Évolution temporelle', f'{tag_name} - Distribution'),
                horizontal_spacing=0.12,
                column_widths=[0.65, 0.35]
            )
            
            # === SUBPLOT 1 : Scatter plot ===
            # Points de données
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data[tag_name],
                    mode='markers',
                    name=tag_name,
                    marker=dict(color=color, opacity=0.3, size=6),
                    hovertemplate='Date: %{x}<br>Valeur: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Ligne de moyenne
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=[moyenne] * len(self.data.index),
                    mode='lines',
                    name=f'Moyenne: {moyenne:.2f}',
                    line=dict(color='red', dash='dash', width=2),
                    hovertemplate=f'Moyenne: {moyenne:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Ligne de médiane
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=[mediane] * len(self.data.index),
                    mode='lines',
                    name=f'Médiane: {mediane:.2f}',
                    line=dict(color='violet', dash='dash', width=2),
                    hovertemplate=f'Médiane: {mediane:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Courbe de tendance
            x = np.arange(len(self.data.index))
            y = self.data[tag_name].values
            
            # Filtrer les valeurs NaN
            mask = ~np.isnan(y)
            x_clean = x[mask]
            y_clean = y[mask]
            
            if len(x_clean) > 1:
                # Régression linéaire
                coeffs = np.polyfit(x_clean, y_clean, 1)
                line = np.polyval(coeffs, x)
                
                # Calcul du R²
                y_pred = np.polyval(coeffs, x_clean)
                ss_res = np.sum((y_clean - y_pred) ** 2)
                ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Tracer la ligne de tendance
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=line,
                        mode='lines',
                        name=f'Tendance (R²={r_squared:.3f})',
                        line=dict(color='green', width=2),
                        hovertemplate=f'Tendance (R²={r_squared:.3f})<br>Valeur: %{{y:.2f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # === SUBPLOT 2 : Histogramme avec KDE ===
            # Histogramme
            fig.add_trace(
                go.Histogram(
                    x=data_series,
                    name='Distribution',
                    marker=dict(color='lightblue', line=dict(color='darkblue', width=1)),
                    opacity=0.7,
                    histnorm='probability density',
                    hovertemplate='Valeur: %{x:.2f}<br>Densité: %{y:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Approximation KDE (Kernel Density Estimation)
            if len(data_series) > 1:
                kde = stats.gaussian_kde(data_series)
                x_range = np.linspace(data_series.min(), data_series.max(), 200)
                kde_values = kde(x_range)
                
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=kde_values,
                        mode='lines',
                        name='KDE',
                        line=dict(color='darkblue', width=2),
                        hovertemplate='Valeur: %{x:.2f}<br>Densité: %{y:.4f}<extra></extra>',
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                # Obtenir la hauteur max du KDE pour ajuster les lignes verticales
                y_max = max(kde_values) * 1.1
            else:
                y_max = 1
            
            # Lignes verticales pour moyenne et médiane
            fig.add_trace(
                go.Scatter(
                    x=[moyenne, moyenne],
                    y=[0, y_max],
                    mode='lines',
                    name=f'Moyenne',
                    line=dict(color='red', dash='dash', width=2),
                    showlegend=False,
                    hovertemplate=f'Moyenne: {moyenne:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[mediane, mediane],
                    y=[0, y_max],
                    mode='lines',
                    name=f'Médiane',
                    line=dict(color='violet', dash='dash', width=2),
                    showlegend=False,
                    hovertemplate=f'Médiane: {mediane:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Lignes verticales pour percentiles 25 et 75
            fig.add_trace(
                go.Scatter(
                    x=[percentile_25, percentile_25],
                    y=[0, y_max],
                    mode='lines',
                    name=f'P25',
                    line=dict(color='orange', dash='dot', width=1.5),
                    showlegend=False,
                    hovertemplate=f'Percentile 25: {percentile_25:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[percentile_75, percentile_75],
                    y=[0, y_max],
                    mode='lines',
                    name=f'P75',
                    line=dict(color='brown', dash='dot', width=1.5),
                    showlegend=False,
                    hovertemplate=f'Percentile 75: {percentile_75:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # === Ajout des statistiques dans la légende ===
            # Créer des traces invisibles pour afficher les stats dans la légende
            stats_text = [
                f"<b>Statistiques {tag_name}</b>",
                f"Nombre de valeurs: {nb_valeurs}",
                f"Moyenne: {moyenne:.2f}",
                f"Médiane: {mediane:.2f}",
                f"Écart-type: {ecart_type:.2f}",
                f"Min: {val_min:.2f}",
                f"Max: {val_max:.2f}",
                f"Percentile 25: {percentile_25:.2f}",
                f"Percentile 75: {percentile_75:.2f}"
            ]
            
            # Ajouter les statistiques comme traces invisibles pour la légende
            for i, stat in enumerate(stats_text):
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode='markers',
                        marker=dict(size=0),
                        name=stat,
                        showlegend=True,
                        hoverinfo='none'
                    ),
                    row=1, col=1
                )
            
            # Mise en page
            fig.update_layout(
                title_text=f"Analyse de {tag_name}",
                height=600,
                width=None,
                showlegend=True,
                hovermode='closest',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="gray",
                    borderwidth=1
                )
            )
            
            fig.update_xaxes(title_text="Date/Temps", row=1, col=1)
            fig.update_yaxes(title_text="Valeur", row=1, col=1)
            fig.update_xaxes(title_text="Valeur", row=1, col=2)
            fig.update_yaxes(title_text="Densité", row=1, col=2)
            
            fig.show()