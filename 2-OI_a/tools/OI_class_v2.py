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
                fig.update_yaxes(title_text=unit, row=1, col=1, secondary_y=False)
                # Forcer l'affichage du bouton de reset et les interactions
                fig.show(config={
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraselayer'],
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'scrollZoom': False,  # Zoom à la molette activé
                })
                #fig.show()