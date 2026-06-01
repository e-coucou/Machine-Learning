"""
═══════════════════════════════════════════════════════════════════════════════
            PATTERNS AVANCÉS ET SNIPPETS RÉUTILISABLES
═══════════════════════════════════════════════════════════════════════════════

Ce fichier contient:
1. Patterns de code réutilisables
2. Fonctions utilitaires
3. Classe de rapport automatisé
4. Intégrations avec Excel, PDF
5. Notifications et alertes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import json


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 1: CLASSE DE RAPPORT AUTOMATISÉ
# ═══════════════════════════════════════════════════════════════════════════════

class RapportAutomate:
    """
    Génère un rapport de production automatisé et formaté
    
    Exemple:
        rapport = RapportAutomate(dashboard)
        rapport.generer_rapport_mensuel(mois=3, annee=2024)
        rapport.exporter_excel('rapport_mars_2024.xlsx')
    """
    
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.processor = dashboard.processor
        self.formatter = dashboard.formatter
        self.calculator = dashboard.calculator
        self.donnees_rapport = {}
    
    def generer_rapport_mensuel(self, mois: int, annee: int):
        """Génère un rapport complet pour le mois"""
        
        # 1. KPIs du mois
        kpis = self.calculator.calculer_kpis_mensuel(mois, annee)
        self.donnees_rapport['kpis'] = kpis
        
        # 2. Bilan détaillé par produit
        bilan = self.formatter.format_bilan_mensuel(mois, annee)
        self.donnees_rapport['bilan'] = bilan
        
        # 3. Données brutes du mois
        df = self.processor.data
        mask = (df.index.month == mois) & (df.index.year == annee)
        df_mois = df[mask]
        self.donnees_rapport['df_mois'] = df_mois
        
        # 4. Statistiques par produit
        stats = self._calculer_stats_mois(df_mois, mois, annee)
        self.donnees_rapport['stats'] = stats
        
        # 5. Top et bottom jours
        self.donnees_rapport['meilleur_jour'] = self._trouver_meilleur_jour(df_mois, mois, annee)
        self.donnees_rapport['pire_jour'] = self._trouver_pire_jour(df_mois, mois, annee)
        
        return self
    
    def _calculer_stats_mois(self, df_mois: pd.DataFrame, mois: int, annee: int) -> Dict:
        """Calcule les statistiques détaillées du mois"""
        stats = {}
        produits = self.dashboard.get_produits()
        
        for prod in produits:
            col_prod = f'production_{prod}'
            if col_prod not in df_mois.columns:
                continue
            
            data_prod = df_mois[col_prod].dropna()
            if len(data_prod) == 0:
                continue
            
            stats[prod] = {
                'moyenne': data_prod.mean(),
                'min': data_prod.min(),
                'max': data_prod.max(),
                'std': data_prod.std(),
                'total': data_prod.sum(),
                'cv': (data_prod.std() / data_prod.mean() * 100) if data_prod.mean() != 0 else 0
            }
        
        return stats
    
    def _trouver_meilleur_jour(self, df_mois: pd.DataFrame, mois: int, annee: int) -> Tuple[int, float]:
        """Trouve le jour avec la production la plus élevée"""
        produits = self.dashboard.get_produits()
        meilleur_jour = None
        meilleur_val = -np.inf
        
        for jour in range(1, 32):
            try:
                bilan = self.formatter.format_bilan_journalier(jour, mois, annee)
                prod_jour = sum(b.production for b in bilan.values())
                
                if prod_jour > meilleur_val:
                    meilleur_val = prod_jour
                    meilleur_jour = jour
            except:
                continue
        
        return meilleur_jour, meilleur_val
    
    def _trouver_pire_jour(self, df_mois: pd.DataFrame, mois: int, annee: int) -> Tuple[int, float]:
        """Trouve le jour avec la production la plus basse"""
        produits = self.dashboard.get_produits()
        pire_jour = None
        pire_val = np.inf
        
        for jour in range(1, 32):
            try:
                bilan = self.formatter.format_bilan_journalier(jour, mois, annee)
                prod_jour = sum(b.production for b in bilan.values())
                
                if prod_jour < pire_val:
                    pire_val = prod_jour
                    pire_jour = jour
            except:
                continue
        
        return pire_jour, pire_val
    
    def exporter_excel(self, nom_fichier: str):
        """Exporte le rapport en Excel avec mise en forme"""
        
        try:
            with pd.ExcelWriter(nom_fichier, engine='openpyxl') as writer:
                
                # Sheet 1: Résumé KPIs
                df_kpis = pd.DataFrame([
                    {'Métrique': kpi.nom, 'Valeur': kpi.valeur, 'Unité': kpi.unite}
                    for kpi in self.donnees_rapport.get('kpis', [])
                ])
                df_kpis.to_excel(writer, sheet_name='KPIs', index=False)
                
                # Sheet 2: Bilan par produit
                bilan = self.donnees_rapport.get('bilan', {})
                df_bilan = pd.DataFrame([
                    {
                        'Produit': b.produit,
                        'Consommation': b.consommation,
                        'Variation Stock': b.delta_stock,
                        'Production': b.production
                    }
                    for b in bilan.values()
                ])
                df_bilan.to_excel(writer, sheet_name='Bilan', index=False)
                
                # Sheet 3: Statistiques
                stats = self.donnees_rapport.get('stats', {})
                df_stats_list = []
                for prod, stat in stats.items():
                    df_stats_list.append({
                        'Produit': prod,
                        'Moyenne': stat['moyenne'],
                        'Min': stat['min'],
                        'Max': stat['max'],
                        'Écart-type': stat['std'],
                        'CV (%)': stat['cv'],
                        'Total': stat['total']
                    })
                df_stats = pd.DataFrame(df_stats_list)
                df_stats.to_excel(writer, sheet_name='Stats', index=False)
                
                # Sheet 4: Données brutes
                df_mois = self.donnees_rapport.get('df_mois', pd.DataFrame())
                df_mois.to_excel(writer, sheet_name='Données Brutes')
            
            print(f"✓ Rapport exporté: {nom_fichier}")
            return True
        
        except Exception as e:
            print(f"✗ Erreur export Excel: {e}")
            return False
    
    def afficher_resume(self):
        """Affiche un résumé console du rapport"""
        
        print("\n" + "="*70)
        print("  RÉSUMÉ DU RAPPORT")
        print("="*70)
        
        # KPIs
        print("\n📊 KPIs PRINCIPAUX:")
        for kpi in self.donnees_rapport.get('kpis', []):
            print(f"  {kpi.nom}: {kpi.valeur:.2f} {kpi.unite}")
        
        # Statistiques
        print("\n📈 STATISTIQUES PAR PRODUIT:")
        stats = self.donnees_rapport.get('stats', {})
        for prod, stat in stats.items():
            print(f"\n  {prod}:")
            print(f"    Moyenne: {stat['moyenne']:.2f}")
            print(f"    Min/Max: {stat['min']:.2f} / {stat['max']:.2f}")
            print(f"    Variation: {stat['cv']:.1f}%")
            print(f"    Total: {stat['total']:.2f}")
        
        # Jours remarquables
        print("\n⭐ JOURS REMARQUABLES:")
        meilleur_jour, meilleur_val = self.donnees_rapport.get('meilleur_jour', (None, 0))
        pire_jour, pire_val = self.donnees_rapport.get('pire_jour', (None, 0))
        
        if meilleur_jour:
            print(f"  Meilleur jour: {meilleur_jour:02d} ({meilleur_val:.2f})")
        if pire_jour:
            print(f"  Pire jour: {pire_jour:02d} ({pire_val:.2f})")
        
        print("\n" + "="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 2: SYSTÈME D'ALERTE
# ═══════════════════════════════════════════════════════════════════════════════

class SystemeAlerte:
    """
    Génère des alertes basées sur seuils personnalisés
    
    Exemple:
        alerte = SystemeAlerte(dashboard)
        alerte.ajouter_seuil('production_Acetate', min=100, max=500)
        alerte.verifier_alertes()
    """
    
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.processor = dashboard.processor
        self.seuils = {}
        self.alertes = []
    
    def ajouter_seuil(self, metrique: str, min_val: Optional[float] = None, 
                      max_val: Optional[float] = None, nom_alerte: str = None):
        """Ajoute un seuil d'alerte"""
        
        if nom_alerte is None:
            nom_alerte = f"Seuil_{metrique}"
        
        self.seuils[nom_alerte] = {
            'metrique': metrique,
            'min': min_val,
            'max': max_val
        }
    
    def verifier_alertes(self) -> List[Dict]:
        """Vérifie les seuils et génère les alertes"""
        
        self.alertes = []
        df = self.processor.data
        
        for nom_seuil, config in self.seuils.items():
            metrique = config['metrique']
            
            if metrique not in df.columns:
                continue
            
            data = df[metrique].dropna()
            min_seuil = config['min']
            max_seuil = config['max']
            
            # Vérification min
            if min_seuil is not None:
                valeurs_basses = data[data < min_seuil]
                if len(valeurs_basses) > 0:
                    self.alertes.append({
                        'type': 'ALERTE MIN',
                        'metrique': metrique,
                        'seuil': min_seuil,
                        'violations': len(valeurs_basses),
                        'pire_valeur': valeurs_basses.min(),
                        'date_pire': valeurs_basses.idxmin()
                    })
            
            # Vérification max
            if max_seuil is not None:
                valeurs_hautes = data[data > max_seuil]
                if len(valeurs_hautes) > 0:
                    self.alertes.append({
                        'type': 'ALERTE MAX',
                        'metrique': metrique,
                        'seuil': max_seuil,
                        'violations': len(valeurs_hautes),
                        'pire_valeur': valeurs_hautes.max(),
                        'date_pire': valeurs_hautes.idxmax()
                    })
        
        return self.alertes
    
    def afficher_alertes(self):
        """Affiche les alertes de manière formatée"""
        
        if not self.alertes:
            print("✓ Aucune alerte")
            return
        
        print("\n" + "⚠️ "  * 35)
        print("  ALERTES DÉTECTÉES")
        print("⚠️ " * 35 + "\n")
        
        for alerte in self.alertes:
            couleur = "🔴" if alerte['type'] == 'ALERTE MIN' else "🟠"
            print(f"{couleur} {alerte['type']}")
            print(f"   Métrique: {alerte['metrique']}")
            print(f"   Seuil: {alerte['seuil']}")
            print(f"   Violations: {alerte['violations']}")
            print(f"   Pire valeur: {alerte['pire_valeur']:.2f}")
            print(f"   Date: {alerte['date_pire']}")
            print()


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 3: UTILITAIRES DE COMPARAISON
# ═══════════════════════════════════════════════════════════════════════════════

def comparer_periodes(dashboard, periode1: Tuple[int, int], periode2: Tuple[int, int]) -> pd.DataFrame:
    """
    Compare deux périodes (mois)
    
    Exemple:
        df_comp = comparer_periodes(dashboard, (3, 2024), (2, 2024))
        print(df_comp)
    """
    
    formatter = dashboard.formatter
    calculator = dashboard.calculator
    
    bilan1 = formatter.format_bilan_mensuel(periode1[0], periode1[1])
    bilan2 = formatter.format_bilan_mensuel(periode2[0], periode2[1])
    
    comparison = []
    
    for prod in dashboard.get_produits():
        b1 = bilan1.get(prod)
        b2 = bilan2.get(prod)
        
        if not b1 or not b2:
            continue
        
        variation_prod = ((b2.production - b1.production) / b1.production * 100) if b1.production != 0 else 0
        variation_conso = ((b2.consommation - b1.consommation) / b1.consommation * 100) if b1.consommation != 0 else 0
        
        comparison.append({
            'Produit': prod,
            'Prod P1': b1.production,
            'Prod P2': b2.production,
            'Var Production %': variation_prod,
            'Conso P1': b1.consommation,
            'Conso P2': b2.consommation,
            'Var Consommation %': variation_conso,
            'Stock P1': b1.delta_stock,
            'Stock P2': b2.delta_stock
        })
    
    return pd.DataFrame(comparison)


def calculer_tendance(dashboard, num_periodes: int = 6) -> Dict:
    """
    Calcule la tendance sur N périodes
    
    Exemple:
        tendances = calculer_tendance(dashboard, num_periodes=6)
        for prod, trend in tendances.items():
            print(f"{prod}: {trend['direction']}")
    """
    
    formatter = dashboard.formatter
    min_date, max_date = dashboard.get_range_dates()
    
    tendances = {}
    
    for prod in dashboard.get_produits():
        productions = []
        
        # Récupérer les productions des N derniers mois
        date_courant = max_date
        for i in range(num_periodes):
            mois = date_courant.month - i
            annee = date_courant.year
            
            if mois <= 0:
                mois += 12
                annee -= 1
            
            try:
                bilan = formatter.format_bilan_mensuel(mois, annee, prod)
                if prod in bilan:
                    productions.append(bilan[prod].production)
            except:
                pass
        
        if len(productions) > 1:
            # Régression linéaire simple
            x = np.arange(len(productions))
            z = np.polyfit(x, productions, 1)
            direction = "Hausse" if z[0] > 0 else "Baisse"
            magnitude = abs(z[0])
            
            tendances[prod] = {
                'direction': direction,
                'pente': z[0],
                'magnitude': magnitude,
                'productions': productions
            }
    
    return tendances


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 4: EXPORT JSON
# ═══════════════════════════════════════════════════════════════════════════════

def exporter_json(dashboard, nom_fichier: str, mois: int, annee: int):
    """
    Exporte les données du dashboard en JSON structuré
    
    Exemple:
        exporter_json(dashboard, 'rapport.json', mois=3, annee=2024)
    """
    
    formatter = dashboard.formatter
    calculator = dashboard.calculator
    
    # Collecter les données
    kpis = calculator.calculer_kpis_mensuel(mois, annee)
    bilan = formatter.format_bilan_mensuel(mois, annee)
    
    # Structurer pour JSON
    data = {
        'meta': {
            'mois': mois,
            'annee': annee,
            'date_generation': datetime.now().isoformat(),
            'produits': dashboard.get_produits()
        },
        'kpis': [
            {
                'nom': kpi.nom,
                'valeur': float(kpi.valeur),
                'unite': kpi.unite,
                'variation': float(kpi.variation) if kpi.variation else None
            }
            for kpi in kpis
        ],
        'bilan_par_produit': {
            prod: {
                'periode': b.periode,
                'consommation': float(b.consommation),
                'delta_stock': float(b.delta_stock),
                'production': float(b.production),
                'est_termine': b.est_termine
            }
            for prod, b in bilan.items()
        }
    }
    
    # Sauvegarder
    with open(nom_fichier, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Données exportées: {nom_fichier}")


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN 5: CLASSE DE FILTRAGE AVANCÉ
# ═══════════════════════════════════════════════════════════════════════════════

class FiltreAvance:
    """
    Filtre et agrège les données selon des critères complexes
    
    Exemple:
        filtre = FiltreAvance(dashboard)
        df_filtre = filtre.par_plage_dates('2024-03-01', '2024-03-31') \\
                          .par_produit('Acetate') \\
                          .par_seuil('production', min=100) \\
                          .executer()
    """
    
    def __init__(self, dashboard):
        self.dashboard = dashboard
        self.df = dashboard.processor.data.copy()
        self.filtres = []
    
    def par_plage_dates(self, start: str, end: str):
        """Filtre par plage de dates"""
        self.filtres.append(
            lambda df: df.loc[(df.index >= start) & (df.index <= end)]
        )
        return self
    
    def par_produit(self, nom_produit: str):
        """Filtre par produit (garde seulement les colonnes du produit)"""
        def filtre(df):
            cols = [c for c in df.columns if nom_produit in c]
            return df[cols]
        self.filtres.append(filtre)
        return self
    
    def par_seuil(self, colonne: str, min_val: float = None, max_val: float = None):
        """Filtre par seuils"""
        def filtre(df):
            if colonne not in df.columns:
                return df
            if min_val is not None:
                df = df[df[colonne] >= min_val]
            if max_val is not None:
                df = df[df[colonne] <= max_val]
            return df
        self.filtres.append(filtre)
        return self
    
    def resample(self, freq: str):
        """Rééchantillonne les données"""
        def filtre(df):
            return df.resample(freq).mean()
        self.filtres.append(filtre)
        return self
    
    def executer(self) -> pd.DataFrame:
        """Exécute la chaîne de filtres"""
        df_result = self.df.copy()
        for filtre in self.filtres:
            df_result = filtre(df_result)
        return df_result


# ═══════════════════════════════════════════════════════════════════════════════
# EXEMPLES D'UTILISATION
# ═══════════════════════════════════════════════════════════════════════════════

"""
# EXEMPLE 1: Rapport automatisé
────────────────────────────────

    rapport = RapportAutomate(dashboard)
    rapport.generer_rapport_mensuel(mois=3, annee=2024)
    rapport.afficher_resume()
    rapport.exporter_excel('rapport_mars_2024.xlsx')


# EXEMPLE 2: Système d'alerte
────────────────────────────

    alerte = SystemeAlerte(dashboard)
    alerte.ajouter_seuil('production_Acetate', min=100, max=500)
    alerte.ajouter_seuil('stock_Acetate', max=1000)
    alerte.verifier_alertes()
    alerte.afficher_alertes()


# EXEMPLE 3: Comparaison de périodes
──────────────────────────────────

    df_comp = comparer_periodes(dashboard, (3, 2024), (2, 2024))
    print(df_comp)
    
    # Visualiser
    import plotly.graph_objects as go
    fig = go.Figure()
    for _, row in df_comp.iterrows():
        fig.add_trace(go.Bar(
            x=['P1', 'P2'],
            y=[row['Prod P1'], row['Prod P2']],
            name=row['Produit']
        ))
    fig.show()


# EXEMPLE 4: Filtrage avancé
───────────────────────────

    df_filtre = FiltreAvance(dashboard) \\
        .par_plage_dates('2024-03-01', '2024-03-31') \\
        .par_produit('Acetate') \\
        .resample('D') \\
        .executer()
    
    print(df_filtre)


# EXEMPLE 5: Export JSON
───────────────────────

    exporter_json(dashboard, 'rapport_production.json', mois=3, annee=2024)
    
    # Charger après
    import json
    with open('rapport_production.json') as f:
        data = json.load(f)
    print(data)


# EXEMPLE 6: Workflow complet
──────────────────────────────

    # 1. Créer le rapport
    rapport = RapportAutomate(dashboard)
    rapport.generer_rapport_mensuel(mois=3, annee=2024)
    
    # 2. Vérifier les alertes
    alerte = SystemeAlerte(dashboard)
    alerte.ajouter_seuil('production_Acetate', min=50, max=500)
    alerte.verifier_alertes()
    
    # 3. Comparer avec mois précédent
    df_comp = comparer_periodes(dashboard, (3, 2024), (2, 2024))
    
    # 4. Exporter tout
    rapport.afficher_resume()
    alerte.afficher_alertes()
    
    rapport.exporter_excel('rapport_mars.xlsx')
    exporter_json(dashboard, 'rapport_mars.json', mois=3, annee=2024)
    
    print("✓ Rapport complet généré!")
"""