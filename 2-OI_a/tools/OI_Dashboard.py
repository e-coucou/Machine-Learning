"""
DASHBOARD DE PRODUCTION - Classes pour Jupyter
===============================================
Orchestration complète : données → formatage → métriques → visualisations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import calendar

warnings.filterwarnings('ignore')


# ==================== CLASSES DE DONNÉES ====================

@dataclass
class KPI:
    """Classe pour stocker une métrique clé"""
    nom: str
    valeur: float
    unite: str
    variation: Optional[float] = None  # % de variation
    couleur: Optional[str] = None  # Pour le rendu
    
    def __str__(self):
        var_str = f" ({self.variation:+.1f}%)" if self.variation is not None else ""
        return f"{self.nom}: {self.valeur:.2f} {self.unite}{var_str}"


@dataclass
class BilanProduit:
    """Structure pour un bilan de produit (jour/mois)"""
    produit: str
    periode: str
    consommation: float
    delta_stock: float
    production: float
    timestamp_start: pd.Timestamp
    timestamp_end: pd.Timestamp
    est_termine: bool = True


# ==================== CLASSE DE FORMATAGE ====================

class DashboardFormatter:
    """Formate et prépare les données pour le dashboard"""
    
    def __init__(self, processor):
        """
        Parameters:
        -----------
        processor : OI_ProductionProcessor
            Instance du processeur de données
        """
        self.processor = processor
        self.df = processor.data
        
    def get_periodes_disponibles(self) -> Dict[str, List[str]]:
        """Retourne les mois et jours disponibles dans les données"""
        if self.df is None or self.df.empty:
            return {}
        
        dates = self.df.index
        mois_dict = {}
        
        for date in dates:
            key = f"{date.year}-{date.month:02d}"
            if key not in mois_dict:
                mois_dict[key] = []
            day = f"{date.day:02d}"
            if day not in mois_dict[key]:
                mois_dict[key].append(day)
        
        return mois_dict
    
    def get_range_dates(self) -> Tuple[datetime, datetime]:
        """Retourne la plage de dates disponibles"""
        if self.df is None or self.df.empty:
            return None, None
        return self.df.index.min(), self.df.index.max()
    
    def get_produits(self) -> List[str]:
        """Retourne la liste des produits"""
        return [p['nom'] for p in self.processor.produits]
    
    def format_bilan_journalier(self, jour: int, mois: int = None, annee: int = None, 
                                 nom_produit: str = None, std: int = 2) -> Dict[str, BilanProduit]:
        """
        Récupère et formate un bilan journalier
        
        Returns:
        --------
        Dict[str, BilanProduit] : {nom_produit: BilanProduit}
        """
        bilan_raw = self.processor.calcul_cumul_journalier(
            jour, mois, annee, nom_produit, std=std
        )
        
        if bilan_raw is None:
            return {}
        
        # Récupérer les timestamps réels
        month_num = mois if isinstance(mois, int) else (self.df.index.month[0] if mois is None else 1)
        year_num = annee if annee else self.df.index.year[0]
        
        bilan_formatted = {}
        for prod_name, metrics in bilan_raw.items():
            bilan = BilanProduit(
                produit=prod_name,
                periode=f"{jour:02d}/{month_num:02d}/{year_num}",
                consommation=metrics['consommation'],
                delta_stock=metrics['delta_stock'],
                production=metrics['production'],
                timestamp_start=self.df.index.min(),
                timestamp_end=self.df.index.max(),
                est_termine=True
            )
            bilan_formatted[prod_name] = bilan
        
        return bilan_formatted
    
    def format_bilan_mensuel(self, mois: int = None, annee: int = None, 
                              nom_produit: str = None) -> Dict[str, BilanProduit]:
        """Récupère et formate un bilan mensuel"""
        bilan_raw = self.processor.calcul_cumul_mensuel(
            mois, annee, nom_produit, std=2
        )
        
        if bilan_raw is None:
            return {}
        
        month_num = mois if isinstance(mois, int) else self.df.index.month[0]
        year_num = annee if annee else self.df.index.year[0]
        nom_mois = calendar.month_name[month_num]
        
        bilan_formatted = {}
        for prod_name, metrics in bilan_raw.items():
            bilan = BilanProduit(
                produit=prod_name,
                periode=f"{nom_mois} {year_num}",
                consommation=metrics['consommation'],
                delta_stock=metrics['delta_stock'],
                production=metrics['production'],
                timestamp_start=self.df.index.min(),
                timestamp_end=self.df.index.max(),
                est_termine=True
            )
            bilan_formatted[prod_name] = bilan
        
        return bilan_formatted


# ==================== CLASSE DE CALCUL DE MÉTRIQUES ====================

class MetricsCalculator:
    """Calcule les KPIs et métriques du dashboard"""
    
    def __init__(self, processor):
        self.processor = processor
        self.df = processor.data
    
    def calculer_kpis_journalier(self, jour: int, mois: int = None, annee: int = None) -> List[KPI]:
        """Calcule les KPIs pour un jour spécifique"""
        formatter = DashboardFormatter(self.processor)
        bilan = formatter.format_bilan_journalier(jour, mois, annee)
        
        kpis = []
        
        # KPI Global : Production totale
        prod_totale = sum(b.production for b in bilan.values())
        kpis.append(KPI(
            nom="Production Totale",
            valeur=prod_totale,
            unite="unités",
            couleur="#4CAF50"
        ))
        
        # KPI Global : Consommation totale
        conso_totale = sum(b.consommation for b in bilan.values())
        kpis.append(KPI(
            nom="Consommation",
            valeur=conso_totale,
            unite="unités",
            couleur="#FF9800"
        ))
        
        # KPI Global : Variation de stock
        var_stock = sum(b.delta_stock for b in bilan.values())
        couleur_stock = "#4CAF50" if var_stock >= 0 else "#F44336"
        kpis.append(KPI(
            nom="Variation Stock",
            valeur=var_stock,
            unite="unités",
            couleur=couleur_stock
        ))
        
        return kpis
    
    def calculer_kpis_mensuel(self, mois: int = None, annee: int = None) -> List[KPI]:
        """Calcule les KPIs pour un mois spécifique"""
        formatter = DashboardFormatter(self.processor)
        bilan = formatter.format_bilan_mensuel(mois, annee)
        
        kpis = []
        
        prod_totale = sum(b.production for b in bilan.values())
        kpis.append(KPI(
            nom="Production Mensuelle",
            valeur=prod_totale,
            unite="unités",
            couleur="#4CAF50"
        ))
        
        conso_totale = sum(b.consommation for b in bilan.values())
        kpis.append(KPI(
            nom="Consommation Mensuelle",
            valeur=conso_totale,
            unite="unités",
            couleur="#FF9800"
        ))
        
        var_stock = sum(b.delta_stock for b in bilan.values())
        couleur_stock = "#4CAF50" if var_stock >= 0 else "#F44336"
        kpis.append(KPI(
            nom="Variation Stock",
            valeur=var_stock,
            unite="unités",
            couleur=couleur_stock
        ))
        
        return kpis
    
    def calculer_efficacite(self, nom_produit: str, jour: int, mois: int = None, 
                            annee: int = None) -> float:
        """Calcule l'efficacité de production (Production / Consommation)"""
        formatter = DashboardFormatter(self.processor)
        bilan = formatter.format_bilan_journalier(jour, mois, annee, nom_produit)
        
        if nom_produit not in bilan:
            return 0.0
        
        b = bilan[nom_produit]
        if b.consommation == 0:
            return 0.0
        
        return (b.production / b.consommation) * 100
    
    def comparaison_jours(self, num_jours: int = 7) -> pd.DataFrame:
        """Compare la production des N derniers jours"""
        if self.df is None or self.df.empty:
            return pd.DataFrame()
        
        last_date = self.df.index.max()
        comparaisons = []
        
        for i in range(num_jours):
            date_i = last_date - timedelta(days=i)
            bilan = DashboardFormatter(self.processor).format_bilan_journalier(
                date_i.day, date_i.month, date_i.year
            )
            
            for prod, b in bilan.items():
                comparaisons.append({
                    'date': date_i.date(),
                    'produit': prod,
                    'production': b.production,
                    'consommation': b.consommation,
                    'stock': b.delta_stock
                })
        
        return pd.DataFrame(comparaisons)


# ==================== CLASSE DE VISUALISATION ====================

class DashboardVisualizer:
    """Crée les visualisations pour le dashboard"""
    
    def __init__(self, processor, formatter: DashboardFormatter, calculator: MetricsCalculator):
        self.processor = processor
        self.formatter = formatter
        self.calculator = calculator
        self.df = processor.data
    
    def afficher_kpis(self, kpis: List[KPI], titre: str = "KPIs"):
        """Affiche les KPIs de manière élégante"""
        print("\n" + "="*70)
        print(f"  {titre.upper()}")
        print("="*70)
        for kpi in kpis:
            var_str = f" ({kpi.variation:+.1f}%)" if kpi.variation else ""
            print(f"  {kpi.nom:30s} : {kpi.valeur:12.2f} {kpi.unite:15s}{var_str}")
        print("="*70 + "\n")
    
    def plot_bilan_journalier(self, jour: int, mois: int = None, annee: int = None):
        """Graphique de bilan journalier par produit"""
        bilan = self.formatter.format_bilan_journalier(jour, mois, annee)
        
        if not bilan:
            print("Aucune donnée disponible")
            return
        
        produits = list(bilan.keys())
        consommations = [bilan[p].consommation for p in produits]
        stocks = [bilan[p].delta_stock for p in produits]
        productions = [bilan[p].production for p in produits]
        
        fig = go.Figure(data=[
            go.Bar(name='Consommation', x=produits, y=consommations, marker_color='#FF9800'),
            go.Bar(name='Variation Stock', x=produits, y=stocks, marker_color='#2196F3'),
            go.Bar(name='Production', x=produits, y=productions, marker_color='#4CAF50')
        ])
        
        fig.update_layout(
            barmode='group',
            title=f"Bilan de Production - {jour:02d}/{mois or 'MM'}/{annee or 'YYYY'}",
            xaxis_title="Produits",
            yaxis_title="Quantité (unités)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        fig.show()
    
    def plot_bilan_mensuel(self, mois: int = None, annee: int = None):
        """Graphique de bilan mensuel par produit"""
        bilan = self.formatter.format_bilan_mensuel(mois, annee)
        
        if not bilan:
            print("Aucune donnée disponible")
            return
        
        produits = list(bilan.keys())
        consommations = [bilan[p].consommation for p in produits]
        stocks = [bilan[p].delta_stock for p in produits]
        productions = [bilan[p].production for p in produits]
        
        fig = go.Figure(data=[
            go.Bar(name='Consommation', x=produits, y=consommations, marker_color='#FF9800'),
            go.Bar(name='Variation Stock', x=produits, y=stocks, marker_color='#2196F3'),
            go.Bar(name='Production', x=produits, y=productions, marker_color='#4CAF50')
        ])
        
        month_name = calendar.month_name[mois]
        fig.update_layout(
            barmode='group',
            title=f"Bilan de Production - {month_name} {annee or 'YYYY'}",
            xaxis_title="Produits",
            yaxis_title="Quantité (unités)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        fig.show()
    
    def plot_comparaison_produits(self, nom_produit: str):
        """Graphique de série temporelle pour un produit"""
        if self.df is None or self.df.empty:
            print("Aucune donnée disponible")
            return
        
        cols_to_plot = [
            f"consommation_{nom_produit}",
            f"stock_{nom_produit}",
            f"production_{nom_produit}"
        ]
        
        cols_valides = [c for c in cols_to_plot if c in self.df.columns]
        
        if not cols_valides:
            print(f"Aucune donnée pour le produit '{nom_produit}'")
            return
        
        fig = make_subplots(specs=[[{"secondary_y": False}]])
        
        colors = {'consommation': '#FF9800', 'stock': '#2196F3', 'production': '#4CAF50'}
        
        for col in cols_valides:
            label = col.replace(f"_{nom_produit}", "").replace("_", " ").title()
            color_key = col.split('_')[0]
            
            fig.add_trace(
                go.Scatter(
                    x=self.df.index,
                    y=self.df[col],
                    mode='lines',
                    name=label,
                    line=dict(color=colors.get(color_key, '#000000'), width=2),
                    fill='tozeroy' if col == cols_valides[0] else None,
                    opacity=0.7
                )
            )
        
        fig.update_layout(
            title=f"Évolution - Produit {nom_produit}",
            xaxis_title="Date/Temps",
            yaxis_title="Quantité (unités)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        fig.show()
    
    def plot_tableau_comparaison(self, num_jours: int = 7):
        """Tableau de comparaison sur les N derniers jours"""
        df_comp = self.calculator.comparaison_jours(num_jours)
        
        if df_comp.empty:
            print("Aucune donnée disponible")
            return
        
        # Créer un graphique type heatmap
        pivot = df_comp.pivot_table(
            index='date',
            columns='produit',
            values='production',
            aggfunc='sum'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='YlOrRd',
            colorbar=dict(title="Production")
        ))
        
        fig.update_layout(
            title=f"Production sur {num_jours} jours",
            xaxis_title="Produits",
            yaxis_title="Dates",
            height=400
        )
        
        fig.show()


# ==================== CLASSE ORCHESTRATRICE ====================

class ProductionDashboard:
    """
    Classe principale du Dashboard - Orchestration complète
    Utilisation : 
        dashboard = ProductionDashboard(processor)
        dashboard.afficher_kpis_journaliers(jour=15, mois=3)
        dashboard.afficher_bilan_produit('Acetate')
    """
    
    def __init__(self, processor):
        """
        Parameters:
        -----------
        processor : OI_ProductionProcessor
            Instance du processeur de production
        """
        self.processor = processor
        self.formatter = DashboardFormatter(processor)
        self.calculator = MetricsCalculator(processor)
        self.visualizer = DashboardVisualizer(processor, self.formatter, self.calculator)
        
        print("✓ Dashboard initialisé")
        print(f"  Produits: {', '.join(self.formatter.get_produits())}")
        min_date, max_date = self.formatter.get_range_dates()
        if min_date and max_date:
            print(f"  Période: {min_date.date()} → {max_date.date()}")
    
    def afficher_kpis_journaliers(self, jour: int, mois: int = None, annee: int = None):
        """Affiche les KPIs pour un jour spécifique"""
        kpis = self.calculator.calculer_kpis_journalier(jour, mois, annee)
        self.visualizer.afficher_kpis(kpis, 
            titre=f"KPIs Journaliers - {jour:02d}/{mois or 'MM'}/{annee or 'YYYY'}")
        return kpis
    
    def afficher_kpis_mensuels(self, mois: int = None, annee: int = None):
        """Affiche les KPIs pour un mois spécifique"""
        kpis = self.calculator.calculer_kpis_mensuel(mois, annee)
        month_name = calendar.month_name[mois]
        self.visualizer.afficher_kpis(kpis, 
            titre=f"KPIs Mensuels - {month_name} {annee or 'YYYY'}")
        return kpis
    
    def afficher_bilan_journalier(self, jour: int, mois: int = None, annee: int = None):
        """Affiche le bilan graphique d'une journée"""
        self.visualizer.plot_bilan_journalier(jour, mois, annee)
    
    def afficher_bilan_mensuel(self, mois: int = None, annee: int = None):
        """Affiche le bilan graphique d'un mois"""
        self.visualizer.plot_bilan_mensuel(mois, annee)
    
    def afficher_bilan_produit(self, nom_produit: str):
        """Affiche la série temporelle d'un produit"""
        self.visualizer.plot_comparaison_produits(nom_produit)
    
    def afficher_comparaison_jours(self, num_jours: int = 7):
        """Affiche une heatmap de comparaison sur N jours"""
        self.visualizer.plot_tableau_comparaison(num_jours)
    
    def get_periodes_disponibles(self) -> Dict:
        """Retourne les périodes disponibles"""
        return self.formatter.get_periodes_disponibles()
    
    def get_produits(self) -> List[str]:
        """Retourne la liste des produits"""
        return self.formatter.get_produits()
    
    def get_range_dates(self) -> Tuple:
        """Retourne la plage de dates"""
        return self.formatter.get_range_dates()
    
    def resume_complet(self):
        """Affiche un résumé complet du dashboard"""
        print("\n" + "="*70)
        print("  RÉSUMÉ COMPLET DU DASHBOARD")
        print("="*70)
        
        min_date, max_date = self.get_range_dates()
        print(f"\n📅 Période de données: {min_date.date()} → {max_date.date()}")
        print(f"📦 Produits disponibles: {', '.join(self.get_produits())}")
        
        print("\n" + "="*70)
        print("  EXEMPLE D'UTILISATION")
        print("="*70)
        print("""
# Afficher les KPIs d'un jour
dashboard.afficher_kpis_journaliers(jour=15, mois=3, annee=2024)

# Afficher les KPIs d'un mois
dashboard.afficher_kpis_mensuels(mois=3, annee=2024)

# Voir le bilan graphique d'une journée
dashboard.afficher_bilan_journalier(jour=15, mois=3)

# Voir l'évolution d'un produit
dashboard.afficher_bilan_produit('Acetate')

# Comparer les N derniers jours
dashboard.afficher_comparaison_jours(num_jours=7)
        """)
        print("="*70 + "\n")