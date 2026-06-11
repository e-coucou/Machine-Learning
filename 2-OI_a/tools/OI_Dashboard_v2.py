"""
═══════════════════════════════════════════════════════════════════════════════
    EXTENSION PRODUCTONDASHBOARD - Avec visualisations avancées
═══════════════════════════════════════════════════════════════════════════════

Extension du ProductionDashboard de base avec 3 nouvelles méthodes:
1. plot_histogramme_tous_produits() - Histogramme comparatif tous produits
2. plot_waterfall_mois() - Waterfall évolution journalière du mois
3. plot_histogramme_jours_mois() - Histogramme production par jour

À importer ET utiliser avec ProductionDashboard existant:

from OI_Dashboard import ProductionDashboard
from EXTENSION_Dashboard_Avance import AjouterVisualisationsAvancees

dashboard = ProductionDashboard(processor)
AjouterVisualisationsAvancees(dashboard)  # Ajouter les méthodes

# Maintenant utiliser:
dashboard.plot_histogramme_tous_produits(mois=3)
dashboard.plot_waterfall_mois(mois=3)
dashboard.plot_histogramme_jours_mois(mois=3)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar


def AjouterVisualisationsAvancees(dashboard):
    """
    Ajoute les méthodes de visualisation avancées au dashboard existant
    
    Parameters:
    -----------
    dashboard : ProductionDashboard
        Instance du dashboard
    
    Example:
        from OI_Dashboard import ProductionDashboard
        from EXTENSION_Dashboard_Avance import AjouterVisualisationsAvancees
        
        dashboard = ProductionDashboard(processor)
        AjouterVisualisationsAvancees(dashboard)
        
        dashboard.plot_histogramme_tous_produits(mois=3)
    """
    
    # ═════════════════════════════════════════════════════════════════════════════
    # MÉTHODE 1: HISTOGRAMME - TOUS LES PRODUITS
    # ═════════════════════════════════════════════════════════════════════════════
    
    def plot_histogramme_tous_produits(mois=None, annea=None):
        """
        Histogramme comparant tous les produits (Production, Consommation, Stock)
        
        Parameters:
        -----------
        mois : int, optionnel (1-12)
        annea : int, optionnel
        """
        
        df = dashboard.processor.data
        if df is None or df.empty:
            print("❌ Aucune donnée disponible")
            return
        
        # Résoudre paramètres
        if mois is None:
            mois = df.index.month[0]
        if annea is None:
            annea = df.index.year[0]
        
        # Récupérer bilan
        bilan = dashboard.formatter.format_bilan_mensuel(mois, annea)
        
        if not bilan:
            print(f"❌ Aucune donnée pour {mois}/{annea}")
            return
        
        # Préparer données
        produits = list(bilan.keys())
        productions = [bilan[p].production for p in produits]
        consommations = [bilan[p].consommation for p in produits]
        variations_stock = [bilan[p].delta_stock for p in produits]
        
        # Créer graphique
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=produits, y=productions, name='Production',
            marker=dict(color='#4CAF50', opacity=0.85, line=dict(color='darkgreen', width=1)),
            text=[f'{v:.1f}' for v in productions], textposition='auto',
            hovertemplate='<b>%{x}</b><br>Production: %{y:.2f}<extra></extra>',
        ))
        
        fig.add_trace(go.Bar(
            x=produits, y=consommations, name='Consommation',
            marker=dict(color='#FF9800', opacity=0.85, line=dict(color='darkorange', width=1)),
            text=[f'{v:.1f}' for v in consommations], textposition='auto',
            hovertemplate='<b>%{x}</b><br>Consommation: %{y:.2f}<extra></extra>',
        ))
        
        fig.add_trace(go.Bar(
            x=produits, y=variations_stock, name='Variation Stock',
            marker=dict(color='#2196F3', opacity=0.85, line=dict(color='darkblue', width=1)),
            text=[f'{v:.1f}' for v in variations_stock], textposition='auto',
            hovertemplate='<b>%{x}</b><br>Variation Stock: %{y:.2f}<extra></extra>',
        ))
        
        nom_mois = calendar.month_name[mois]
        
        fig.update_layout(
            title=f'<b>📊 Histogramme Production - {nom_mois} {annea}</b>',
            xaxis_title='<b>Produits</b>', yaxis_title='<b>Quantité (unités)</b>',
            barmode='group', hovermode='x unified', template='plotly_white',
            height=650, margin=dict(l=100, r=80, t=120, b=100),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="rgba(0, 0, 0, 0.3)", borderwidth=2),
            plot_bgcolor='rgba(240, 240, 240, 0.5)'
        )
        
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1.5)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
        
        fig.show()
        
        # Tableau
        print("\n" + "="*95)
        print(f"  📋 RÉSUMÉ - HISTOGRAMME {nom_mois.upper()} {annea}")
        print("="*95)
        print(f"{'Produit':<20} {'Production':<18} {'Consommation':<18} {'Var. Stock':<18} {'Efficacité':<15}")
        print("-"*95)
        
        for i, prod in enumerate(produits):
            prod_val = productions[i]
            conso_val = consommations[i]
            stock_val = variations_stock[i]
            eff = (prod_val / conso_val * 100) if conso_val > 0 else 0
            print(f"{prod:<20} {prod_val:<18.2f} {conso_val:<18.2f} {stock_val:<18.2f} {eff:<15.1f}%")
        
        print("-"*95)
        print(f"{'TOTAL':<20} {sum(productions):<18.2f} {sum(consommations):<18.2f} {sum(variations_stock):<18.2f}")
        print("="*95 + "\n")
    
    # ═════════════════════════════════════════════════════════════════════════════
    # MÉTHODE 2: WATERFALL - MOIS EN COURS
    # ═════════════════════════════════════════════════════════════════════════════
    
    def plot_waterfall_mois(mois=None, annea=None):
        """
        Waterfall montrant l'évolution du mois jour par jour
        
        Parameters:
        -----------
        mois : int, optionnel (1-12)
        annea : int, optionnel
        """
        
        df = dashboard.processor.data
        if df is None or df.empty:
            print("❌ Aucune donnée disponible")
            return
        
        # Résoudre paramètres
        if mois is None:
            mois = df.index.month[-1]
        if annea is None:
            annea = df.index.year[-1]
        
        num_jours_mois = calendar.monthrange(annea, mois)[1]
        
        # Collecter données
        jours_valides = []
        productions_jour = []
        consommations_jour = []
        
        for jour in range(1, num_jours_mois + 1):
            try:
                bilan = dashboard.formatter.format_bilan_journalier(jour, mois, annea)
                
                if bilan:
                    prod_jour = sum(b.production for b in bilan.values())
                    conso_jour = sum(b.consommation for b in bilan.values())
                    
                    if prod_jour > 0 or conso_jour > 0:
                        jours_valides.append(jour)
                        productions_jour.append(prod_jour)
                        consommations_jour.append(conso_jour)
            except:
                pass
        
        if not jours_valides:
            print(f"❌ Aucune donnée pour {mois}/{annea}")
            return
        
        # Calculer deltas
        delta_stock_jour = [prod - conso for prod, conso in zip(productions_jour, consommations_jour)]
        
        labels = [f"J{j}" for j in jours_valides]
        
        # Créer subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('<b>Production Cumulée par Jour</b>', '<b>Stock Net Cumulé par Jour</b>'),
            vertical_spacing=0.15,
            specs=[[{"type": "waterfall"}], [{"type": "waterfall"}]],
            row_heights=[0.5, 0.5]
        )
        
        # WATERFALL 1: Production
        measure_production = ["relative"] * len(jours_valides) + ["total"]
        labels_production = labels + ["TOTAL"]
        values_production = productions_jour + [sum(productions_jour)]
        text_production = [f'{v:.0f}' for v in productions_jour] + [f'{sum(productions_jour):.0f}']
        
        fig.add_trace(
            go.Waterfall(
                x=labels_production, y=values_production, measure=measure_production,
                text=text_production, textposition='outside',
                connector=dict(line=dict(color='#4CAF50', width=2.5)),
                decreasing=dict(marker=dict(color='#FF5722', line=dict(color='darkred', width=1))),
                increasing=dict(marker=dict(color='#4CAF50', line=dict(color='darkgreen', width=1))),
                totals=dict(marker=dict(color='#2196F3', line=dict(color='darkblue', width=2))),
                hovertemplate='<b>%{x}</b><br>Production: %{y:.2f}<extra></extra>',
            ),
            row=1, col=1
        )
        
        # WATERFALL 2: Stock
        measure_stock = ["relative"] * len(jours_valides) + ["total"]
        labels_stock = labels + ["TOTAL"]
        values_stock = delta_stock_jour + [sum(delta_stock_jour)]
        text_stock = [f'{v:.0f}' for v in delta_stock_jour] + [f'{sum(delta_stock_jour):.0f}']
        
        fig.add_trace(
            go.Waterfall(
                x=labels_stock, y=values_stock, measure=measure_stock,
                text=text_stock, textposition='outside',
                connector=dict(line=dict(color='#9C27B0', width=2.5)),
                decreasing=dict(marker=dict(color='#FF5722', line=dict(color='darkred', width=1))),
                increasing=dict(marker=dict(color='#4CAF50', line=dict(color='darkgreen', width=1))),
                totals=dict(marker=dict(color='#9C27B0', line=dict(color='purple', width=2))),
                hovertemplate='<b>%{x}</b><br>Stock Net: %{y:.2f}<extra></extra>',
            ),
            row=2, col=1
        )
        
        nom_mois = calendar.month_name[mois]
        
        fig.update_layout(
            title=f'<b>💧 Waterfall - {nom_mois} {annea} ({len(jours_valides)} jours)</b>',
            height=900, showlegend=False, hovermode='x unified', template='plotly_white',
            margin=dict(l=100, r=80, t=120, b=100), plot_bgcolor='rgba(240, 240, 240, 0.3)'
        )
        
        fig.update_yaxes(title_text="<b>Quantité (unités)</b>", row=1, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
        fig.update_yaxes(title_text="<b>Quantité (unités)</b>", row=2, col=1, showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
        fig.update_xaxes(title_text="<b>Jours</b>", row=2, col=1)
        
        fig.show()
        
        # Tableau
        print("\n" + "="*120)
        print(f"  📊 RÉCAPITULATIF WATERFALL - {nom_mois.upper()} {annea}")
        print("="*120)
        print(f"{'Jour':<8} {'Production':<15} {'Consommation':<15} {'Stock Net':<15} {'Prod. Cumul':<15} {'Stock Cumul':<15} {'Efficacité':<12}")
        print("-"*120)
        
        cumul_prod = 0
        cumul_stock = 0
        
        for jour, prod, conso, delta in zip(jours_valides, productions_jour, consommations_jour, delta_stock_jour):
            cumul_prod += prod
            cumul_stock += delta
            eff = (prod / conso * 100) if conso > 0 else 0
            print(f"{jour:<8} {prod:<15.2f} {conso:<15.2f} {delta:<15.2f} {cumul_prod:<15.2f} {cumul_stock:<15.2f} {eff:<12.1f}%")
        
        print("-"*120)
        print(f"{'TOTAL':<8} {sum(productions_jour):<15.2f} {sum(consommations_jour):<15.2f} {sum(delta_stock_jour):<15.2f} {cumul_prod:<15.2f} {cumul_stock:<15.2f}")
        print("="*120 + "\n")
    
    # ═════════════════════════════════════════════════════════════════════════════
    # MÉTHODE 3: HISTOGRAMME - PRODUCTION PAR JOUR
    # ═════════════════════════════════════════════════════════════════════════════
    
    def plot_histogramme_jours_mois(mois=None, annea=None, nom_produit=None):
        """
        Histogramme production par jour (avec gradient de couleur)
        
        Parameters:
        -----------
        mois : int, optionnel (1-12)
        annea : int, optionnel
        nom_produit : str, optionnel (si None, affiche total)
        """
        
        df = dashboard.processor.data
        if df is None or df.empty:
            print("❌ Aucune donnée disponible")
            return
        
        # Résoudre paramètres
        if mois is None:
            mois = df.index.month[-1]
        if annea is None:
            annea = df.index.year[-1]
        
        num_jours_mois = calendar.monthrange(annea, mois)[1]
        
        # Collecter données
        jours_valides = []
        productions = []
        
        for jour in range(1, num_jours_mois + 1):
            try:
                bilan = dashboard.formatter.format_bilan_journalier(jour, mois, annea)
                
                if nom_produit:
                    if nom_produit in bilan:
                        prod = bilan[nom_produit].production
                    else:
                        continue
                else:
                    prod = sum(b.production for b in bilan.values())
                
                if prod > 0:
                    jours_valides.append(jour)
                    productions.append(prod)
            except:
                pass
        
        if not jours_valides:
            print(f"❌ Aucune donnée pour {mois}/{annea}")
            return
        
        # Créer graphique
        fig = go.Figure()
        
        # Couleurs
        max_prod = max(productions)
        min_prod = min(productions)
        moyenne_prod = np.mean(productions)
        
        colors = []
        for p in productions:
            if p >= moyenne_prod:
                intensity = (p - moyenne_prod) / (max_prod - moyenne_prod) if max_prod > moyenne_prod else 0.5
                r = int(255 * (1 - intensity * 0.7))
                g = 204
                b = int(0 + intensity * 50)
                colors.append(f'rgba({r}, {g}, {b}, 0.85)')
            else:
                intensity = (moyenne_prod - p) / (moyenne_prod - min_prod) if moyenne_prod > min_prod else 0.5
                r = 255
                g = int(192 - intensity * 100)
                b = 0
                colors.append(f'rgba({r}, {g}, {b}, 0.85)')
        
        fig.add_trace(go.Bar(
            x=[f'J{j}' for j in jours_valides], y=productions,
            marker=dict(color=colors, line=dict(color='darkgray', width=1.5)),
            text=[f'{p:.1f}' for p in productions], textposition='auto',
            hovertemplate='<b>Jour %{x}</b><br>Production: %{y:.2f}<extra></extra>',
            showlegend=False
        ))
        
        # Ligne moyenne
        fig.add_hline(y=moyenne_prod, line_dash="dash", line_color="red", line_width=2.5,
                     annotation_text=f"  Moyenne: {moyenne_prod:.2f}", annotation_position="right")
        
        # Zones
        fig.add_hrect(y0=moyenne_prod, y1=max_prod*1.1, fillcolor="green", opacity=0.05, layer="below")
        fig.add_hrect(y0=min_prod*0.9, y1=moyenne_prod, fillcolor="orange", opacity=0.05, layer="below")
        
        nom_mois = calendar.month_name[mois]
        titre = f"Production par Jour - {nom_mois} {annea}"
        if nom_produit:
            titre = f"Production {nom_produit} par Jour - {nom_mois} {annea}"
        
        fig.update_layout(
            title=f'<b>📈 {titre}</b>',
            xaxis_title='<b>Jours</b>', yaxis_title='<b>Quantité (unités)</b>',
            hovermode='x unified', template='plotly_white', height=600,
            margin=dict(l=100, r=100, t=120, b=100), plot_bgcolor='rgba(240, 240, 240, 0.3)',
            showlegend=False
        )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
        
        fig.show()
        
        # Stats
        print("\n" + "="*90)
        print(f"  📊 STATISTIQUES - {titre.upper()}")
        print("="*90)
        print(f"Nombre de jours complets:           {len(jours_valides)}")
        print(f"Production moyenne:                 {np.mean(productions):.2f}")
        print(f"Production min/max:                 {np.min(productions):.2f} / {np.max(productions):.2f}")
        print(f"Écart-type:                         {np.std(productions):.2f}")
        print(f"Coefficient de variation:           {(np.std(productions)/np.mean(productions)*100):.1f}%")
        print(f"Production totale mois:             {sum(productions):.2f}")
        print(f"TRS mois: (jours complets)          {sum(productions) / len(jours_valides) :.2f}")
        print(f"Jours au-dessus de la moyenne:      {len([p for p in productions if p >= moyenne_prod])} / {len(productions)}")
        print("="*90 + "\n")

    def plot_histogramme_jours_mois_v1(mois=None, annea=None, nom_produit=None):
        
        df = dashboard.processor.data
        if df is None or df.empty:
            print("❌ Aucune donnée disponible")
            return
        
        # Résoudre paramètres
        if mois is None:
            mois = df.index.month[-1]
        if annea is None:
            annea = df.index.year[-1]
        
        num_jours_mois = calendar.monthrange(annea, mois)[1]
        
        # Collecter données
        jours_valides = []
        productions = []
        
        for jour in range(1, num_jours_mois + 1):
            try:
                bilan = dashboard.formatter.format_bilan_journalier(jour, mois, annea)
                
                if nom_produit:
                    if nom_produit in bilan:
                        prod = bilan[nom_produit].production
                    else:
                        continue
                else:
                    prod = sum(b.production for b in bilan.values())
                
                if prod > 0:
                    jours_valides.append(jour)
                    productions.append(prod)
            except:
                pass
        
        if not jours_valides:
            print(f"❌ Aucune donnée pour {mois}/{annea}")
            return
        
        # Créer graphique
        fig = go.Figure()
        
        # Couleurs
        max_prod = max(productions)
        min_prod = min(productions)
        moyenne_prod = np.mean(productions)
        
        colors = []
        for p in productions:
            if p >= moyenne_prod:
                intensity = (p - moyenne_prod) / (max_prod - moyenne_prod) if max_prod > moyenne_prod else 0.5
                r = int(255 * (1 - intensity * 0.7))
                g = 204
                b = int(0 + intensity * 50)
                colors.append(f'rgba({r}, {g}, {b}, 0.85)')
            else:
                intensity = (moyenne_prod - p) / (moyenne_prod - min_prod) if moyenne_prod > min_prod else 0.5
                r = 255
                g = int(192 - intensity * 100)
                b = 0
                colors.append(f'rgba({r}, {g}, {b}, 0.85)')
        
        fig.add_trace(go.Bar(
            x=[f'J{j}' for j in jours_valides], y=productions,
            marker=dict(color=colors, line=dict(color='darkgray', width=1.5)),
            text=[f'{p:.1f}' for p in productions], textposition='auto',
            hovertemplate='<b>Jour %{x}</b><br>Production: %{y:.2f}<extra></extra>',
            showlegend=False
        ))
        
        # Ligne moyenne
        fig.add_hline(y=moyenne_prod, line_dash="dash", line_color="red", line_width=2.5,
                    annotation_text=f"  Moyenne: {moyenne_prod:.2f}", annotation_position="right")
        
        # Zones
        fig.add_hrect(y0=moyenne_prod, y1=max_prod*1.1, fillcolor="green", opacity=0.05, layer="below")
        fig.add_hrect(y0=min_prod*0.9, y1=moyenne_prod, fillcolor="orange", opacity=0.05, layer="below")
        
        nom_mois = calendar.month_name[mois] if 1 <= mois <= 12 else 'Mois'
        titre = f"Production par Jour - {nom_mois} {annea}"
        if nom_produit:
            titre = f"Production {nom_produit} par Jour - {nom_mois} {annea}"
        
        fig.update_layout(
            title=f'<b>📈 {titre}</b>',
            xaxis_title='<b>Jours</b>', yaxis_title='<b>Quantité (unités)</b>',
            hovermode='x unified', template='plotly_white', height=600,
            margin=dict(l=100, r=100, t=120, b=100), plot_bgcolor='rgba(240, 240, 240, 0.3)',
            showlegend=False
        )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
        
        fig.show()
        
        # ✨ STATISTIQUES AVEC RATIO oee
        print("\n" + "="*90)
        print(f"  📊 STATISTIQUES - {titre.upper()}")
        print("="*90)
        print(f"Nombre de jours complets:           {len(jours_valides)}")
        print(f"Production moyenne:                 {np.mean(productions):.2f}")
        print(f"Production min/max:                 {np.min(productions):.2f} / {np.max(productions):.2f}")
        print(f"Écart-type:                         {np.std(productions):.2f}")
        print(f"Coefficient de variation:           {(np.std(productions)/np.mean(productions)*100):.1f}%")
        print(f"Production totale mois:             {sum(productions):.2f}")
        print(f"Jours au-dessus de la moyenne:      {len([p for p in productions if p >= moyenne_prod])} / {len(productions)}")
        
        # ✨ RATIO oee (Production / CMJ en %)
        if nom_produit:
            col_cmj = 'CMJ'
            if col_cmj in df.columns:
                print(f"\n🎯 RATIO oee (Production / CMJ en %):")
                
                oee_ratios = []
                oee_par_jour = {}
                
                for jour in jours_valides:
                    try:
                        bilan = dashboard.formatter.format_bilan_journalier(jour, mois, annea, nom_produit)
                        
                        if nom_produit in bilan:
                            # Récupérer CMJ du jour
                            date_jour = pd.Timestamp(year=annea, month=mois, day=jour)
                            mask = (df.index.date == date_jour.date())
                            
                            if mask.any():
                                cmj = df[col_cmj][mask].iloc[0]
                                if cmj > 0:
                                    prod = bilan[nom_produit].production
                                    oee = (prod / cmj) * 100
                                    oee_ratios.append(oee)
                                    oee_par_jour[jour] = oee
                    except:
                        pass
                
                if oee_ratios:
                    oee_moyen = np.mean(oee_ratios)
                    oee_min = np.min(oee_ratios)
                    oee_max = np.max(oee_ratios)
                    oee_jours_sup_100 = len([o for o in oee_ratios if o > 100])
                    
                    print(f"  oee Moyen:                          {oee_moyen:.1f}%")
                    print(f"  oee Min/Max:                        {oee_min:.1f}% / {oee_max:.1f}%")
                    print(f"  Jours > 100% (surproduction):       {oee_jours_sup_100} / {len(oee_ratios)}")
                    
                    # Détail par jour
                    print(f"\n  Détail par jour:")
                    for jour in sorted(oee_par_jour.keys()):
                        oee = oee_par_jour[jour]
                        statut = "✅" if oee > 100 else "⚠️ " if oee > 80 else "❌"
                        print(f"    J{jour:02d}: {oee:6.1f}% {statut}")
            else:
                print(f"\n⚠️  Colonne CMJ non trouvée dans les données")
                print(f"   Colonnes disponibles: {[c for c in df.columns if 'CMJ' in c]}")
        
        print("="*90 + "\n") 

    def plot_histogramme_jours_mois_v2(mois=None, annea=None, nom_produit=None, std=2):
        """
        Histogramme production par jour avec courbe oee cumulée
        
        oee cumulé = sum(productions jour 1 à N) / (N * CMJ) * 100
        
        Parameters:
        -----------
        mois : int, optionnel (1-12)
        annea : int, optionnel
        nom_produit : str, optionnel (si None, affiche total)
        """
        
        df = dashboard.processor.data
        if df is None or df.empty:
            print("❌ Aucune donnée disponible")
            return
        
        # Résoudre paramètres
        if mois is None:
            mois = df.index.month[-1]
        if annea is None:
            annea = df.index.year[-1]
        
        num_jours_mois = calendar.monthrange(annea, mois)[1]
        
        # Récupérer CMJ depuis config produits
        cmj = None
        if nom_produit:
            for prod_config in dashboard.processor.produits:
                if prod_config['nom'] == nom_produit:
                    cmj = prod_config.get('CMJ')
                    break
        
        # Collecter données
        jours_valides = []
        productions = []
        oee_jours = []  # oee pour chaque jour
        oee_cumules = []  # oee cumulé
        somme_prod = 0
        
        for jour in range(1, num_jours_mois + 1):
            try:
                bilan = dashboard.formatter.format_bilan_journalier(jour, mois, annea, std=std)
                
                if nom_produit:
                    if nom_produit in bilan:
                        prod = bilan[nom_produit].production
                    else:
                        continue
                else:
                    prod = sum(b.production for b in bilan.values())
                
                if prod > 0:
                    jours_valides.append(jour)
                    productions.append(prod)
                    somme_prod += prod
                    
                    # Calculer oee cumulé: sum(prod_j1..jN) / (N * CMJ) * 100
                    if nom_produit and cmj and cmj > 0:
                        oee_jour = (prod / cmj) * 100  # oee du jour seul
                        oee_jours.append(oee_jour)
                        
                        oee_cumul = (somme_prod / (len(jours_valides) * cmj)) * 100
                        oee_cumules.append(oee_cumul)
                    else:
                        oee_jours.append(None)
                        oee_cumules.append(None)
            except:
                pass
        
        if not jours_valides:
            print(f"❌ Aucune donnée pour {mois}/{annea}")
            return
        
        # Créer graphique
        fig = go.Figure()
        
        # Couleurs barres
        max_prod = max(productions)
        min_prod = min(productions)
        moyenne_prod = np.mean(productions)
        
        colors = []
        for p in productions:
            if p >= moyenne_prod:
                intensity = (p - moyenne_prod) / (max_prod - moyenne_prod) if max_prod > moyenne_prod else 0.5
                r = int(255 * (1 - intensity * 0.7))
                g = 204
                b = int(0 + intensity * 50)
                colors.append(f'rgba({r}, {g}, {b}, 0.85)')
            else:
                intensity = (moyenne_prod - p) / (moyenne_prod - min_prod) if moyenne_prod > min_prod else 0.5
                r = 255
                g = int(192 - intensity * 100)
                b = 0
                colors.append(f'rgba({r}, {g}, {b}, 0.85)')
        
        # BARRES: Production
        fig.add_trace(go.Bar(
            x=[f'J{j}' for j in jours_valides], y=productions,
            marker=dict(color=colors, line=dict(color='darkgray', width=1.5)),
            text=[f'{p:.1f}' for p in productions], textposition='auto',
            hovertemplate='<b>Jour %{x}</b><br>Production: %{y:.2f}<extra></extra>',
            showlegend=True,
            name='Production',
            yaxis='y'
        ))
        
        # COURBE: oee cumulé
        if nom_produit and cmj and cmj > 0 and any(o is not None for o in oee_cumules):
            fig.add_trace(go.Scatter(
                x=[f'J{j}' for j in jours_valides],
                y=oee_cumules,
                mode='lines+markers',
                name=f'oee Cumulé ({nom_produit})',
                line=dict(color='#0F6B6B', width=3),
                marker=dict(size=8, color='#0F6B6B', symbol='diamond'),
                hovertemplate='<b>Jour %{x}</b><br>oee Cumul: %{y:.1f}%<extra></extra>',
                yaxis='y2'
            ))
        
        # Ligne moyenne
        fig.add_hline(y=moyenne_prod, line_dash="dash", line_color="red", line_width=2.5,
             annotation_text=f" Moy Prod: {moyenne_prod:.2f}", annotation_position="right",
             yref='y')

        # Target oee
#        print(nom_produit, cmj)
#        if nom_produit and cmj :
#        fig.add_hline(y=8.148, line_dash="dash", line_color="#8f97F3", line_width=2.5,
        fig.add_hline(y=84, line_dash="dash", line_color="#8f97F3", line_width=2.5,
            annotation_text=" oee: 84%", annotation_position="right",
            yref='y2')

        # Zones
        fig.add_hrect(y0=moyenne_prod, y1=max_prod*1.1, fillcolor="green", opacity=0.05, layer="below", yref='y')
        fig.add_hrect(y0=min_prod*0.9, y1=moyenne_prod, fillcolor="orange", opacity=0.05, layer="below", yref='y')
        
        nom_mois = calendar.month_name[mois] if 1 <= mois <= 12 else 'Mois'
        titre = f"Production par Jour - {nom_mois} {annea}"
        if nom_produit:
            titre = f"Production {nom_produit} par Jour - {nom_mois} {annea}"

        fig.update_layout(
            title=f'<b>📈 {titre}</b>',
            hovermode='x unified',
            template='plotly_white',
            height=700,
            margin=dict(l=100, r=100, t=120, b=100),
            plot_bgcolor='rgba(240, 240, 240, 0.3)',

            xaxis=dict(title='<b>Jours</b>'),
            yaxis=dict(
                title=dict(text='<b>Production (unités)</b>', font=dict(color='#1f77b4')),
                tickfont=dict(color='#1f77b4'),
                side='left',
                range=[0, 12]
            ),
            yaxis2=dict(
                title=dict(text='<b>oee Cumulé (%)</b>', font=dict(color='#0F6B6B')) if nom_produit and cmj else None,
                tickfont=dict(color='#0F6B6B'),
                overlaying='y',
                side='right',
                range=[0, 123.7]
            ) if nom_produit and cmj else None,

            legend=dict(x=0.01, y=0.99)
        )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')        
        fig.show()
        
        # STATISTIQUES
        print("\n" + "="*90)
        print(f"  📊 STATISTIQUES - {titre.upper()}")
        print("="*90)
        print(f"Nombre de jours complets:           {len(jours_valides)}")
        print(f"Production moyenne:                 {np.mean(productions):.2f}")
        print(f"Production min/max:                 {np.min(productions):.2f} / {np.max(productions):.2f}")
        print(f"Écart-type:                         {np.std(productions):.2f}")
        print(f"Coefficient de variation:           {(np.std(productions)/np.mean(productions)*100):.1f}%")
        print(f"Production totale mois:             {sum(productions):.2f}")
        print(f"TRS mois: (jours complets)          {sum(productions) / len(jours_valides):.2f}")
        print(f"Jours au-dessus de la moyenne:      {len([p for p in productions if p >= moyenne_prod])} / {len(productions)}")
        
        # oee
        if nom_produit:
            if cmj and cmj > 0:
                print(f"\n🎯 RATIO oee (Production / CMJ en %):")
                print(f"  CMJ (Cible Journalière):            {cmj:.2f}")
                
                if oee_jours and any(o is not None for o in oee_jours):
                    oee_valides = [o for o in oee_jours if o is not None]
                    oee_moyen = np.mean(oee_valides)
                    oee_min = np.min(oee_valides)
                    oee_max = np.max(oee_valides)
                    oee_cumul_final = oee_cumules[-1] if oee_cumules and oee_cumules[-1] is not None else 0
                    oee_jours_sup_100 = len([o for o in oee_valides if o > 100])
                    
                    print(f"  oee Moyen (par jour):               {oee_moyen:.1f}%")
                    print(f"  oee Min/Max (par jour):             {oee_min:.1f}% / {oee_max:.1f}%")
                    print(f"  oee Cumulé à date:                  {oee_cumul_final:.1f}%")
                    print(f"  Jours > 100%:                       {oee_jours_sup_100} / {len(oee_valides)}")
                    
                    print(f"\n  Détail oee par jour:")
                    for jour, oee in zip(jours_valides, oee_jours):
                        if oee is not None:
                            statut = "✅" if oee > 100 else "⚠️ " if oee > 80 else "❌"
                            print(f"    J{jour:02d}: {oee:6.1f}% {statut}")
            else:
                print(f"\n⚠️  CMJ non trouvée pour {nom_produit}")
        
        print("="*90 + "\n")

    def plot_histogramme_annee(annea=None, nom_produit=None):
        """
        Affiche histogrammes + oee cumulé pour tous les mois de l'année
        12 subplots (3 lignes x 4 colonnes)
        
        Parameters:
        -----------
        annea : int, optionnel
        nom_produit : str, optionnel (si None, affiche total)
        """
        
        df = dashboard.processor.data
        if df is None or df.empty:
            print("❌ Aucune donnée disponible")
            return
        
        if annea is None:
            annea = df.index.year[-1]
        
        # Récupérer CMJ
        cmj = None
        if nom_produit:
            for prod_config in dashboard.processor.produits:
                if prod_config['nom'] == nom_produit:
                    cmj = prod_config.get('CMJ')
                    break
        
        # Créer subplots (3 lignes x 4 colonnes = 12 mois)
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=3, cols=4,
            subplot_titles=[calendar.month_name[i] for i in range(1, 13)],
            specs=[[{"secondary_y": True}]*4]*3,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Résumé annuel
        resume_mois = {}
        
        # Boucle sur les 12 mois
        for mois in range(1, 13):
            row = ((mois - 1) // 4) + 1
            col = ((mois - 1) % 4) + 1
            
            # Collecter données du mois
            num_jours_mois = calendar.monthrange(annea, mois)[1]
            jours_valides = []
            productions = []
            oee_cumules = []
            somme_prod = 0
            oee_jours = []
            
            for jour in range(1, num_jours_mois + 1):
                try:
                    bilan = dashboard.formatter.format_bilan_journalier(jour, mois, annea)
                    
                    if nom_produit:
                        if nom_produit in bilan:
                            prod = bilan[nom_produit].production
                        else:
                            continue
                    else:
                        prod = sum(b.production for b in bilan.values())
                    
                    if prod > 0:
                        jours_valides.append(jour)
                        productions.append(prod)
                        somme_prod += prod
                        
                        if nom_produit and cmj and cmj > 0:
                            oee_jour = (prod / cmj) * 100
                            oee_jours.append(oee_jour)
                            oee_cumul = (somme_prod / (len(jours_valides) * cmj)) * 100
                            oee_cumules.append(oee_cumul)
                        else:
                            oee_jours.append(None)
                            oee_cumules.append(None)
                except:
                    pass
            
            if not jours_valides:
                continue
            
            # Couleurs
            max_prod = max(productions)
            min_prod = min(productions)
            moyenne_prod = np.mean(productions)
            
            colors = []
            for p in productions:
                if p >= moyenne_prod:
                    intensity = (p - moyenne_prod) / (max_prod - moyenne_prod) if max_prod > moyenne_prod else 0.5
                    r = int(255 * (1 - intensity * 0.7))
                    g = 204
                    b = int(0 + intensity * 50)
                    colors.append(f'rgba({r}, {g}, {b}, 0.85)')
                else:
                    intensity = (moyenne_prod - p) / (moyenne_prod - min_prod) if moyenne_prod > min_prod else 0.5
                    r = 255
                    g = int(192 - intensity * 100)
                    b = 0
                    colors.append(f'rgba({r}, {g}, {b}, 0.85)')
            
            # Ajouter barres
            fig.add_trace(
                go.Bar(
                    x=[f'J{j}' for j in jours_valides],
                    y=productions,
                    marker=dict(color=colors, line=dict(color='darkgray', width=0.5)),
                    text=[f'{p:.0f}' for p in productions],
                    textposition='none',
                    hovertemplate='<b>%{x}</b><br>Prod: %{y:.1f}<extra></extra>',
                    showlegend=False,
                    name='Prod'
                ),
                row=row, col=col, secondary_y=False
            )
            
            # Ajouter courbe oee
            if nom_produit and cmj and cmj > 0 and any(o is not None for o in oee_cumules):
                oee_valides = [o for o in oee_cumules if o is not None]
                fig.add_trace(
                    go.Scatter(
                        x=[f'J{j}' for j in jours_valides],
                        y=oee_cumules,
                        mode='lines+markers',
                        line=dict(color='#FF6B6B', width=2),
                        marker=dict(size=3, color='#FF6B6B'),
                        hovertemplate='<b>%{x}</b><br>oee: %{y:.0f}%<extra></extra>',
                        showlegend=False,
                        name='oee'
                    ),
                    row=row, col=col, secondary_y=True
                )
            
            # Sauvegarder résumé
            oee_valides_jour = [o for o in oee_jours if o is not None]
            resume_mois[mois] = {
                'jours': len(jours_valides),
                'prod_total': sum(productions),
                'prod_moy': np.mean(productions),
                'oee_moy': np.mean(oee_valides_jour) if oee_valides_jour else 0,
                'oee_cumul': oee_cumules[-1] if oee_cumules and oee_cumules[-1] is not None else 0
            }
        
        # Mise en forme
        fig.update_layout(
            title=f'<b>📊 Production & oee Cumulé - {annea}</b>',
            height=900,
            showlegend=False,
            hovermode='closest',
            template='plotly_white'
        )
        
        # Axes Y
        fig.update_yaxes(title_text="Prod", row=1, col=1)
        fig.update_yaxes(title_text="oee %", secondary_y=True, row=1, col=1)
        
        fig.show()
        
        # RÉSUMÉ ANNUEL SEULEMENT
        print("\n" + "="*100)
        print(f"  📊 RÉSUMÉ ANNUEL {annea}" + (f" - {nom_produit}" if nom_produit else ""))
        print("="*100)
        print(f"{'Mois':<12} {'Jours':<8} {'Prod Total':<15} {'Prod Moy':<15} {'oee Moy':<12} {'oee Cumul':<12}")
        print("-"*100)
        
        for mois in range(1, 13):
            if mois in resume_mois:
                data = resume_mois[mois]
                nom_mois = calendar.month_name[mois]
                print(f"{nom_mois:<12} {data['jours']:<8} {data['prod_total']:<15.2f} {data['prod_moy']:<15.2f} {data['oee_moy']:<12.1f}% {data['oee_cumul']:<12.1f}%")
        
        print("="*100 + "\n")

    def plot_histogramme_annee_complet(annea=None, nom_produit=None, std=2):
        """
        Affiche l'année COMPLÈTE sur un seul graphique (365 jours)
        avec histogrammes production + courbe oee cumulé
        
        Parameters:
        -----------
        annea : int, optionnel
        nom_produit : str, optionnel (si None, affiche total)
        std : int, optionnel (défault: 2)
        """
        
        df = dashboard.processor.data
        if df is None or df.empty:
            print("❌ Aucune donnée disponible")
            return
        
        if annea is None:
            annea = df.index.year[-1]
        
        # Récupérer CMJ
        cmj = None
        if nom_produit:
            for prod_config in dashboard.processor.produits:
                if prod_config['nom'] == nom_produit:
                    cmj = prod_config.get('CMJ')
                    break
        
        # Collecter données pour l'année entière
        jours_dates = []
        productions = []
        oee_cumules = []
        somme_prod = 0
        oee_jours = []
        jour_annee = 0
        
        # Déterminer si c'est une année bissextile
        est_bissextile = (annea % 4 == 0 and annea % 100 != 0) or (annea % 400 == 0)
        num_jours_annee = 366 if est_bissextile else 365
        
        # Boucler sur tous les jours de l'année
        for mois in range(1, 13):
            num_jours_mois = calendar.monthrange(annea, mois)[1]
            
            for jour in range(1, num_jours_mois + 1):
                jour_annee += 1
                
                try:
                    bilan = dashboard.formatter.format_bilan_journalier(jour, mois, annea, std=std)
                    
                    if nom_produit:
                        if nom_produit in bilan:
                            prod = bilan[nom_produit].production
                        else:
                            prod = 0
                    else:
                        prod = sum(b.production for b in bilan.values()) if bilan else 0
                    
                    if prod > 0:
                        jours_dates.append(f"J{jour_annee}")
                        productions.append(prod)
                        somme_prod += prod
                        
                        if nom_produit and cmj and cmj > 0:
                            oee_jour = (prod / cmj) * 100
                            oee_jours.append(oee_jour)
                            oee_cumul = (somme_prod / (jour_annee * cmj)) * 100
                            oee_cumules.append(oee_cumul)
                        else:
                            oee_jours.append(None)
                            oee_cumules.append(None)
                    else:
                        jours_dates.append(f"J{jour_annee}")
                        productions.append(0)
                        oee_jours.append(None)
                        
                        if nom_produit and cmj and cmj > 0 and somme_prod > 0:
                            oee_cumul = (somme_prod / (jour_annee * cmj)) * 100
                            oee_cumules.append(oee_cumul)
                        else:
                            oee_cumules.append(None)
                except:
                    jours_dates.append(f"J{jour_annee}")
                    productions.append(0)
                    oee_jours.append(None)
                    oee_cumules.append(None)
        
        if not productions or all(p == 0 for p in productions):
            print(f"❌ Aucune donnée pour l'année {annea}")
            return
        
        # Créer graphique
        fig = go.Figure()
        
        # Couleurs barres
        max_prod = max([p for p in productions if p > 0]) if any(p > 0 for p in productions) else 1
        min_prod = min([p for p in productions if p > 0]) if any(p > 0 for p in productions) else 0
        moyenne_prod = np.mean([p for p in productions if p > 0]) if any(p > 0 for p in productions) else 0
        
        colors = []
        for p in productions:
            if p == 0:
                colors.append('rgba(200, 200, 200, 0.3)')  # Gris pour jours sans données
            elif p >= moyenne_prod:
                intensity = (p - moyenne_prod) / (max_prod - moyenne_prod) if max_prod > moyenne_prod else 0.5
                r = int(255 * (1 - intensity * 0.7))
                g = 204
                b = int(0 + intensity * 50)
                colors.append(f'rgba({r}, {g}, {b}, 0.85)')
            else:
                intensity = (moyenne_prod - p) / (moyenne_prod - min_prod) if moyenne_prod > min_prod else 0.5
                r = 255
                g = int(192 - intensity * 100)
                b = 0
                colors.append(f'rgba({r}, {g}, {b}, 0.85)')
        
        # BARRES: Production
        fig.add_trace(go.Bar(
            x=jours_dates,
            y=productions,
            marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.2)', width=0.5)),
            text=None,
            hovertemplate='<b>%{x}</b><br>Prod: %{y:.1f}<extra></extra>',
            showlegend=True,
            name='Production',
            yaxis='y'
        ))
        
        # COURBE: oee cumulé
        if nom_produit and cmj and cmj > 0 and any(o is not None for o in oee_cumules):
            fig.add_trace(go.Scatter(
                x=jours_dates,
                y=oee_cumules,
                mode='lines',
                name=f'oee Cumulé ({nom_produit})',
                line=dict(color='#FF6B6B', width=2.5),
                hovertemplate='<b>%{x}</b><br>oee Cumul: %{y:.1f}%<extra></extra>',
                yaxis='y2',
                fill='tozeroy',
                fillcolor='rgba(255, 107, 107, 0.1)'
            ))
        
        # Ligne moyenne production
        fig.add_hline(y=moyenne_prod, line_dash="dash", line_color="green", line_width=2,
                    annotation_text=f"  Moy: {moyenne_prod:.1f}", annotation_position="right",
                    yref='y')
        
        # Ligne target oee
        if nom_produit and cmj and cmj > 0:
            fig.add_hline(y=84, line_dash="dash", line_color="#8f97F3", line_width=2,
                        annotation_text=f"  OEE Target: 84%", annotation_position="right",
                        yref='y2')
        
        # Ajouter séparateurs mensuels
        jour_cumul = 0
        for mois in range(1, 13):
            num_jours_mois = calendar.monthrange(annea, mois)[1]
            jour_cumul += num_jours_mois
            
            if jour_cumul < len(jours_dates):
                fig.add_vline(x=jour_cumul-0.5, line_dash="solid", line_color="rgba(0,0,0,0.1)", 
                            line_width=1)
        
        nom_mois_court = ["", "Jan", "Fév", "Mar", "Avr", "Mai", "Juin", 
                        "Juil", "Aoû", "Sep", "Oct", "Nov", "Déc"]
        
        titre = f"Production Annuelle {annea}"
        if nom_produit:
            titre = f"Production {nom_produit} - Année {annea}"
        
        fig.update_layout(
            title=f'<b>📈 {titre}</b>',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            margin=dict(l=100, r=100, t=120, b=100),
            plot_bgcolor='rgba(240, 240, 240, 0.3)',
            xaxis=dict(
                title='<b>Jours de l\'année</b>',
                showticklabels=False
            ),
            yaxis=dict(
                title=dict(text='<b>Production (unités)</b>', font=dict(color='#1f77b4')),
                tickfont=dict(color='#1f77b4'),
                side='left',
                range=[0, max_prod*1.1]
            ),
            yaxis2=dict(
                title=dict(text='<b>oee Cumulé (%)</b>', font=dict(color='#FF6B6B')) if nom_produit and cmj else None,
                tickfont=dict(color='#FF6B6B'),
                overlaying='y',
                side='right',
                range=[0, 110] if nom_produit and cmj else None
            ) if nom_produit and cmj else None,
            legend=dict(x=0.01, y=0.99)
        )
        
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.3)')
        
        fig.show()
        
        # RÉSUMÉ ANNUEL
        print("\n" + "="*90)
        print(f"  📊 RÉSUMÉ ANNUEL {annea}" + (f" - {nom_produit}" if nom_produit else ""))
        print("="*90)
        print(f"Nombre de jours avec données:       {len([p for p in productions if p > 0])}/{num_jours_annee}")
        print(f"Production totale année:            {sum(productions):.2f}")
        print(f"Production moyenne (jours actifs):  {np.mean([p for p in productions if p > 0]):.2f}")
        print(f"Production min/max:                 {min([p for p in productions if p > 0]):.2f} / {max([p for p in productions if p > 0]):.2f}")
        print(f"Écart-type:                         {np.std([p for p in productions if p > 0]):.2f}")
        
        if nom_produit and cmj and cmj > 0:
            oee_valides = [o for o in oee_jours if o is not None]
            oee_cumul_final = oee_cumules[-1] if oee_cumules and oee_cumules[-1] is not None else 0
            oee_jours_sup_100 = len([o for o in oee_valides if o > 100])
            
            print(f"\n🎯 RATIO oee:")
            print(f"  CMJ (Cible Journalière):            {cmj:.2f}")
            print(f"  oee Moyen (par jour):               {np.mean(oee_valides):.1f}%")
            print(f"  oee Cumulé à date (fin d'année):    {oee_cumul_final:.1f}%")
            print(f"  Jours > 100% (surproduction):       {oee_jours_sup_100} / {len(oee_valides)}")
        
        print("="*90 + "\n")

    # ═════════════════════════════════════════════════════════════════════════════
    # AJOUTER LES MÉTHODES AU DASHBOARD
    # ═════════════════════════════════════════════════════════════════════════════
    
    dashboard.plot_histogramme_tous_produits = plot_histogramme_tous_produits
    dashboard.plot_waterfall_mois = plot_waterfall_mois
    dashboard.plot_histogramme_jours_mois = plot_histogramme_jours_mois
    dashboard.plot_histogramme_jours_mois_v1 = plot_histogramme_jours_mois_v1
    dashboard.plot_histogramme_jours_mois_v2 = plot_histogramme_jours_mois_v2
    dashboard.plot_histogramme_annuee = plot_histogramme_annee
    dashboard.plot_histogramme_annee_complet = plot_histogramme_annee_complet

    print("✅ Visualisations avancées ajoutées au dashboard!")
    print("\n   Nouvelles méthodes disponibles:")
    print("   • dashboard.plot_histogramme_tous_produits(mois=3)")
    print("   • dashboard.plot_waterfall_mois(mois=3)")
    print("   • dashboard.plot_histogramme_jours_mois_v1(mois=3, nom_produit='Ester')")
    print("   • dashboard.plot_histogramme_jours_mois_v2(mois=3, nom_produit='Ester')")
    print()

"""
═══════════════════════════════════════════════════════════════════════════════
UTILISATION DANS JUPYTER
═════════════════════════════════════════════════════════════════════════════════

# Import
from OI_class_OP import OI_ProductionProcessor
from OI_Dashboard import ProductionDashboard
from EXTENSION_Dashboard_Avance import AjouterVisualisationsAvancees

# Initialiser
processor = OI_ProductionProcessor(...)
processor.merge()
processor.compute_production_balance()

dashboard = ProductionDashboard(processor)

# ← Ajouter les visualisations avancées
AjouterVisualisationsAvancees(dashboard)

# Utiliser les 3 nouvelles méthodes
dashboard.plot_histogramme_tous_produits(mois=3, annea=2024)
dashboard.plot_waterfall_mois(mois=3, annea=2024)
dashboard.plot_histogramme_jours_mois(mois=3, annea=2024)
dashboard.plot_histogramme_jours_mois(mois=3, annea=2024, nom_produit='Acetate')

# Plus toutes les méthodes existantes du dashboard
dashboard.afficher_kpis_journaliers(jour=15, mois=3)
dashboard.afficher_bilan_journalier(jour=15, mois=3)
...
"""