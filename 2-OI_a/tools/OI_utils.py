# Ensemble d'utilitaires pour l'analyse des données OI
import pandas as pd

def extract_xls(data,nom_fichier,index=False):
    df = data.copy()
    nom_fichier = nom_fichier+".xlsx"
    if (index):
        # Supprimer la timezone de l'index
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Corriger TOUTES les colonnes datetime avec timezone
        for col in df.columns:
            # Vérifier si c'est une colonne datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                # Si elle a une timezone, la supprimer
                if df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_localize(None)

    df.to_excel(nom_fichier, index=index)
    print(f"Fichier '{nom_fichier}' créé avec succès.")


    import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf

def analyser_correlation_croisee(df, col1, col2, max_lag=50, plot=True, seuil_significativite=0.05):
    """
    Analyse la corrélation croisée entre deux colonnes d'un DataFrame avec décalage temporel.
    
    Paramètres:
    -----------
    df : pandas.DataFrame
        Le DataFrame contenant les données
    col1 : str
        Nom de la première colonne (série de référence)
    col2 : str
        Nom de la deuxième colonne (série à décaler)
    max_lag : int, default=50
        Nombre maximum de décalages à tester
    plot : bool, default=True
        Si True, affiche les graphiques
    seuil_significativite : float, default=0.05
        Seuil pour calculer l'intervalle de confiance
    
    Retourne:
    ---------
    dict : Dictionnaire contenant:
        - 'meilleur_lag': Le décalage avec la corrélation maximale
        - 'max_correlation': La valeur de la corrélation maximale
        - 'correlations': Array des corrélations pour chaque lag
        - 'lags': Array des lags testés
        - 'intervalle_confiance': Limite de l'intervalle de confiance
    """
    
    # Extraire les séries et supprimer les NaN
    serie1 = df[col1].dropna()
    serie2 = df[col2].dropna()
    
    # Vérifier qu'on a assez de données
    if len(serie1) < max_lag or len(serie2) < max_lag:
        raise ValueError(f"Les séries doivent avoir au moins {max_lag} observations")
    
    # Aligner les séries sur les mêmes index
    data_aligned = pd.DataFrame({col1: serie1, col2: serie2}).dropna()
    serie1 = data_aligned[col1]
    serie2 = data_aligned[col2]
    
    # Calculer la corrélation croisée
    cross_corr = ccf(serie1, serie2, adjusted=False)[:max_lag+1]
    lags = np.arange(0, max_lag+1)
    
    # Trouver le meilleur lag
    best_lag = np.argmax(np.abs(cross_corr))
    max_corr = cross_corr[best_lag]
    
    # Intervalle de confiance (approximation pour grandes séries)
    n = len(serie1)
    ic = 1.96 / np.sqrt(n)  # 95% de confiance
    
    # Résultats
    resultats = {
        'meilleur_lag': int(best_lag),
        'max_correlation': float(max_corr),
        'correlations': cross_corr,
        'lags': lags,
        'intervalle_confiance': ic,
        'n_observations': n
    }
    
    # Affichage des résultats
    print("="*60)
    print("ANALYSE DE CORRÉLATION CROISÉE")
    print("="*60)
    print(f"Série 1: {col1}")
    print(f"Série 2: {col2}")
    print(f"Nombre d'observations: {n}")
    print(f"\nMeilleur décalage (lag): {best_lag}")
    print(f"Corrélation maximale: {max_corr:.4f}")
    print(f"Intervalle de confiance (95%): ±{ic:.4f}")
    
    if abs(max_corr) > ic:
        print(f"\n✓ La corrélation est SIGNIFICATIVE")
        if best_lag == 0:
            print(f"  → Les séries sont corrélées sans décalage")
        else:
            print(f"  → '{col2}' suit '{col1}' avec un décalage de {best_lag} périodes")
    else:
        print(f"\n✗ La corrélation n'est PAS significative")
    
    # Trouver les autres lags significatifs
    lags_significatifs = np.where(np.abs(cross_corr) > ic)[0]
    if len(lags_significatifs) > 1:
        print(f"\nAutres lags significatifs: {lags_significatifs.tolist()}")
    
    # Visualisation
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Graphique 1: Séries temporelles
        ax1 = axes[0]
        ax1.plot(serie1.values, label=col1, alpha=0.7, linewidth=2)
        ax1.plot(serie2.values, label=col2, alpha=0.7, linewidth=2)
        ax1.set_xlabel('Index temporel', fontsize=11)
        ax1.set_ylabel('Valeur', fontsize=11)
        ax1.set_title('Séries temporelles', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Graphique 2: Corrélation croisée
        ax2 = axes[1]
        ax2.stem(lags, cross_corr, basefmt=' ', linefmt='C0-', markerfmt='C0o')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.axhline(y=ic, color='red', linestyle='--', linewidth=1.5, 
                    label=f'IC 95% (±{ic:.3f})')
        ax2.axhline(y=-ic, color='red', linestyle='--', linewidth=1.5)
        
        # Marquer le meilleur lag
        ax2.plot(best_lag, max_corr, 'r*', markersize=15, 
                label=f'Max: lag={best_lag}, r={max_corr:.3f}')
        
        ax2.set_xlabel('Décalage (lag)', fontsize=11)
        ax2.set_ylabel('Corrélation', fontsize=11)
        ax2.set_title(f'Fonction de corrélation croisée: {col1} vs {col2}', 
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(-1, max_lag+1)
        
        plt.tight_layout()
        plt.show()
    
    return resultats