import pandas as pd

def ajouter_element_campagne(df, fill_empty=False):
    """
    Ajoute trois colonnes au dataframe : 'Element' (M ou R), 'M_Campagne' et 'R_Campagne'.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame contenant les colonnes M_Poids et R_Poids
    fill_empty : bool, default=False
        Si True, propage l'élément sur les lignes vides (forward fill)
        Si False, les lignes sans données restent à NaN
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame avec les nouvelles colonnes 'Element', 'M_Campagne' et 'R_Campagne'
    """
    import numpy as np
    
    # Créer une copie pour ne pas modifier l'original
    df_result = df.copy()
    
    # Identifier l'élément actif basé uniquement sur M_Poids et R_Poids
    df_result['Element'] = np.where(
        df_result['A1000M_Poids'].notna(),
        'M',
        np.where(
            df_result['A1000R_Poids'].notna(),
            'R',
            np.nan
        )
    )
    
    # Détecter les transitions d'élément sur toutes les lignes (y compris NaN)
    element_change = (df_result['Element'] != df_result['Element'].shift()) & df_result['Element'].notna()
    
    # Initialiser les colonnes de campagne
    df_result['M_Campagne'] = np.nan
    df_result['R_Campagne'] = np.nan
    
    # Compteurs de campagne
    m_counter = 0
    r_counter = 0
    
    # Parcourir le dataframe pour attribuer les numéros de campagne
    for idx in df_result.index:
        if df_result.loc[idx, 'Element'] == 'M':
            if element_change.loc[idx]:
                m_counter += 1
            df_result.loc[idx, 'M_Campagne'] = m_counter
        elif df_result.loc[idx, 'Element'] == 'R':
            if element_change.loc[idx]:
                r_counter += 1
            df_result.loc[idx, 'R_Campagne'] = r_counter
    
    # S'assurer que les compteurs commencent à 1 si des valeurs existent
    if df_result['M_Campagne'].notna().any() and df_result['M_Campagne'].min() == 0:
        df_result.loc[df_result['M_Campagne'].notna(), 'M_Campagne'] += 1
    if df_result['R_Campagne'].notna().any() and df_result['R_Campagne'].min() == 0:
        df_result.loc[df_result['R_Campagne'].notna(), 'R_Campagne'] += 1
    
    return df_result


def reincrementer_campagnes_par_temps(df, jours_seuil=3):
    """
    Réincrémente les numéros de campagne en fonction des écarts de temps entre les mesures.
    Si l'écart entre deux lignes dépasse le seuil, le numéro de campagne est incrémenté.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame avec timestamp en index (datetime) et une colonne de campagne
    jours_seuil : int or float, default=3
        Nombre de jours au-delà duquel on incrémente la campagne
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame avec les numéros de campagne mis à jour
    """

    # Créer une copie pour ne pas modifier l'original
    df_result = df.copy()
    
    # Identifier la colonne de campagne (M_Campagne ou R_Campagne)
    #if 'M_Campagne' in df_result.columns:
    col_campagne = 'campagne'
    #elif 'R_Campagne' in df_result.columns:
    #    col_campagne = 'R_Campagne'
    #else:
    #    raise ValueError("Le DataFrame doit contenir une colonne M_Campagne ou R_Campagne")
    
    # Calculer les écarts de temps entre les lignes
    time_diff = df_result.index.to_series().diff()
    
    # Détecter les sauts de campagne (écart > seuil)
    saut_campagne = time_diff > pd.Timedelta(days=jours_seuil)
    
    # Réincrémenter les campagnes
    df_result[col_campagne] = saut_campagne.cumsum() + 1

    # Ajout colonne mois-année au format mm-aa
    df_result["mois_annee"] = df_result.index.strftime("%y-%m")
    df_result["mois_annee_dt"] = df_result.index.tz_localize(None).to_period('M').to_timestamp()
    return df_result


def resume_par_campagne(df):
    # On suppose que df contient déjà la colonne "campagne"
    # et que les colonnes sont : Poids, UV, labo, teneur

    # UV pondéré = somme(Poids * UV) / somme(Poids)
    df["UV_pondere"] = df["A1000M_Poids"] * df["A1000M_UV_Auto"]
    df["UV_labo"] = df["A1000M_Poids"] * df["A1000M_Labo"]

    grouped = df.groupby(["campagne","mois_annee","mois_annee_dt"]).agg(
        nb_lots=("A1000M_Poids", "count"),
        poids_total=("A1000M_Poids", "sum"),
        uv_pondere=("UV_labo", lambda x: x.sum() / df.loc[x.index, "A1000M_Poids"].sum())
#        uv_pondere=("UV_pondere", lambda x: x.sum() / df.loc[x.index, "CTY_A1000M_Poids container"].sum())
    )

    return grouped.reset_index()