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