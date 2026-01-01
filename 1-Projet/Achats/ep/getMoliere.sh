#!/bin/bash

# Dossier temporaire
mkdir -p moliere_temp
cd moliere_temp || exit

# Liste des pièces (titres pour URL Wikisource)
pieces=(
"Le_Médecin_volant"
"La_Jalousie_du_barbouillé"
"L'Étourdi_ou_les_Contretemps"
"Le_Dépit_amoureux"
"Les_Précieuses_ridicules"
"Sganarelle_ou_le_Cocu_imaginaire"
"Dom_Garcie_de_Navarre_ou_le_Prince_jaloux"
"L'École_des_maris"
"Les_Fâcheux"
"L'École_des_femmes"
"La_Critique_de_l'École_des_femmes"
"L'Impromptu_de_Versailles"
"Le_Mariage_forcé"
"La_Princesse_d'Élide"
"Tartuffe_ou_l'Imposteur"
"Dom_Juan_ou_le_Festin_de_pierre"
"L'Amour_médecin"
"Le_Misanthrope_ou_l'Atrabilaire_amoureux"
"Le_Médecin_malgré_lui"
"Mélicerte"
"Pastorale_comique"
"Le_Sicilien_ou_l'Amour_peintre"
"Amphitryon"
"George_Dandin_ou_le_Mari_confondu"
"L'Avare_ou_l'École_du_mensonge"
"Monsieur_de_Pourceaugnac"
"Les_Amants_magnifiques"
"Le_Bourgeois_gentilhomme"
"Psyché"
"Les_Fourberies_de_Scapin"
"La_Comtesse_d'Escarbagnas"
"Les_Femmes_savantes"
"Le_Malade_imaginaire"
)

# Fichier final
output="../moliere_corpus_clean.txt"
> "$output"

# Fonction de nettoyage
clean_text() {
    # Supprime didascalies entre crochets, noms de personnages, actes/scènes
    sed -E \
        -e 's/\[[^]]*\]//g' \
        -e 's/^[A-ZÉÈÀ\-]{2,}\.//g' \
        -e '/^ACTE/d' \
        -e '/^SCENE/d' \
        -e '/^\s*$/d'
}

# Télécharger chaque pièce et nettoyer
for piece in "${pieces[@]}"; do
    echo "Téléchargement de $piece..."
    
    # Télécharger le texte brut
    curl -s -o temp.txt "https://fr.wikisource.org/w/index.php?title=$piece&action=raw"
    
    # Nettoyer et ajouter au fichier final
    clean_text temp.txt >> "$output"
    
    # Séparation entre pièces
    echo -e "\n\n" >> "$output"
done

# Nettoyage temporaire
rm -f temp.txt
cd ..
rm -rf moliere_temp

echo "Corpus complet généré : $output"
