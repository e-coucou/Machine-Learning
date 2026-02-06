import os
import re

def clean_gutenberg_hardcore(text):
    # 1. On cherche le marqueur de début (insensible à la casse)
    # Ce marqueur est standard : *** START OF THE PROJECT GUTENBERG EBOOK ... ***
    start_match = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)
    
    if start_match:
        # On ne garde que ce qui est APRES le marqueur
        text = text[start_match.end():]
    
    # 2. On cherche le marqueur de fin
    end_match = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)
    
    if end_match:
        # On ne garde que ce qui est AVANT le marqueur de fin
        text = text[:end_match.start()]

    # 3. Nettoyage des résidus spécifiques (comme les crédits de production)
    # Souvent répétés juste après le début

    # 2. Suppression des séparateurs d'astérisques (ex: * * *, * * *, * *)
    # Ce pattern cherche toute ligne contenant uniquement des astérisques et des espaces
    text = re.sub(r'^[ \t]*\*([ \t]*\*)+[ \t]*$', '', text, flags=re.MULTILINE)

    # 3. Suppression des lignes de "Table des matières" (lignes avec beaucoup de points)
    # Ex: Chapitre premier ................... 12
    text = re.sub(r'\.\.+\s*\d*$', '', text, flags=re.MULTILINE)

    patterns_a_supprimer = [
        r"Produced by.*",
        r"Translated by.*",
        r"Credits:.*",
        r"\[eBook #\d+\]"
    ]
    
    for pattern in patterns_a_supprimer:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # 4. Nettoyage des sauts de ligne excessifs
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 3. Normalisation typographique
    # Remplacer les tirets cadratins par des tirets simples ou espaces
    text = text.replace('—', '-').replace('–', '-')
    # Remplacer les guillemets exotiques
    text = text.replace('«', '"').replace('»', '"').replace('“', '"').replace('”', '"')
    
    # 4. Supprimer les retours à la ligne inutiles (souvent les .txt classiques ont des sauts à 80 char)
    # On ne garde que les doubles retours (paragraphes)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # 5. Supprimer les espaces multiples
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

# Utilisation
input_dir = "data/epub/txt"
output_file = "data/corpus_litteraire_clean.txt"

with open(output_file, "w", encoding="utf-8") as f_out:
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f_in:
                print(f"Nettoyage de {filename}...")
                raw_content = f_in.read()
                cleaned_content = clean_gutenberg_hardcore(raw_content)
                f_out.write(cleaned_content + "\n\n")

print(f"Terminé ! Ton fichier unique est prêt : {output_file}")
