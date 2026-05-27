import os
import re

blacklist = [
    "http", "www.", ".com", ".org", ".net",
    "©", "copyright", "tous droits réservés", "all rights reserved",
    "isbn", "issn", "numérisation", "poids du fichier",
    "ebook", "e-book", "epub", "format électronique",
    "licence", "license", "gutenberg", "feedbooks", "ebooksgratuits",
    "table des matières", "table of contents",
    "propos de cette édition", "version numérique",
    "ne peut être vendu", "vendu séparément",
    "couverture", "frontispice", "illustration de", "gravure", "cette édition numérisée",
    "nouvelle édition", "professeur au collége", "tome deuxième",
    "droits de reproduction", "représentée pour la première fois",
    "préface", "introduction", "notice sur", "avertissement",
    "bibliothèque", "calmann lévy", "maison michel lévy","la ponctuation n'a pas été", "l'orthographe a été conservé",
    "we thank the"
]

FOOTER_PATTERNS = [
    r"à\s*propos\s*de\s*cette\s*édition",
    r"cet\s*ouvrage\s*est\s*le",
    r"bibliothèque\s*électronique",
    r"reproduction\s*interdite",
    r"ce\s*livre\s*numérique",
    r"la\s*photo\s*de\s*première\s*page",
    r"nous\s*sommes\s*des\s*bénévoles",
    r"end\s*of\s*the",
    r"fin\s*de\s*don",
    r"\{1\}"
]

def clean_book_content(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        full_text = f.read()

    # --- ÉTAPE 0 : LE "HARD CUT" GUTENBERG ---
    # On cherche la balise de début officielle. Si on la trouve, on jette tout ce qui précède.
    guten_match = re.search(r"\*\*\* START OF.*? \*\*\*", full_text, re.IGNORECASE)
    if guten_match:
        full_text = full_text[guten_match.end():]

    lines = full_text.splitlines()
    total_lines = len(lines)
    if total_lines < 10: return ""

    # --- ÉTAPE 1 : TROUVER LA FIN (FOOTER) ---
    # On cherche dans les derniers 50% du texte restant
    end_idx = total_lines
    search_from = int(total_lines * 0.5)

    for i in range(search_from, total_lines):
        line_lower = lines[i].lower()
        if any(re.search(pattern, line_lower) for pattern in FOOTER_PATTERNS):
            end_idx = i
            break
    
    valid_lines = lines[:end_idx]

    # --- ÉTAPE 2 : TROUVER LE DÉBUT RÉEL (HEADER) ---
    start_idx = 0
    # On analyse les 300 premières lignes pour trouver le déclencheur narratif
    for i in range(min(300, len(valid_lines))):
        line_strip = valid_lines[i].strip()
        line_lower = line_strip.lower()
        
        if not line_strip:
            continue
            
        # Si la ligne contient un mot de la blacklist, on continue de sauter
        if any(word in line_lower for word in blacklist):
            start_idx = i + 1
            continue
        
        # DÉCLENCHEUR : Ligne longue (>60), commence par une Majuscule, pas de mot blacklisté
        # C'est ici que Germinal sera détecté ("Dans la plaine rase...")
        if len(line_strip) > 60 and line_strip[0].isupper():
            start_idx = i
            break

    # --- ÉTAPE 3 : RECONSTRUCTION ---
    final_content = []
    for l in valid_lines[start_idx:]:
        s = l.strip()
        if s:
            final_content.append(s)
            
    return " ".join(final_content)

def build_corpus(input_dir, output_file):
    count = 0
    with open(output_file, "w", encoding="utf-8") as f_out:
        for filename in sorted(os.listdir(input_dir)):
            if filename.endswith(".txt"):
                path = os.path.join(input_dir, filename)
                clean_text = clean_book_content(path)
                
                if len(clean_text) > 500:
                    f_out.write("\n\n<|book_start|>\n\n")
                    f_out.write(clean_text)
                    f_out.write("\n\n<|book_end|>\n\n")
                    count += 1
                    print(f"✅ {filename} : Purifié")
    
    print(f"\n✨ Corpus terminé : {count} livres intégrés dans {output_file}")

if __name__ == "__main__":
    build_corpus("data/txt_litt", "data/corpus_litteraire.txt")
