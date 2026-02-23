import re
import os
import glob

# --- CONFIG ---
data_root = "data/train/wiki_fr_test" 
output_file_train = "data/corpus_wiki_clean_train.txt"
output_file_val = "data/corpus_wiki_clean_val.txt"

def clean_wiki_content(text):
    # 1. Supprimer les balises <doc ...> et </doc>
    text = re.sub(r'<doc [^>]*>', '', text)
    text = re.sub(r'</doc>', '', text)
    
    # 2. Remplacer les entités HTML
    text = text.replace('&amp;', '&').replace('&quot;', '"').replace('&apos;', "'")
    text = text.replace('&lt;', '<').replace('&gt;', '>')
    
    # 3. Nettoyer les sauts de ligne excessifs
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def process_files():
    all_articles = []
    # 1. Lister tous les fichiers
    files = glob.glob(os.path.join(data_root, "**", "*"), recursive=True)
    # Filtrer pour ne garder que les fichiers (évite les dossiers)
    files = [f for f in files if os.path.isfile(f)]
    print(f"📚 Trouvé {len(files)} fichiers.")

    for f_path in files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
                if len(raw_text) > 0:
                    clean_text = clean_wiki_content(raw_text)
                    if clean_text:
                        all_articles.append(clean_text)
        except Exception as e:
            print(f"Skipped {f_path}: {e}")

    # 4. Split Train (90%) / Val (10%) sur la LISTE des articles
    # Comme ça, on coupe entre deux articles, jamais au milieu d'une phrase !
    num_train = int(0.9 * len(all_articles))
    train_articles = all_articles[:num_train]
    val_articles = all_articles[num_train:]

    # On joint avec un double saut de ligne pour bien séparer les documents
    return "\n\n".join(train_articles), "\n\n".join(val_articles)

if __name__ == '__main__':
    train_str, val_str = process_files()

    with open(output_file_train, 'w', encoding='utf-8') as f:
        f.write(train_str)

    with open(output_file_val, 'w', encoding='utf-8') as f:
        f.write(val_str)

    print(f"✅ Terminé !")
    print(f"   Train: {len(train_str)/1e6:.2f} millions de caractères")
    print(f"   Val:   {len(val_str)/1e6:.2f} millions de caractères")
