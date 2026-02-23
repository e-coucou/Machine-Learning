import numpy as np
import regex as re
import os
import tools.ai_token as tk

# --- CONFIG ---
input_file = "data/corpus_wiki_clean_val.txt"
output_file = "data/encoded/val_wiki.bin"
chunk_size = 10 * 1024 * 1024  # On traite 10 Mo de texte à la fois

# Initialisation Tokenizer
token = tk.BPETokenizer()
token.load_merges('data/ep_merges_full.json')
tko = tk.OptimizedTokenizer(merges=token.merges)

def preprocess_text(text):
    # Remplacements rapides
    replacements = {'—': '-', '«': '"', '»': '"', '“': '"', '”': '"', 
                    '[': '(', ']': ')', '{': '(', '}': ')', '…': '...', 
                    '’': "'", '‘': "'", '--': '-'}
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Regex de filtrage
    keep_pattern = r'[^a-zA-Z0-9 .,;:!?\'\"\nàâçèéêëîïôùûÀÂÇÉÈÊËÎÏÔÙÛœŒ()-]'
    text = re.sub(keep_pattern, '', text)
    
    # Ponctuation et espaces
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def process_large_file():
    print(f"🚀 Début de l'encodage streamé...")
    
    # On ouvre le fichier de sortie en mode 'ab' (append binary)
    with open(output_file, 'wb') as f_out:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            while True:
                chunk = f_in.read(chunk_size)
                if not chunk:
                    break
                
                # S'assurer de ne pas couper un article ou une ligne au milieu 
                # On lit jusqu'au prochain saut de ligne pour finir proprement le bloc
                extra = f_in.readline()
                chunk += extra
                
                # 1. Nettoyage
                clean_chunk = preprocess_text(chunk)
                
                # 2. Encodage
                ids = tko.encode(clean_chunk)
                
                # 3. Conversion immédiate en uint16 et écriture disque
                if ids:
                    np_ids = np.array(ids, dtype=np.uint16)
                    f_out.write(np_ids.tobytes())
                    
                print(f"📦 Bloc traité... (Sortie: {os.path.getsize(output_file)/1e6:.2f} Mo)", end="\r")

if __name__ == '__main__':
    if not os.path.exists("data/encoded"):
        os.makedirs("data/encoded")
    
    # On vide le fichier s'il existe déjà
    if os.path.exists(output_file):
        os.remove(output_file)
        
    process_large_file()
    print("\n✅ Encodage terminé avec succès !")
