import pandas as pd
import tools.ai_token as tk
import numpy as np
import regex as re
import os

def read_file(fileName):
    with open(fileName,'r',encoding='utf-8') as f:
        text_raw = f.read()
    return text_raw

def preprocess_text(text_raw):
    # 1. Remplacements de caractères (très rapide)
    replacements = {
        '—': '-', '«': '"', '»': '"', '“': '"', '”': '"',
        '[': '(', ']': ')', '{': '(', '}': ')',
        '…': '...', '’': "'", '‘': "'",'--':'-'
    }
    for old, new in replacements.items():
        text_raw = text_raw.replace(old, new)

    # 2. Filtrage global via Regex (beaucoup plus rapide que la boucle for)
    # On définit ce qu'on veut GARDER
    keep_pattern = r'[^a-zA-Z0-9 .,;:!?\'\"\nàâçèéêëîïôùûÀÂÇÉÈÊËÎÏÔÙÛœŒ()-]'
    text_cleaned = re.sub(keep_pattern, '', text_raw)

    # 3. Normalisation de la ponctuation (pas d'espace avant)
    text_cleaned = re.sub(r'\s+([.,;:!?])', r'\1', text_cleaned)

    # 4. Normalisation finale des espaces
    # On remplace les tabs et espaces multiples par un seul espace
    text_cleaned = re.sub(r'[ \t]+', ' ', text_cleaned)
    # On limite à maximum 2 sauts de ligne (garde les paragraphes, vire le vide)
    text_cleaned = re.sub(r'\n{3,}', '\n\n', text_cleaned)
    
    text_cleaned = text_cleaned.strip()
    
    return text_cleaned


# on charge le tokenizer
input_file = "data/corpus_wiki_clean_train.txt"
token = tk.BPETokenizer()
token.load_merges('data/ep_merges_full.json')
#print(len(token.vocab), len(token.text_cleaned))
tko = tk.OptimizedTokenizer(merges=token.merges)
print(f"Taille du merge: {len(token.merges)} et donc du vocab associé: {len(tko.vocab)}")
text_raw = read_file(input_file)
print(f"Le fichier raw contient : {len(text_raw)}")
print(f"On procède au nettoyage du fichier ...")
text_cleaned = preprocess_text(text_raw)

print(f"Fichier nettoyé : {len(text_cleaned)}\n---------------")
print(text_cleaned[:10000])

output_file = "train_wiki.bin"
output_dir = "data/encoded"

print(f"Encodage en cours ...")
# On charge le parquet
all_tokens = []

ids = tko.encode(text_cleaned)
all_tokens.extend(ids)

total_tokens = len(all_tokens)

print(f"📊 Total tokens: {total_tokens}")

# 3. Conversion en numpy array optimisé (uint16 suffit si vocab < 65535)
# Si ton vocab est > 65535, utilise np.int32
print("💾 Conversion en binaire...")
data = np.array(all_tokens, dtype=np.uint16)

# 5. Sauvegarde sur disque
data.tofile(os.path.join(output_dir, output_file))

print(f"✅ Terminé ! Fichiers sauvegardés dans {output_dir}")
print(f"   Train: {len(data)/1e6:.2f}M tokens")

        # bin_ids = np.array(ids, dtype=np.uint16)
        # f.write(bin_ids.tobytes())
