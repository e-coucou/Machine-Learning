import rust_bpe as bpe
import time
import tools.ai_token as tk
import re


def read_file(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def text_sanitizer(text_raw):
    # 1. Remplacements de caractères (très rapide)
    replacements = {
        '—': '-', '«': '"', '»': '"', '“': '"', '”': '"',
        '[': '(', ']': ')', '{': '(', '}': ')',
        '…': '...', '’': "'", '‘': "'"
    }
    for old, new in replacements.items():
        text_raw = text_raw.replace(old, new)

    # 2. Filtrage global via Regex (beaucoup plus rapide que la boucle for)
    # On définit ce qu'on veut GARDER
    # Le tiret est déplacé à la fin, juste avant le ]
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

def preprocess_text(data):
    print(f"--- Nettoyage du dataset: {data[:300]} ---")
    # Regex pour capturer les balises <doc ...> et </doc>
    # On utilise une version qui capture aussi les retours à la ligne potentiels
    tag_re = re.compile(r'<doc.*?>|</doc>')
        
    # Suppression des balises
    clean_content = tag_re.sub('', data)
    # Suppression des lignes vides superflues
    clean_content = re.sub(r'\n\s*\n', '\n', clean_content)

    sanitize_content = text_sanitizer(clean_content)
    
    print(f"Taille après nettoyage : {len(clean_content) / 1e6:.2f} MB")
    return sanitize_content

def main(file,num_merges = 100):

    # 0. Lecture du fichier
    content = read_file(file)
    init_chars = len(set(content))
    print(f"Nombre de caractères uniques dans le fichier : {init_chars}")
    
    # 1. Nettoyage
    text = preprocess_text(content)
    print(f"Texte sans balise ? {text[:300]}")
    #1.1 On charge le tokenizer pour finir le clening du texte
    old_token = tk.BPETokenizer(texte = text , addToken = num_merges)
    data = old_token.text_cleaned
   
    # 2. Entraînement Rust
    print(f"--- Lancement de l'entraînement BPE (Cible: +{num_merges} tokens) ---")
    trainer = bpe.BPETrainer(num_merges)
    
    start_time = time.time()
    vocab, encoded_ids = trainer.train(text)
    end_time = time.time()
    
    # 3. Statistiques
    duration = end_time - start_time
    tokens_per_sec = len(encoded_ids) / duration
    
    print(f"--- Entraînement terminé en {duration:.2f}s ---")
    print(f"Vitesse : {tokens_per_sec:.2f} tokens/s")
    print(f"Taille finale du vocabulaire : {len(vocab)}")
    
#    reset_time = time.time()
#    ids = old_token.train()
#    python_time = time.time()
#    print(f"Python : {python_time - reset_time}")

    # 3. Sauvegarde du modèle complet (Vocab + Merges)
    # On utilise la méthode .save() qu'on a ajoutée en Rust
    trainer.save("rust_token.json")
    print("Modèle sauvegardé dans rust_token.json")

    # 4. Petit test de vérification
    # On affiche les 10 derniers tokens créés (les plus longs/complexes)
    print("\nTop 10 des derniers tokens fusionnés :")
    sorted_ids = sorted(vocab.keys(), reverse=True)
    for i in range(10):
        token_id = sorted_ids[i]
        # On décode les bytes en texte
        token_str = bytes(vocab[token_id]).decode('utf-8', errors='replace')
        print(f"ID {token_id}: '{token_str}'")

def test_encode(file):

    content = read_file(file)
    data = preprocess_text(content)

    # 1. Charger le tokenizer
    tokenizer = bpe.BPETrainer.load("rust_token.json")
    print(f"Vocab chargé : {len(tokenizer.vocab)} tokens") # Si 0, le load a échoué !

    # 2. Texte original
#    original_text = "Le chant grégorien est un trésor monastique."
    original_text = data

    # 3. Encodage (Texte -> IDs)
    ids = tokenizer.encode(original_text)
    print(f"IDs : {ids[:200]}")

    # 4. Décodage (IDs -> Texte)
    decoded_text = tokenizer.decode(ids)
    print(f"Décodé : {decoded_text[:200]}")

    # 5. Vérification
    if original_text == decoded_text:
        print("Succès ! La reconstruction est parfaite.")
    else:
        print("pas marché !")  

if __name__ == "__main__":
    file = "../../data/corpus_tokenizer_15mb.txt"
    main(file,num_merges = 20000)
#    test_encode(file)
