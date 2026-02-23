import tools.ai_token as tk
import os

# --- CONFIG ---
data_root = "data/train/wiki_fr_test" # Ton dossier
output_dir = "data/encoded"
os.makedirs(output_dir, exist_ok=True)

# Ton tokenizer (remplace par le tien si c'est une classe custom)
# tokenizer = ... 
# enc = tiktoken.get_encoding("gpt2") 

def process_files_vf():
    # 1. Lister tous les fichiers
    files = glob.glob(os.path.join(data_root, "**", "*"), recursive=True)
    print(f"📚 Trouvé {len(files)} fichiers.")

    # 2. Tout lire dans une liste géante (Attention à la RAM ici si > 10Go de texte)
    # Pour 6000 fichiers wiki, ça tient large sur un M1
    all_tokens = []
    
    print("🔄 Tokenization en cours...")
    for f_path in (files):
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                text = f.read()
                if len(text) > 0:
                    # Ajout du token de fin de texte (EOT) pour séparer les docs
                    tokens = tko.encode(text) # + [tko.eot_token] 
                    all_tokens.extend(tokens)
        except Exception as e:
            print(f"Skipped {f_path}: {e}")

    total_tokens = len(all_tokens)
    print(f"📊 Total tokens: {total_tokens}")

    # 3. Conversion en numpy array optimisé (uint16 suffit si vocab < 65535)
    # Si ton vocab est > 65535, utilise np.int32
    print("💾 Conversion en binaire...")
    data = np.array(all_tokens, dtype=np.uint16)

    # 4. Split Train (90%) / Val (10%)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # 5. Sauvegarde sur disque
    train_data.tofile(os.path.join(output_dir, 'train.bin'))
    val_data.tofile(os.path.join(output_dir, 'val.bin'))
    
    print(f"✅ Terminé ! Fichiers sauvegardés dans {output_dir}")
    print(f"   Train: {len(train_data)/1e6:.2f}M tokens")
    print(f"   Val:   {len(val_data)/1e6:.2f}M tokens")

if __name__ == '__main__':
    process_files_vf()
