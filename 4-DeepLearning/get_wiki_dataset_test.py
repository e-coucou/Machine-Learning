# Configuration
base_folder = "data/train/wiki_fr_test"
output_file = "data/corpus_wiki_all.txt"
num_files_to_merge = 15

# 1. Récupérer tous les fichiers de tous les répertoires
all_files = []

# Parcourir tous les répertoires (AA à CN)
for subdir in os.listdir(base_folder):
    subdir_path = os.path.join(base_folder, subdir)
    
    # Vérifier que c'est bien un répertoire
    if os.path.isdir(subdir_path):
        # Récupérer tous les fichiers .txt du répertoire
        files_in_subdir = [f for f in os.listdir(subdir_path)]
        
        # Ajouter le chemin complet à la liste
        for fname in files_in_subdir:
            full_path = os.path.join(subdir_path, fname)
            all_files.append(full_path)

print(f"Total fichiers trouvés : {len(all_files)}")

# 2. Sélection aléatoire
if len(all_files) > num_files_to_merge:
    selected_files = random.sample(all_files, num_files_to_merge)
else:
    selected_files = all_files
    print(f"Attention : Seulement {len(selected_files)} fichiers disponibles")

print(f"Fusion de {len(selected_files)} fichiers...")

# 3. Fusion
total_size = 0
with open(output_file, 'w', encoding='utf-8') as outfile:
    for file_path in selected_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content + "\n")
                total_size += len(content.encode('utf-8'))
                print(f"✓ Ajouté : {file_path}")
        except Exception as e:
            print(f"✗ Erreur avec {file_path} : {e}")

print(f"\nFichier créé : {output_file}")
print(f"Taille approximative : {total_size / (1024*1024):.2f} MB")
