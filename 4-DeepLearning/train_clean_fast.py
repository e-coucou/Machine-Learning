import os

def clean_local():
    input_dir = "data/epub/txt"
    output_file = "data/corpus_final_manuel.txt"
    
    # On définit les "mots de fin" du blabla
    # On prend des versions courtes pour éviter les pièges d'accents
    markers = ["en aucun cas être vendu", "START OF THE PROJECT", "Ebooks libres et gratuits"]

    with open(output_file, "w", encoding="utf-8") as f_out:
        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                path = os.path.join(input_dir, filename)
                with open(path, "r", encoding="utf-8", errors="ignore") as f_in:
                    content = f_in.read()
                
                # On cherche la position de découpe
                cut_idx = 0
                for m in markers:
                    pos = content.rfind(m) # Cherche la DERNIÈRE occurrence
                    if pos != -1:
                        cut_idx = max(cut_idx, pos + len(m))
                
                # On coupe et on nettoie les espaces
                clean_text = content[cut_idx:].strip()
                
                # On vire les restes de points ou de headers mal coupés
                if clean_text:
                    # On saute les 100 premiers caractères s'ils sont encore du "bruit"
                    f_out.write(f"\n\n--- {filename} ---\n\n")
                    f_out.write(clean_text)
                    print(f"✅ Nettoyé : {filename}")

if __name__ == "__main__":
    clean_local()
