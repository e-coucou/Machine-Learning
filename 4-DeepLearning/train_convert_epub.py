import os
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

def epub_to_text(epub_path):
    try:
        book = epub.read_epub(epub_path)
        chapters = []
        
        # On parcourt les éléments du livre
        for item in book.get_items():
            # On ne récupère que les documents de type texte (HTML/XHTML)
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                
                # Extraction du texte en supprimant les scripts et styles
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                
                # Nettoyage basique des lignes vides
                clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
                chapters.append(clean_text)
        
        return "\n\n".join(chapters)
    
    except Exception as e:
        print(f"❌ Erreur lors de la lecture de {epub_path}: {e}")
        return None

def batch_convert(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".epub"):
            print(f"📖 Conversion de : {filename}...")
            epub_path = os.path.join(input_folder, filename)
            
            text_content = epub_to_text(epub_path)
            
            if text_content:
                # On crée le nom du fichier .txt
                text_content = '/START/* '+ text_content
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_path = os.path.join(output_folder, txt_filename)
                
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
                print(f"✅ Sauvegardé : {txt_filename}")

# --- CONFIGURATION ---
dossier_source = "data/epub"       # Mets tes EPUB ici
dossier_destination = "data/epub/txt"   # Les fichiers TXT iront ici

batch_convert(dossier_source, dossier_destination)
