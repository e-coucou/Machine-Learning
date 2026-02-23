# import sys
# import torch
import argparse
import tools.ai_token as tk  # Tokinizer
import tools.ai_gpt_ep as gpt  # GenerateGPT

def main():
    # Création du parser avec une description globale (le rôle du programme)
    parser = argparse.ArgumentParser(
        description="✨ GPT Wiki Generator ✨\nCe programme permet de charger un modèle GPT entraîné et de générer du texte à partir d'un prompt en utilisant différentes techniques de sampling (Top-K, Température, Pénalité).",
        formatter_class=argparse.RawTextHelpFormatter # Permet de garder les retours à la ligne dans la description
    )
    
    # --- DÉFINITION DES ARGUMENTS ---
    parser.add_argument("prompt", type=str, help="Le texte de départ (amorce) pour l'IA.")
    parser.add_argument("-t", "--temp", type=float, default=0.8, help="Température de génération (défaut: 0.8).\n- Basse (0.2) : très prévisible, factuel.\n- Haute (1.0+) : créatif, risque de dérailler.")
    parser.add_argument("-n", "--tokens", type=int, default=100, help="Nombre maximum de nouveaux tokens à générer (défaut: 100).")
    parser.add_argument("-k", "--topk", type=int, default=40, help="Limite le choix aux K mots les plus probables (défaut: 40).")
    parser.add_argument("-p", "--rep_penalty", type=float, default=1.2, help="Pénalité de répétition (défaut: 1.2).\n> 1.0 réduit les boucles infinies (ex: 'de l'ADN de l'ADN').")
    parser.add_argument("-m", "--model", type=str, default="model/my_wiki_inference.pth", help="Chemin vers le fichier .pth du modèle (défaut: model/my_wiki_inference.pth).")
    parser.add_argument("-d", "--default", type=str, default="model", help="Nom de la clé du modèle par défaut dans le checkpoint (défaut: 'model') peut être ema_model si disponible.")

    # Analyse des arguments passés dans le terminal
    args = parser.parse_args()

    # --- INITIALISATION ET GÉNÉRATION ---
    token_ = tk.BPETokenizer()
    token_.load_merges('data/ep_merges_full.json')
    token = tk.OptimizedTokenizer(merges=token_.merges)    

#    print(f"📡 Chargement de : {args.model}")
    try:
        # On passe le tokenizer et le chemin du checkpoint
        model_gen = gpt.GenerateGPT(tokenizer=token, ckpt_path=args.model,default_model=args.default)
        model_gen.load_for_inference()
 #       print(model_gen.config)
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return

#    print(f"🧠 Génération en cours... (Penalty: {args.rep_penalty})")
    
    # Appel de la fonction
    print("\n" + "-"*50)
    print("\033[1m"+args.prompt+"\033[0m")
    print(f"🤖 IA : ", end="")
    out_ = model_gen.generate_text(
        prompt=args.prompt, 
        max_new_tokens=args.tokens, 
        temperature=args.temp,
        top_k=args.topk,
        rep_penalty=args.rep_penalty
    )

    # print("\n" + "-"*50)
    # print(f"🤖 IA : {out_}")
    print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
