import os
import sys
import torch
import tools.ai_token as tk
import tools.ai_gpt_ep as gpt

CMDE = "nohup python3 -u train_gpt_ep.py > model/training.log 2>&1 &"     

def main():
    # --- CONFIGURATION DU MODÈLE ---
    # Architecture fixe pour garantir la compatibilité avec les checkpoints
    model_config = {
        'n_embd': 768,
        'num_heads': 12,
        'n_layers': 10,
        'block_size': 256, 
        'dropout': 0.15
    }

    # --- HYPERPARAMÈTRES D'ENTRAÎNEMENT ---
    train_params = {
        'batch_size': 8, # pour libérer de la mémoire ...
        'grad_accum_steps': 16,      # Batch effectif de 128
        'learning_rate': 3e-4, 
        'min_lr': 3e-5, 
        'warmup_iters': 2000,
        'lr_decay_iters': 80000, # initialement 100_000 mais le modèle rebondit vers 7000 steps 
        'eval_interval': 200,
        'eval_iters': 20, # mini 20 pour lisser 
        'save_interval': 200,
        'n_version' : 5,
        'use_compile': False,
        'cult_data': True,
        'mixed_ratio': 0.35,  # Ratio de données CulturaX dans chaque batch
        'ema_decay': 0., # 0 pour désactiver
        'monitor_interval': 10,
    }

    # --- INITIALISATION DES COMPOSANTS ---
    print(f"🚀 Initialisation de l'entraînement sur {torch.backends.mps.is_available() and 'MPS' or 'CPU'}...")
    
    # Tokenizer
    token = tk.BPETokenizer()
    try:
        token.load_merges('data/ep_merges_full.json')
        vocab_size = len(token.vocab)
        print(f"✅ Tokenizer chargé (Vocab: {vocab_size})")
    except FileNotFoundError:
        print("❌ Erreur : Fichier merges introuvable.")
        return

    # Trainer
    trainer = gpt.ContinuousTrainer(
        model_class=gpt.GPTLanguageModel,
        tokenizer=token,
        config=model_config,
        train_params=train_params,
        data_root="data/train/wiki_fr_test",
        log_file="model/my_wiky_log.txt",
        ckpt_path="model/my_wiki.pth",
        history_path='model/my_wiky_history.json'
    )

    # --- LANCEMENT ---
    print(f"📦 Démarrage du mode binaire (Continuous Training)")
    print(f"🌡️  Surveillance Mac Pro : On lance MAc Fans Control pour refroidir ....")
    # --- PRINT STARTUP CONFIG ---
    U = {"G": "\033[92m", "Y": "\033[93m", "C": "\033[96m", "B": "\033[1m", "RE": "\033[0m"}
    # Affichage Architecture
    print(f"{U['C']}[Model config]{U['RE']}")
    for k, v in model_config.items():
        color = U['Y'] if k == 'dropout' else ""
        print(f"  {k:<18} : {color}{v}{U['RE']}")

    # Affichage Hyperparamètres
    print(f"{U['C']}[Train params]{U['RE']}")
    for k, v in train_params.items():
        color = U['G'] if k in ['eval_interval', 'learning_rate'] else ""
        print(f"  {k:<18} : {color}{v}{U['RE']}")

    # Calcul du Batch Effectif (pour info)
    eff_batch = train_params['batch_size'] * train_params['grad_accum_steps']
    print(f"  {U['B']}Effective Batch Size: {eff_batch}{U['RE']}")
    print("."*50 + "\n")    

    try:
        # Lance la boucle d'entraînement principale
        trainer.train_bin()
    except KeyboardInterrupt:
        print("\n\n🛑 Entraînement interrompu par l'utilisateur.")
        print("💾 Sauvegarde de sécurité en cours...")
        # Optionnel : forcer une sauvegarde ici si ton trainer ne le fait pas déjà
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur critique durant l'entraînement : {e}")
        raise e

if __name__ == '__main__':
    main()
