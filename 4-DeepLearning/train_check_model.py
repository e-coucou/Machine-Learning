import torch
import os

def inspect_checkpoint(inference_path, train_path):
    print("--- Analyse des Checkpoints ---")
    
    # 1. Analyse du modèle d'inférence
    if os.path.exists(inference_path):
        sd = torch.load(inference_path, map_location='cpu')
        # Vérification si 'model' est une clé ou si c'est le state_dict direct
        model = sd['model'] if 'model' in sd else sd
        print(f"✅ Clés du checkpoint d'inférence : {list(sd.keys())}")
        
        n_params = sum(t.numel() for t in model.values() if isinstance(t, torch.Tensor))
        print(f'✅ Taille réelle : {n_params/1e6:.2f}M paramètres')
    else:
        print(f"❌ {inference_path} introuvable.")

    # 2. Analyse du checkpoint d'entraînement (pour les optimizers/époques)
    if os.path.exists(train_path):
        ckpt = torch.load(train_path, map_location='cpu')
        print(f"✅ Clés du checkpoint d'entraînement : {list(ckpt.keys())}")
        print(f"    - Steps : {ckpt['total_steps_done']}")
        print(f"    - Wiki  : {ckpt['total_step_wiki']}")
        print(f"    - Cult  : {ckpt['total_step_cult']}")
        print(f"    - Vocab : {ckpt['vocab_size']}")
        print(f"- CONFIG ----")
        for c in ckpt['config']:
            print(f"    {c} : {ckpt['config'][c]}")
    else:
        print(f"❌ {train_path} introuvable.")

if __name__ == "__main__":
    inspect_checkpoint('model/my_wiki_inference.pth', 'model/my_wiki.pth')
