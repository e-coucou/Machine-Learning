
import torch

checkpoint_path = 'model/my_wiki.pth'
# On charge sur le CPU pour ne pas stresser le GPU pendant l'entraînement
ckpt = torch.load(checkpoint_path, map_location='cpu')

print(f"--- 🔎 Diagnostic du Checkpoint ---")
print(f"📍 Step Global   : {ckpt.get('total_steps_done', 'N/A')}")
print(f"📚 Cursor Wiki    : {ckpt.get('total_step_wiki', 'N/A')}")
print(f"📚 Cursor Cult    : {ckpt.get('total_step_cult', 'N/A')}")

# Petit calcul de ratio réel pour vérifier tes 35%
if 'step_wiki' in ckpt and 'step_cult' in ckpt:
    total = ckpt['total_steps_done'] + ckpt['total_step_cult']
    if total > 0:
        ratio_cult = (ckpt['total_step_cult'] / total) * 100
        print(f"📊 Ratio effectif : {ratio_cult:.1f}% CulturaX / {100-ratio_cult:.1f}% Wiki")