
# import torch

# checkpoint_path = 'model/my_wiki.pth'
# # On charge sur le CPU pour ne pas stresser le GPU pendant l'entraînement
# ckpt = torch.load(checkpoint_path, map_location='cpu')

# print(f"--- 🔎 Diagnostic du Checkpoint ---")
# print(f"📍 Step Global   : {ckpt.get('total_steps_done', 'N/A')}")
# print(f"📚 Cursor Wiki    : {ckpt.get('total_step_wiki', 'N/A')}")
# print(f"📚 Cursor Cult    : {ckpt.get('total_step_cult', 'N/A')}")

# # Petit calcul de ratio réel pour vérifier tes 35%
# if 'step_wiki' in ckpt and 'step_cult' in ckpt:
#     total = ckpt['total_steps_done'] + ckpt['total_step_cult']
#     if total > 0:
#         ratio_cult = (ckpt['total_step_cult'] / total) * 100
#         print(f"📊 Ratio effectif : {ratio_cult:.1f}% CulturaX / {100-ratio_cult:.1f}% Wiki")

import torch

path = 'model/my_wiki.pth'
ckpt = torch.load(path)

# On définit le point de bascule où l'unité a changé
START_STEP = 49800 

params = ckpt.get('params', None)
batch_size = params.get('batch_size',32)
current_wiki = ckpt.get('total_step_wiki', START_STEP)
current_cult = ckpt.get('total_step_cult', 0)
current_step = ckpt.get('total_steps_done', START_STEP)
print(f"📍 Wiki : {ckpt['total_step_wiki']} steps")
print(f"📍 Cult : {ckpt['total_step_cult']} steps")
print(f"📍 Total Mega: {ckpt['total_steps_done']} steps")
# 1. On calcule ce qui a été fait DEPUIS la reprise en micro-batches
new_batches_wiki = current_wiki * batch_size
new_batches_cult = current_cult * batch_size
print(f"📍 Wiki : {new_batches_wiki} steps")
print(f"📍 Cult : {new_batches_cult} steps")


# 2. On convertit ces nouveaux batches en steps réels (// 4)
#compare_wiki = current_wiki - START_STEP*4

# 3. On met à jour le dictionnaire
ckpt['total_step_wiki'] =  new_batches_wiki
ckpt['total_step_cult'] =  new_batches_cult

#print(f"✅ Vérification : {current_cult/(current_cult+compare_wiki)*100:.2f}% CulturaX (devrait être proche de 35%)")

#torch.save(ckpt, 'model/my_wiki_fixed.pth')
