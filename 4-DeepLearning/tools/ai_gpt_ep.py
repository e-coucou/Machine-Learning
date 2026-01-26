import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os, psutil
import glob
import random, math
import json
import threading
import queue, gc


# -----------------------------------------------------------------------------
# 1. BLOCS DE BASE DU MODÈLE (Architecture GPT "Decoder-Only")
# -----------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """ Causal Self-Attention. C'est le coeur du mécanisme GPT. """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        
        # Projection clés, requêtes, valeurs
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Masque causal (tril) pour empêcher de voir le futur
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.scale = head_size ** -0.5

    def forward(self, x):
        B, T, C = x.shape
        
        # Calcul Q, K, V en une seule opération (optimisé M1)
        qkv = self.qkv(x)  # (B, T, 3*n_embd)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_size)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calcul des scores d'attention
        # (B, num_heads, T, head_size) @ (B, num_heads, head_size, T) -> (B, num_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Application du masque causal : on remplace les 0 du masque par -inf
        # On slice [:T, :T] pour gérer les séquences plus courtes que block_size (ex: génération)
        mask = self.tril[:T, :T]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Agrégation des valeurs
        out = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_size)
        
        # Recomposition
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, T, C)
        
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ Simple réseau de neurones avec GELU (standard moderne) """
    def __init__(self, n_embd, dropout):
        super().__init__()
        hidden_dim = 4 * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Un bloc Transformer standard : Communication (Attention) + Calcul (FFwd) """
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # La Pre-Norm (LayerNorm AVANT l'opération) est plus stable pour l'entraînement
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    """ Le modèle final assemblé (renommé pour être clair) """
    def __init__(self, vocab_size, n_embd, num_heads, n_layers, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(*[
            Block(n_embd, num_heads, block_size, dropout) for _ in range(n_layers)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        # Initialisation des poids (souvent aide à la convergence)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        
        # Embeddings
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb
        
        # Passage dans les blocs
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx est (B, T) tableau d'indices
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # Focus sur le dernier token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
# -----------------------------------------------------------------------------
# 2. CLASSE DE GESTION D'ENTRAÎNEMENT (Optimisée Multi-Fichiers + M1)
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(self, tokenizer, config, parameters, data_folder, file_out='ckpt.pth'):
        self.tokenizer = tokenizer
        self.config = config
        self.params = parameters
        self.file_out = file_out
        self.data_folder = data_folder
        
        # Device Setup
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("🚀 Utilisation du device MPS (Apple Silicon)")
        else:
            self.device = torch.device("cpu")
            print("⚠️ MPS non disponible. Utilisation CPU (Lent)")

        # Init Model
        self.model = GPTLanguageModel(
            vocab_size=len(tokenizer.vocab),
            **self.config
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=parameters['learning_rate'])
        
        # Historique
        self.history = {'train_loss': [], 'files_processed': 0}

    def get_file_list(self):
        return glob.glob(os.path.join(self.data_folder, "*.txt"))

    def get_batch(self, data_tensor):
        """ Extrait un batch aléatoire depuis le tenseur du fichier courant """
        block_size = self.config['block_size']
        batch_size = self.params['batch_size']
        
        ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
        
        x = torch.stack([data_tensor[i:i+block_size] for i in ix])
        y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])
        
        return x.to(self.device), y.to(self.device)

    def train_one_file(self, file_path):
        """ Charge, tokenize et entraîne sur un seul fichier """
        # 1. Chargement
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Erreur lecture {file_path}: {e}")
            return 0.0

        if len(text) < self.config['block_size'] + 10:
            return 0.0 # Fichier trop petit

        # 2. Tokenization & Transfert GPU
        tokens = self.tokenizer.encode(text)
        data_tensor = torch.tensor(tokens, dtype=torch.long) # Sur CPU d'abord pour économiser VRAM
        
        # 3. Calcul du nombre d'itérations pour ce fichier
        # On veut passer environ 1 fois sur tout le fichier
        n_tokens = len(data_tensor)
        batch_tokens = self.params['batch_size'] * self.config['block_size']
        iters_per_file = max(10, n_tokens // batch_tokens) 
        
        # Limite haute pour éviter de rester bloqué sur un énorme fichier
        iters_per_file = min(iters_per_file, 500) 
        
        grad_accum_steps = self.params.get('grad_accum_steps', 4)
        avg_loss = 0
        
        self.model.train()
        
        for i in range(iters_per_file):
            
            # Gradient Accumulation Loop
            loss_accum = 0
            self.optimizer.zero_grad()
            
            for _ in range(grad_accum_steps):
                X, Y = self.get_batch(data_tensor)
                logits, loss = self.model(X, Y)
                
                # Normalisation du loss pour l'accumulation
                loss = loss / grad_accum_steps
                loss_accum += loss.item()
                loss.backward()
            
            self.optimizer.step()
            avg_loss += loss_accum

        return avg_loss / iters_per_file

    def train_global(self):
        epochs = self.params['epochs']
        files = self.get_file_list()
        print(f"Début de l'entraînement sur {len(files)} fichiers pour {epochs} époques.")
        print(f"Paramètres: Batch={self.params['batch_size']}, Accum={self.params.get('grad_accum_steps', 1)}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            random.shuffle(files) # Important pour la généralisation
            
            print(f"\n--- ÉPOQUE {epoch+1}/{epochs} ---")
            
            for i, file_path in enumerate(files):
                fname = os.path.basename(file_path)
                
                loss = self.train_one_file(file_path)
                
                self.history['train_loss'].append(loss)
                self.history['files_processed'] += 1
                
                # Feedback console
                if i % 1 == 0: # Afficher à chaque fichier
                    elapsed = time.time() - start_time
                    print(f"[{epoch+1}] File {i+1}/{len(files)} ({fname}) -> Loss: {loss:.4f} | Time: {elapsed:.1f}s")
                
                # Sauvegarde régulière (checkpoint)
                if i % 10 == 0:
                    self.save_checkpoint()

        print("Entraînement terminé.")
        self.save_checkpoint()

    def save_checkpoint(self):
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
            'vocab_size': len(self.tokenizer.vocab)
        }
        torch.save(ckpt, self.file_out)
        # print("Checkpoint saved.")

    def generate_text(self, prompt, max_tokens=100):
        self.model.eval()
        tokens = self.tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)
        gen_idx = self.model.generate(idx, max_tokens)
        return self.tokenizer.decode(gen_idx[0].tolist(), self.tokenizer.vocab)
    
class DeterministicProvider:
    def __init__(self, data, batch_size, block_size, device, start_step, grad_accum_steps=4, seed=1965):
        self.data = data # Le memmap NumPy
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.step = start_step * grad_accum_steps
        self.seed = seed
        
        # 1. Création de la route fixe (Couverture 100%)
        # On définit les points de départ tous les 'block_size'
        self.indices = np.arange(0, len(self.data) - self.block_size, self.block_size)
        self.total_indices = len(self.indices)
        # Calcul de l'epoch actuelle pour caler le shuffle au démarrage
        self.last_epoch = (self.step * self.batch_size) // self.total_indices        
        
        # 2. Shuffle déterministe avec le Seed
        rng = np.random.default_rng(self.seed + self.last_epoch)
        rng.shuffle(self.indices)
        
        # 3. Pré-calcul du vecteur d'offsets pour la grille NumPy
        self.offsets = np.arange(self.block_size)
        
        print(f"✅ Provider HP initialisé : {self.total_indices} blocs uniques. Epoch : {self.last_epoch}")
        print(f"🔄 Reprise au batch {self.step} [step : {start_step}]")

    def __iter__(self):
        while True:
            # À chaque epoch complète, on reshuffle les indices
            current_epoch = (self.step * self.batch_size) // self.total_indices
            if current_epoch > self.last_epoch:
                rng = np.random.default_rng(self.seed + current_epoch)
                rng.shuffle(self.indices)
                self.last_epoch = current_epoch
                print(f"🔄 Nouveau shuffle des indices à l'epoch {self.last_epoch}")
            # Calcul de la position dans la liste d'indices
            start_pos = (self.step * self.batch_size) % self.total_indices
            # print(f"DEBUG PROVIDER: Step {self.step} | Start_pos {start_pos}")            
            # Extraction des indices de départ pour ce batch
            if start_pos + self.batch_size <= self.total_indices:
                ix = self.indices[start_pos : start_pos + self.batch_size]
            else:
                part1 = self.indices[start_pos:]
                part2 = self.indices[:self.batch_size - len(part1)]
                ix = np.concatenate([part1, part2])
            
            # --- LA MAGIE NUMPY (Vectorized Slicing) ---
            # Construction de la grille (batch_size, block_size)
            grid_indices = ix[:, None] + self.offsets
            
            x_np = self.data[grid_indices]
            y_np = self.data[grid_indices + 1]

            # Conversion en Tensor et envoi vers le GPU
            # On le fait ici pour que le BackgroundGenerator livre un produit fini
            x = torch.from_numpy(x_np).to(self.device).long()
            y = torch.from_numpy(y_np).to(self.device).long()

            del x_np,y_np
            
            yield x, y
            self.step += 1

# Supposons que vos classes GPTLanguageModel et tokenizer sont importées ou définies au dessus
# from model import GPTLanguageModel 
class ContinuousTrainer:
    def __init__(self, model_class, tokenizer, config, train_params, data_root, 
                 ckpt_path='ckpt.pth', log_file='processed_log.txt',history_path='history.json',data_dir="data/encoded"):
        
        self.tokenizer = tokenizer
        self.config = config
        self.params = train_params
        self.data_root = data_root
        self.ckpt_path = ckpt_path
        self.log_file = log_file
        self.history_path = history_path
        # Gestion du Device / préférence sur MPS /
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🚀 Device: {self.device}")
        # Instanciation du modèle
        self.model = model_class(vocab_size=len(tokenizer.vocab), **config)
        # On applique l'initialisation personnalisée sur le modèle neuf
        self.model.apply(self._init_weights)
        self.model.to(self.device)
        # Check >Dtype
        print(f"Dtype du modèle : {next(self.model.parameters()).dtype}")
        # Chargement d'un checkpoint existant
        self.total_steps_done = 0
        self._load_checkpoint()
        # Compilation optionnelle avec torch.compile (PyTorch 2.0+)    
        self.is_compiled = False
        if self.params.get('use_compile', False):
            if hasattr(torch, "compile"):
                print("🚀 Activation de torch.compile (Backend: MPS)...")
                try:
                    # On compile AVANT l'optimizer
                    # fullgraph=False est plus stable pour l'architecture GPT
                    self.model = torch.compile(self.model)
                    self.is_compiled = True
                    print("✅ Modèle compilé avec succès.")
                except Exception as e:
                    print(f"⚠️ Échec compilation : {e}")
            else:
                print("⚠️ torch.compile non supporté (nécessite PyTorch 2.0+)")

        # Optimiseur (On l'initialise ici, mais son état peut être écrasé si on charge un checkpoint)
        self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), 
                    lr=train_params['learning_rate'],
                    weight_decay=train_params.get('weight_decay', 0.1)
                )
        # Chargement de l'état de l'optimiseur si disponible
        # 5. Restauration de l'état de l'Optimiseur
        if hasattr(self, '_saved_optimizer_state') and self._saved_optimizer_state:
            self.optimizer.load_state_dict(self._saved_optimizer_state)
            del self._saved_optimizer_state # Libère la mémoire
            print("✅ État de l'optimiseur restauré.")
        # Liste des fichiers déjà traités (obsolète, maintenant on charge un fichier de token unique) & l'historique
        # self.processed_files = self._load_processed_log()
        self.history = self._load_history()    
        self._setup_data(data_dir)
        # Créer un itérateur pour piocher dedans manuellement comme avant
        # self.train_iter = iter(self.train_loader)
        # abandonné au profit du shuffle : // self.train_queue = BackgroundGenerator(lambda: self.get_batch_bin('train'), max_prefetch=1)
        # Préparation du fournisseur de données déterministe
        # Il commence exactement au step où on s'est arrêté
        self.train_provider = DeterministicProvider(
            data=self.train_data,
            batch_size=self.params['batch_size'],
            block_size=self.config['block_size'],
            device = self.device,
            start_step=self.total_steps_done,
            grad_accum_steps=self.params.get('grad_accum_steps', 4),
            seed=1965 # Le Salt fixe
        )
        # 1. On crée l'itérateur persistant ici
        self.train_iterator = iter(self.train_provider)
        # 2. Le BackgroundGenerator utilise l'itérateur existant
        # Note : on utilise 'next(self.train_iterator)' SANS le 'iter()' sinon reset à chaque iter
        self.train_queue = BackgroundGenerator(
            lambda: next(self.train_iterator), 
            max_prefetch=2 # peut monter à 5 pour plus de fluidité ... mais on sature le bus ram unifié et ralenti le GPU !
        )
        # Initialisation (une seule fois au début de la classe) passage en float16 sur MPS
        self.scaler = torch.amp.GradScaler(self.device, enabled=True)
        print(f"🚀 Init terminé.")

    def _setup_data(self,data_dir):
        # --- CHARGEMENT DES DONNÉES BINAIRES (Nouveau) ---
        # On utilise memmap pour lire le fichier sur le disque sans charger la RAM
        train_path = os.path.join(data_dir, 'train.bin')
        val_path = os.path.join(data_dir, 'val.bin')
        
        if os.path.exists(train_path):
            self.train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
            self.val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
            # self.train_data = np.fromfile(train_path, dtype=np.uint16)
            # self.val_data = np.fromfile(val_path, dtype=np.uint16)
            """ test tensor 
            train_np = np.fromfile(train_path, dtype=np.uint16).astype(np.int32)
            self.train_data = torch.from_numpy(train_np).to(self.device)
            val_np = np.fromfile(val_path, dtype=np.uint16).astype(np.int32)
            self.val_data = torch.from_numpy(val_np).to(self.device)
            del train_np, val_np # Libère la mémoire CPU NumPy immédiatement
            """
            print(f"🚀 Dataset Train: {len(self.train_data)/1e6:.2f}M tokens.")
            # print(f"🚀 Val Dataset chargé (RAM). Train: {len(self.val_data)/1e6:.2f}M tokens.")
        else:
            print(f"⚠️ Fichiers binaires introuvables dans {data_dir}")      

        # train_ds = TokenDataset(train_path, self.config['block_size'])
        # self.train_loader = DataLoader(
        #     train_ds, 
        #     batch_size=self.params['batch_size'],
        #     shuffle=True,           # Pour une meilleure convergence
        #     num_workers=1,          # C'est ici que la magie du parallélisme opère (CPU)
        #     pin_memory=False,       # Sur MPS (mémoire unifiée), pin_memory est souvent inutile
        #     prefetch_factor=2       # Chaque worker prépare 2 batches d'avance
        # )
        # print(f"🚀 Train Loader Activé.")

    def _init_weights(self, module):
        """
        Règle d'initialisation standard pour les architectures GPT.
        """
        if isinstance(module, torch.nn.Linear):
            # Initialisation normale avec std=0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            # Les LayerNorm commencent avec un gain de 1 et un biais de 0
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
 
    def _load_processed_log(self):
        """Lit la liste des fichiers déjà entraînés pour ne pas les refaire"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return set(line.strip() for line in f)
        return set()
    
    def _load_history(self):
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                return json.load(f)
        return {'train_loss': [], 'val_loss': [], 'steps': [], 'time_elapse': [], 'time_model':[],'time_batch': [], 'time_bacwd': [], 'time_optim': [], 'time_init': [], 'time_eval': []}

    def _save_history(self):
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f)

    def print_memory_stats(self):
        """
        Analyse précise des paramètres et de la consommation RAM (Modèle + AdamW).
        """
        # 1. Compte des paramètres
        n_params = sum(p.numel() for p in self.model.parameters())
        
        # 2. Calcul des poids du modèle (en Float32 = 4 octets par paramètre)
        model_mem_mb = (n_params * 4) / 1024**2
        
        # 3. Calcul pour l'optimiseur AdamW
        # AdamW stocke DEUX états par paramètre (Moyenne et Variance) en Float32
        # + le gradient de chaque paramètre.
        # Total Optimiseur = 8 octets (états) + 4 octets (gradients) = 12 octets/paramètre
        optim_mem_mb = (n_params * 12) / 1024**2
        
        # 4. Estimation du fichier Memmap (si chargé partiellement ou via cache OS)
        # Ton fichier fait 1.5 Go, macOS va essayer d'en garder un maximum en RAM
        
        print(f"\n🚀 --- BILAN MÉMOIRE (M1) ---")
        print(f"• Paramètres totaux : {n_params:,}")
        print(f"• Poids du Modèle   : {model_mem_mb:.2f} Mo")
        print(f"• État Optimiseur   : {optim_mem_mb:.2f} Mo (AdamW)")
        print(f"• Total nécessaire  : {model_mem_mb + optim_mem_mb:.2f} Mo")
        print(f"------------------------------")
        
        if (model_mem_mb + optim_mem_mb) > 12000:
            print("⚠️ ATTENTION : La RAM est très sollicitée. Risque de swap.")
        else:
            print("✅ MÉMOIRE OK : Ton M1 gérera l'entraînement confortablement.")

    def _mark_file_as_done(self, file_path):
        """Ajoute un fichier à la liste des traités"""
        with open(self.log_file, 'a') as f:
            f.write(file_path + "\n")
        self.processed_files.add(file_path)

    def _load_checkpoint(self):
        """Charge les poids. Permet de changer les hyperparams d'entraînement (LR) mais garde les poids."""
        if os.path.exists(self.ckpt_path):
            print(f"📥 Chargement du checkpoint : {self.ckpt_path}")
            # Chargement sur CPU d'abord pour sécurité
            ckpt = torch.load(self.ckpt_path, map_location=self.device) # à verifier ...
            
            # 1. Chargement des poids du modèle
            self.model.load_state_dict(ckpt['model'])
            
            # 2. On NE charge PAS l'optimizer si on veut changer le learning rate manuellement
            # pour une continuité parfaite de l'optimizer, décommentez la ligne suivante :
            # On stocke l'état de l'optimiseur pour le charger PLUS TARD
            self._saved_optimizer_state = ckpt.get('optimizer')
            # self.optimizer.load_state_dict(ckpt['optimizer'])
            # récupératoin du nombre de step effectués
            self.total_steps_done = ckpt.get('total_steps_done', 0)
            # 4. SYNCHRONISATION DU SCHEDULER (Important !)
            # On avance le scheduler jusqu'au point actuel pour que le LR 
            # corresponde à la courbe de warmup/decay.
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                for _ in range(self.total_steps_done):
                    self.scheduler.step()
            print(f"✅ Modèle restauré. Steps faits: {self.total_steps_done}")
        else:
            print("✨ Aucun checkpoint trouvé, démarrage à zéro.")
    
    def _load_optimizer_state(self):
        if hasattr(self, '_saved_optimizer_state') and self._saved_optimizer_state:
            try:
                self.optimizer.load_state_dict(self._saved_optimizer_state)
                print("✅ État de l'optimiseur restauré.")
            except Exception as e:
                print(f"⚠️ Note: Impossible de restaurer l'optimiseur (normal si changement d'architecture) : {e}")
    
    def save_checkpoint(self, n_versions=5):
        # print(f"💾 Sauvegarde du checkpoint...")
        print(f" | 💾", end="")
        # Récupération des poids "propres" (sans le wrapper de compilation)
        model_to_save = self.model._orig_mod if self.is_compiled else self.model
        ckpt = {
            'model': model_to_save.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps_done': self.total_steps_done, # Crucial pour le Scheduler
            'vocab_size': len(self.tokenizer.vocab)
        }
        torch.save(ckpt, self.ckpt_path)
        # on sauvegarde le dernier model avec un horodatage sur le step
        version_path = self.ckpt_path.replace('.pth', f'_step_{self.total_steps_done}.pth')
        torch.save(ckpt, version_path)
        # on maintient uniquement les n derniers fichiers
        pattern = self.ckpt_path.replace('.pth', '_step_*.pth')
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        if len(files) > n_versions:
            for f in files[:-n_versions]:
                os.remove(f)
                # print(f"🗑️ Ancien checkpoint supprimé : {os.path.basename(f)}")
                print(f" | 🗑️", end="")
        # on sauvergade le model d'inférence uniquement pour le genrate
        light_model_path = self.ckpt_path.replace('.pth', '_inference.pth')
        light_ckpt = {
            'model': self.model.state_dict(),
            'config': self.config,
            'vocab_size': len(self.tokenizer.vocab)
        }
        torch.save(light_ckpt, light_model_path)
        # print(f"✨ Modèle d'inférence prêt : {os.path.basename(light_model_path)}")
        print(f" | ✨")
        
    def get_batch(self, data_tensor):
        block_size = self.config['block_size']
        batch_size = self.params['batch_size'] # faut il verifier len(data_tensor) / batch_size
        ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
        x = torch.stack([data_tensor[i:i+block_size] for i in ix])
        y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])
        return x.to(self.device), y.to(self.device)

    def get_batch_bin(self, split='train'):
        """
        Récupère un lot de données aléatoire depuis le fichier binaire (memmap).
        N'est plus utilisé que pour le l'eval. 
        """
        data = self.train_data if split == 'train' else self.val_data
        # 1. Générer tous les indices d'un coup
        ix = np.random.randint(0, len(data) - self.config['block_size'], (self.params['batch_size'],))
        
        # 2. Utiliser la magie de NumPy pour extraire les blocs sans boucle Python lente
        # On crée une grille d'indices // à voir pour le créer dans l'init ?
        offsets = np.arange(self.config['block_size'])
        indices = ix[:, None] + offsets  # Forme (batch_size, block_size)
        
        x_np = data[indices]
        y_np = data[indices + 1]

        # Conversion directe en Tenseur sur le device
        # Note: On passe en .long() car la CrossEntropy ne prend pas le uint16
        x = torch.from_numpy(x_np).to(self.device).long()
        y = torch.from_numpy(y_np).to(self.device).long()

        del x_np, y_np
        # gc.collect()        
                
        return x, y
    
    def get_batch_bin_tensor(self, split='train'):
        data = self.train_data if split == 'train' else self.val_data
        
        # Générer les indices sur CPU (plus rapide pour l'aléatoire simple)
        ix = torch.randint(0, data.size(0) - self.config['block_size'], (self.params['batch_size'],))
        
        # Extraire les séquences (Slicing sur GPU)
        # On utilise une liste de compréhension car le slicing indexé 
        # est parfois plus stable sur MPS que les grilles d'indices complexes
        x = torch.stack([data[i:i+self.config['block_size']] for i in ix])
        y = torch.stack([data[i+1:i+1+self.config['block_size']] for i in ix])
        
        # On convertit en long() au dernier moment (requis pour la loss)
        return x.long(), y.long()

    def get_batch_parallel(self):
        try:
            x, y = next(self.train_iter)
        except StopIteration:
            # Si on arrive au bout du dataset, on recommence
            self.train_iter = iter(self.train_loader)
            x, y = next(self.train_iter)
        
        # Seul le transfert final vers MPS est bloquant
        return x.to(self.device), y.to(self.device)

    def get_lr(self,it):
        # Utilisation des paramètres passés à l'init
        warmup = self.params.get('warmup_iters', 500)
        max_iters = self.params.get('lr_decay_iters', 50000)
        lr_max = self.params['learning_rate']
        lr_min = self.params.get('min_lr', lr_max * 0.1)

        # 1) Phase de warmup
        if it < warmup:
            return lr_max * it / warmup
        # 2) Phase de plateau bas
        if it > max_iters:
            return lr_min
        # 3) Phase de Cosine Decay
        decay_ratio = (it - warmup) / (max_iters - warmup)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return lr_min + coeff * (lr_max - lr_min)
    
    def train_file(self, file_path):
        """Entraîne sur UN seul fichier"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"❌ Erreur lecture {file_path}, ignoré.")
            return

        if len(text) < self.config['block_size'] + 10:
            return

        # Tokenization
        tokens = self.tokenizer.encode(text)
        data_tensor = torch.tensor(tokens, dtype=torch.long, device= self.device)
        # --- SPLIT TRAIN / VAL (90% / 10%) ---
        n = int(0.9 * len(data_tensor))
        train_data = data_tensor[:n]
        val_data = data_tensor[n:]
        
        # Calcul dynamique des itérations pour voir tout le fichier
        batch_size = self.params['batch_size']
        block_size = self.config['block_size']
        grad_accum = self.params.get('grad_accum_steps', 1)
        
        # Nombre total de tokens / (batch * block)
        n_batches = len(train_data) // (batch_size * block_size)
        if n_batches < 1: n_batches = 1
        
        self.model.train()
        total_loss = 0
        
        # Boucle d'entraînement sur ce fichier
        for i in range(n_batches):
            
            # On utilise le step global pour le calcul du cosinus
            current_lr = self.get_lr(self.total_steps_done)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            # Gradient Accumulation
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0
            
            for _ in range(grad_accum):
                X, Y = self.get_batch(train_data)
                _, loss = self.model(X, Y)
                loss = loss / grad_accum
                accum_loss += loss.item()
                loss.backward()
            
            # Empêche le modèle de diverger si un batch est "bizarre"
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)            
            self.optimizer.step()
            self.total_steps_done += 1
            total_loss += accum_loss

        avg_loss = total_loss / n_batches
        return avg_loss, train_data, val_data

    def train_buffer(self, file_paths):
        """
        Fusionne une liste de fichiers en un seul grand bloc et entraîne le modèle dessus.
        """
        # --- 1. Lecture et Fusion (Memory Efficient) ---
        content_list = []
        valid_files = [] # On garde trace des fichiers qui ont bien été lus
        
        # Séparateur pour aider le modèle à comprendre qu'on change de document
        # Si tu n'as pas de token spécial <|endoftext|>, \n\n suffit.
        separator = "\n\n" 
        start_time = time.time()

        for fname in file_paths:
            try:
                with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                    # On ignore les fichiers vides (< 100 char)
                    if len(text) > 100: 
                        content_list.append(text)
                        valid_files.append(fname)
            except Exception:
                print(f"⚠️ Erreur lecture : {fname}")
                continue

        if not content_list:
            return None, [], None, None

        # Fusion optimisée
        full_text = separator.join(content_list)
        print(f"🗂️ Fichiers lus. {(time.time()-start_time):.1f} s")
        start_time = time.time()
        
        # --- 2. Tokenization & Tensor ---
        # On garde le tenseur sur le CPU pour ne pas saturer la VRAM du M1
        tokens = self.tokenizer.encode(full_text)
        data_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device)
        
        # --- 3. Split Train/Val ---
        n = int(0.9 * len(data_tensor))
        train_data = data_tensor[:n]
        val_data = data_tensor[n:]
        print(f"𝌬 Encodage des fichiers [{len(train_data):d}], lancement des Batchs : {(time.time()-start_time):.1f} s")
        start_time = time.time()


        # --- 4. Configuration de l'entraînement ---
        batch_size = self.params['batch_size']
        block_size = self.config['block_size']
        grad_accum = self.params.get('grad_accum_steps', 4)

        # Combien de batches complets peut-on faire ?
        n_batches = len(train_data) // (batch_size * block_size)
        
        if n_batches < 1: 
            return None, valid_files, None, None

        self.model.train()
        total_loss = 0
        
        # --- 5. Boucle d'entraînement sur le Buffer ---
        for i in range(n_batches):
            
            # A. Learning Rate Dynamique
            lr = self.get_lr(self.total_steps_done)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # B. Zero Grad
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0
            
            # C. Accumulation de Gradient
            for _ in range(grad_accum):
                # C'est ici qu'on envoie les données sur le GPU (MPS)
                X, Y = self.get_batch(train_data)
                
                # Mixed Precision n'est pas nécessaire sur M1 (MPS gère bien le Float32)
                logits, loss = self.model(X, Y)
                
                loss = loss / grad_accum
                accum_loss += loss.item()
                loss.backward()
            
            # D. Clipping & Update
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            self.total_steps_done += 1
            total_loss += accum_loss

            # --- AJOUT : ÉVALUATION RÉGULIÈRE ---
            # On évalue toutes les 20 itérations (tu peux ajuster ce chiffre)
            if self.total_steps_done % 20 == 0:
                losses = self.estimate_loss(train_data, val_data)
                
                # Mise à jour de l'historique en temps réel
                self.history['train_loss'].append(losses['train'])
                self.history['val_loss'].append(losses['val'])
                self.history['steps'].append(self.total_steps_done)
                
                # Sauvegarde automatique pour ne rien perdre
                self.save_checkpoint()
                self._save_history()
                
                print(f"\n📈 Step {self.total_steps_done} | Loss Val: {losses['val']:.4f} | LR: {lr:.2e} | {(time.time()-start_time):.1f}")

        avg_loss = total_loss / n_batches
        
        return avg_loss, valid_files, train_data, val_data

    def train_bin(self):
        """
        Entraînement continu sur fichier binaire.
        - Sauvegarde et évalue tous les 'eval_interval' steps.
        - Gère la reprise parfaite de l'optimizer.
        """
        self.model.train()
        
        # --- PARAMÈTRES ---
        # Fréquence des points sur le graphique (ex: toutes les 500 mises à jour)
        eval_interval = self.params.get('eval_interval', 1000) 
        eval_iters = self.params.get('eval_iters', 30) 
        save_interval = self.params.get('save_interval', 1000)
        
        batch_size = self.params['batch_size']
        grad_accum = self.params.get('grad_accum_steps', 1)
        
        # Combien de steps on vise au total (ex: 100 000)
        max_steps = self.params.get('lr_decay_iters', 100_000)

        print(f"📦 Démarrage Training sur 3Go (Mode Continu)")
        print(f"   ▶️ Reprise au step : {self.total_steps_done}")
        print(f"   📉 Evaluation tous les : {eval_interval} steps")
        print(f"   💾 Sauvegarde tous les : {save_interval} steps")

        self.log_memory_status(v=True)

        # --- BOUCLE INFINIE (Pilotée par les steps) ---
        # On ne boucle pas sur 'epoch', on boucle jusqu'à 'max_steps'
        start_time = time.time()
        t_bacwd=0
        t_batch=0
        t_optim=0
        t_model=0
        t_init=0
        t_eval=0
        previous_step = self.total_steps_done
        while self.total_steps_done < max_steps:
            start_step = time.time()
            # 1. Update LR (Essentiel pour la reprise)
            lr = self.get_lr(self.total_steps_done)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # 2. Forward / Backward avec Accumulation
            self.optimizer.zero_grad(set_to_none=True)
            t_init += time.time() - start_step
            accum_loss = 0
            
            for _ in range(grad_accum):
                # On pioche au hasard dans les 3Go (Global Shuffling naturel)
                t0 = time.time()
                # X, Y = self.get_batch_parallel()
                # X, Y = self.get_batch_bin('train')
                X, Y = self.train_queue.next()
                # print(f"DEBUG DATA: {self.tokenizer.decode(X[0][:50].tolist())}")        
                # torch.mps.synchronize()
                t1 = time.time() # Temps chargement données
                t_batch += t1 - t0
                # Mixed precision auto gérée par PyTorch si dispo, sinon float32
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    logits, loss = self.model(X, Y)
                loss = loss / grad_accum
                t2 = time.time() # Temps forward pass
                t_model += t2 - t1
                accum_loss += loss.item()
                self.scaler.scale(loss).backward()
                t3 = time.time() # Temps backward pass (souvent le plus long)
                t_bacwd += t3 - t2
            
            # 3. Step Optimizer
            """ ancienne version float32
            t4_start = time.time()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            """
            # On redescend les gradients avant le clipping
            t4_start = time.time()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # Step final
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # fin de scaler""""            
            t4 = time.time() # Temps mise à jour optimizer
            # print(f"Data: {t1-t0:.3f}s | Fwd: {t2-t1:.3f}s | Bwd: {t3-t2:.3f}s | Opt: {t4-t3:.3f}s | Step: {t4-start_step:.3f}s")
            t_optim += t4 - t4_start
            # On incrémente le compteur GLOBAL
            self.total_steps_done += 1
            # --- EVALUATION & PLOT (Fréquent) ---
            t5 = time.time()
            if self.total_steps_done % eval_interval == 0:
                elapsed = time.time() - start_time
                # Calcul du Loss Val (Le vrai juge)
                losses, status, pression, mem_rss = self.estimate_loss_bin(eval_iters=eval_iters)
                swap = psutil.swap_memory().used / (1_048_576)
                if self.total_steps_done % save_interval == 0:
                    print(f"step {self.total_steps_done}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e} ({elapsed:.2f}s) | {status} Po: {pression:.1f}% SW: {swap:.0f}Mo MPS: {torch.mps.current_allocated_memory() / 1_048_576:.0f}Mo", end="")
                else:
                    print(f"step {self.total_steps_done}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e} ({elapsed:.2f}s) | {status} Po: {pression:.1f}% SW: {swap:.0f}Mo MPS: {torch.mps.current_allocated_memory() / 1_048_576:.0f}")
                # Mise à jour historique
                nb_step = self.total_steps_done - previous_step
                previous_step = self.total_steps_done
                self.history['train_loss'].append(losses['train'])
                self.history['val_loss'].append(losses['val'])
                self.history['steps'].append(self.total_steps_done)
                self.history['time_elapse'].append(elapsed)
                self.history['time_model'].append(t_model/nb_step)
                self.history['time_batch'].append(t_batch/nb_step)
                self.history['time_bacwd'].append(t_bacwd/nb_step)
                self.history['time_optim'].append(t_optim/nb_step)
                self.history['time_init'].append(t_init/nb_step)
                self.history['time_eval'].append(t_eval/nb_step)
                self._save_history() # Sauvegarde JSON pour le graphique
                start_time = time.time() # Reset timer
                # torch.mps.empty_cache() # Libère la RAM : plus utile car dans estimate_loss_bin on gère bien
                # Reset des timers internes
                t_bacwd=0
                t_batch=0
                t_optim=0
                t_model=0
                t_init=0
                t_eval=0
                # print(f"Mémoire allouée MPS : {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")
            # --- SAUVEGARDE CHECKPOINT (Sécurité) ---
            if self.total_steps_done % save_interval == 0:
                self.save_checkpoint(n_versions=self.params.get('n_version',5))
                # print(f"💾 Checkpoint sauvegardé au step {self.total_steps_done}")
            t_eval += time.time() - t5

        print("✅ Fin de l'entraînement (Max Steps atteint).")

    def run_training_loop(self, max_files_session=50):
        """
        Lance l'entraînement.
        max_files_session: nombre de fichiers à traiter avant d'arrêter le script (pour procéder par lots)
        """
        # 1. Lister tous les fichiers
        all_files = glob.glob(os.path.join(self.data_root, "*", "*"))
        all_files = [f for f in all_files if os.path.isfile(f)]
        
        # 2. Filtrer ceux déjà faits
        remaining_files = [f for f in all_files if f not in self.processed_files]
        
        print(f"📊 Statut : {len(self.processed_files)} terminés, {len(remaining_files)} restants.")
        
        if not remaining_files:
            print("🎉 Tous les fichiers ont été traités !")
            return

        # 3. Mélanger pour éviter le biais (ex: ne pas apprendre que les articles commençant par 'A')
        random.shuffle(remaining_files)
        
        # 4. Sélectionner le lot pour cette session
        files_to_do = remaining_files[:max_files_session]
        print(f"▶️ Démarrage de la session sur {len(files_to_do)} fichiers...\n")

        start_time = time.time()
        
        for idx, fname in enumerate(files_to_do):
            # Affichage "pretty"
            short_name = f"{os.path.basename(os.path.dirname(fname))}/{os.path.basename(fname)}"
            
            loss, train_data, val_data = self.train_file(fname)
            
            if loss is not None:
                # Marquer comme fait
                self._mark_file_as_done(fname)
                
                elapsed = time.time() - start_time
                print(f"[{idx+1}/{len(files_to_do)}] Loss: {loss:.4f} | Fichier: {short_name}")
                
                # --- EVALUATION ---
                losses = self.estimate_loss(train_data, val_data)

                # Enregistrement
                self.history['train_loss'].append(losses['train'])
                self.history['val_loss'].append(losses['val'])
                self.history['steps'].append(len(self.processed_files))

                # Sauvegarde régulière (à chaque fichier pour sécurité maximale sur M1)
                self.save_checkpoint()
                self._save_history()

        print(f"\n✅ Session terminée. Vous pouvez relancer le script pour la suite.")

    def run_training_loop_merge(self, files_per_buffer=50, max_buffers=1000):
        """
        Nouvelle boucle optimisée par fusion.
        files_per_buffer : Nombre de fichiers à coller ensemble (ex: 50).
        max_buffers : Nombre de lots à traiter dans cette session.
        """
        # 1. Lister tous les fichiers
        all_files = glob.glob(os.path.join(self.data_root, "*", "*")) # Adapté à ta structure
        all_files = [f for f in all_files if os.path.isfile(f)]
        
        # 2. Filtrer ceux déjà faits
        remaining_files = [f for f in all_files if f not in self.processed_files]
        
        print(f"📊 Statut Global : {len(self.processed_files)} terminés, {len(remaining_files)} restants.")
        
        if not remaining_files:
            print("🎉 Tous les fichiers ont été traités !")
            return

        # 3. Mélanger pour éviter le biais
        random.shuffle(remaining_files)
        
        # 4. Création des 'Chunks' (Les lots de 50 fichiers)
        chunks = [remaining_files[i:i + files_per_buffer] for i in range(0, len(remaining_files), files_per_buffer)]
        
        # On ne prend que le nombre demandé pour cette session
        chunks_to_do = chunks[:max_buffers]
        
        print(f"▶️ Démarrage : {len(chunks_to_do)} buffers (lots) à traiter.\n")
        start_time = time.time()
        
        for idx, chunk_files in enumerate(chunks_to_do):
            
            print(f"📦 Buffer {idx+1}/{len(chunks_to_do)} ({len(chunk_files)} fichiers)... ", end="", flush=True)
            
            # Appel du Worker
            loss, done_files, t_data, v_data = self.train_buffer(chunk_files)
            
            if loss is not None:
                # On marque TOUS les fichiers du lot comme faits
                for f in done_files:
                    self._mark_file_as_done(f)
                
                # elapsed = time.time() - start_time
                # print(f"OK | Loss: {loss:.4f} | Steps: {self.total_steps_done}")
                
                # # --- EVALUATION & SAUVEGARDE ---
                # # On sauvegarde à la fin de chaque buffer car ça représente beaucoup de travail (~5-10 min)
                # losses = self.estimate_loss(t_data, v_data)

                # # Enregistrement historique
                # self.history['train_loss'].append(losses['train'])
                # self.history['val_loss'].append(losses['val'])
                # self.history['steps'].append(self.total_steps_done) # On log les steps, pas le nombre de fichiers

                # self.save_checkpoint()
                # self._save_history()
            else:
                print("SKIPPED (Données insuffisantes)")

        print(f"\n✅ Session terminée. {len(chunks_to_do)} buffers traités.")

    @torch.no_grad()
    def estimate_loss(self, train_data, val_data):
        """ Évalue le loss sur les deux splits du fichier courant """
        self.model.eval()
        out = {}
        # On évalue sur un nombre fixe de petits batchs pour aller vite
        eval_iters = 20 
        for split, data in [('train', train_data), ('val', val_data)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.get_batch(data)
                if X is None: continue
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out
    
    def estimate_loss_bin(self, eval_iters=30):
        """
        Fonction helper pour estimer le loss sans dropout (mode eval).
        eval_iters: nombre de batchs pour moyenner et avoir un score stable.
        """
        out = {}
        self.model.eval() # Désactive Dropout
        # 1. Purge préventive avant l'effort d'évaluation
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # 2. Utilisation de inference_mode (plus rapide que no_grad sur Metal)
        with torch.autocast(device_type='mps', dtype=torch.float16):
            with torch.inference_mode():
            # with torch.no_grad():
                for split in ['train', 'val']:
                    losses = torch.zeros(eval_iters)
                    for k in range(eval_iters):
                        X, Y = self.get_batch_bin(split)
                        _, loss = self.model(X, Y)
                        losses[k] = loss.item()
                    out[split] = losses.mean().item()
        
        self.model.train() # Réactive Dropout
            
        # 3. Purge préventive avant l'effort d'évaluation
        if torch.backends.mps.is_available():
            gc.collect()           # Libère la RAM CPU (NumPy/Tensors CPU)
            torch.mps.empty_cache()

        status, pression, mem_rss = self.log_memory_status()

        return out, status, pression, mem_rss
    
    def log_memory_status(self, v=False):
        # RAM système
        vm = psutil.virtual_memory()
        # Pression mémoire (en %) - c'est l'indicateur le plus important sur macOS
        # Sur Mac, psutil ne donne pas la "pression" exacte d'Apple, mais on l'estime ainsi :
        pression = vm.percent 
        
        # Mémoire spécifique à ton process Python
        process = psutil.Process(os.getpid())
        mem_rss = process.memory_info().rss / (1024**2) # En Mo
        
        status = "🟢" if pression < 70 else "🟡" if pression < 85 else "🔴"
        if (v):
            print(f"{status} RAM Système: {pression}% | Process: {mem_rss:.0f}Mo | Swap: {psutil.swap_memory().used / (1024**2):.0f}Mo")
        return status, pression, mem_rss

class BackgroundGenerator:
    def __init__(self, generator_func, max_prefetch=1):
        self.queue = queue.Queue(maxsize=max_prefetch)
        self.generator_func = generator_func
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_event.is_set():
            # Prépare le batch (CPU + NumPy)
            batch = self.generator_func()
            # Attend que la queue ait de la place pour le déposer
            self.queue.put(batch)

    def next(self):
        return self.queue.get()

class TokenDataset(Dataset):
    def __init__(self, data_path, block_size):
        # On utilise memmap pour ne pas exploser la RAM
        # Le système d'exploitation gérera le cache intelligemment
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        # Cette partie s'exécutera sur le CPU (dans les workers)
        chunk = self.data[idx : idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

class GenerateGPT:
    def __init__(self, tokinizer, ckpt_path):
        self.tokenizer = tokinizer
        self.ckpt_path = ckpt_path
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        
    def load_for_inference(self):
        if not os.path.exists(self.ckpt_path):
            print("❌ Aucun modèle trouvé !")
            return None, None

        print(f"Loading {self.ckpt_path} on {self.device}...")
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        
        self.config = checkpoint['config']
        
        # On recrée le modèle avec la config exacte de l'entraînement
        self.model = GPTLanguageModel(vocab_size=len(self.tokenizer.vocab), **self.config)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval() # TRES IMPORTANT : désactive le Dropout
        
        # return model, config

    def generate_text(self, prompt, max_new_tokens=100, temperature=0.8, top_k=40, rep_penalty=1.2):
        # 1. Encodage
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # 2. Génération
        # Note: J'adapte légèrement la méthode generate pour inclure la température
        # Si votre méthode generate dans la classe Model ne gère pas la température,
        # elle fera une génération standard.
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context si trop long
                idx_cond = input_tensor[:, -self.model.block_size:]
                
                # Forward
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature # Applique la température

                # On parcourt les tokens déjà générés pour pénaliser les logits et réduire les répétitions
                for token_id in set(input_tensor[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= rep_penalty
                    else:
                        logits[0, token_id] *= rep_penalty
            
                # ajout du top_k qui ne garde que les mots les plus probable : mask
                if top_k is not None:
                    # On ne garde que les top_k meilleures valeurs
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # On remplace tout ce qui est plus petit que la plus petite valeur du top_k par -inf
                    logits[logits < v[:, [-1]]] = -float('Inf')
            
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Sampling
                idx_next = torch.multinomial(probs, num_samples=1)
                input_tensor = torch.cat((input_tensor, idx_next), dim=1)

                next_token_id = idx_next.item()
                print(self.tokenizer.decode([next_token_id]), end='', flush=True)

        # 3. Décodage
        output_text = self.tokenizer.decode(input_tensor[0].tolist(), self.tokenizer.vocab)
        return output_text

class TrainingVisualizer:
    def __init__(self, history_path='history.json'):
        self.history_path = history_path

    def load_data(self):
        if not os.path.exists(self.history_path):
            print(f"❌ Erreur : Le fichier {self.history_path} n'existe pas encore.")
            return None
        
        with open(self.history_path, 'r') as f:
            return json.load(f)

    def plot_metrics(self, window_size=5, save_path=None, lim_y = None):
        """
        Trace les courbes de loss.
        window_size : taille de la fenêtre pour le lissage (moyenne mobile).
        save_path : chemin pour enregistrer l'image (ex: 'progress.png').
        """
        data = self.load_data()
        if not data: return

        train_loss = data.get('train_loss', [])
        val_loss = data.get('val_loss', [])
#        steps = range(len(train_loss))
        steps = data.get('steps', [])

        if len(train_loss) == 0:
            print("Aucune donnée à tracer.")
            return

        plt.figure(figsize=(12, 6))

        last_val = val_loss[-1]
        if lim_y == None:
            pass
        elif (lim_y[0] == -1):
            plt.ylim( last_val-0.5, last_val+0.5)
        else:
            plt.ylim( lim_y[0], lim_y[1])
        
        # 1. Tracé des données brutes (en pointillés légers)
        plt.plot(steps, train_loss, color='blue', alpha=0.4, label='Train (brut)')
        plt.plot(steps, val_loss, color='red', alpha=0.4, label='Val (brut)')

        # 2. Calcul et tracé des moyennes mobiles (lissage)
        if len(train_loss) >= window_size:
            # 1. Calcul des moyennes mobiles (inchangé)
            train_smooth = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
            val_smooth = np.convolve(val_loss, np.ones(window_size)/window_size, mode='valid')
            
            # 2. Récupération des steps réels correspondants
            # On prend la liste des steps à partir de l'indice (window_size - 1)
            steps_array = np.array(steps)
            smooth_steps = steps_array[window_size - 1:]
            
            # 3. Tracé
            plt.plot(smooth_steps, train_smooth, label=f'Train (smooth {window_size})', alpha=0.8)
            plt.plot(smooth_steps, val_smooth, label=f'Val (smooth {window_size})', alpha=0.8)
        """ Avant
        if len(train_loss) >= window_size:
            train_smooth = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
            val_smooth = np.convolve(val_loss, np.ones(window_size)/window_size, mode='valid')
            
            # Ajustement des steps pour le décalage de la moyenne mobile
            smooth_steps = range(window_size - 1, len(train_loss))
            
            plt.plot(smooth_steps, train_smooth, color='blue', linewidth=2, label=f'Train (lissé {window_size}nd)')
            plt.plot(smooth_steps, val_smooth, color='red', linewidth=2, label=f'Val (lissé {window_size}nd)')
        """

        # Configuration du graphique
        plt.title('Évolution du Loss pendant l\'apprentissage (Wiki FR)', fontsize=14)
        plt.xlabel('Nombre de steps', fontsize=12)
        plt.ylabel('Cross Entropy Loss', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Ajout d'une zone de texte avec les dernières stats
        last_train = train_loss[-1]
        last_val = val_loss[-1]
        stats_text = f"Dernier Train Loss: {last_train:.4f}\nDernier Val Loss: {last_val:.4f}"
        plt.annotate(stats_text, xy=(0.05, 0.05), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        if save_path:
            plt.savefig(save_path)
            print(f"✅ Graphique enregistré dans : {save_path}")
        
        plt.show()
        plt.close()
