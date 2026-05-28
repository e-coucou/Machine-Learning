import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os, psutil
import glob
import math
import json
import threading
import queue, gc
# -----------------------------------------------------------------------------
VERSION = "v6.4.0"
VERSION_INFO = "Cette version 6.2 inclus les calculs de loss avec les datasets val & train au ratio des datasets et devrait donc se rapprocher du train_raw avec l'exception du dropout"
# -----------------------------------------------------------------------------
# 1. BLOCS DE BASE DU MODÈLE (Architecture GPT "Decoder-Only")
# -----------------------------------------------------------------------------
class MultiHeadAttention(nn.Module): # version light
    """ Causal Self-Attention. C'est le coeur du mécanisme GPT. 
        Utilisation de la fonction intégrée dans Torch v2 avec le composant metal. """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        
        # Projection clés, requêtes, valeurs
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        
        self.dropout_val = dropout # pour le passer en argument à la SPDA
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        
        # Calcul Q, K, V en une seule opération (optimisé M1)
        qkv = self.qkv(x)  # (B, T, 3*n_embd)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_size)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calcul des scores d'attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = None,
            dropout_p = self.dropout_val if self.training else 0.0,
            is_causal = True
        )
        # Recomposition
        #1 out = out.permute(0, 2, 1, 3).contiguous().reshape(B, T, C)
        #2 out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = out.transpose( 1, 2).reshape(B, T, C)
        
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

        # on initialise ici pour éviter de le refaire à chaque cycle de forward
        self.register_buffer('pos_idx', torch.arange(block_size))
        
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
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)   # on fait l'init ici de tous les poids
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        
        # Embeddings
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        # On utilise le buffer de l'init. plus besoin de faire le arange
        #pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        pos_emb = self.position_embedding_table(self.pos_idx[:T])
        x = tok_emb + pos_emb
        
        # Passage dans les blocs
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x) # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            # on utilise le reshape plus sure ...
            #logits = logits.view(B*T, C)
            #targets = targets.view(B*T)
            logits = logits.reshape(-1, C)
            targets = targets.reshape(-1)
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
# 2. FOURNISSEUR DE DONNÉES DÉTERMINISTE POUR CONTINUOUS TRAINING
# -----------------------------------------------------------------------------

class DeterministicProvider:
    def __init__(self, data, batch_size, block_size, device, start_step, grad_accum_steps=4, seed=1965):
        self.data = data # Le memmap NumPy
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.step = start_step #* grad_accum_steps
        self.seed = seed
        
        # 1. Création de la route fixe (Couverture 100%)
        # On définit les points de départ tous les 'block_size'
        self.indices = np.arange(0, len(self.data) - self.block_size, self.block_size)
        self.total_indices = len(self.indices)
        # Calcul de l'epoch actuelle pour caler le shuffle au démarrage
        self.last_epoch = (self.step * self.batch_size) // self.total_indices        
        
        # 2. Shuffle déterministe avec le Seed et +1 à chaque époque 
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
            
#            x_np = self.data[grid_indices]
#            y_np = self.data[grid_indices + 1]
            # on fait désormais une copie pour voir si moins de fuite mémoire vs chrono
            x_np = np.array(self.data[grid_indices], dtype=np.int64)
            y_np = np.array(self.data[grid_indices + 1], dtype=np.int64)


            # Conversion en Tensor et envoi vers le GPU
            # On le fait ici pour que le BackgroundGenerator livre un produit fini
#            x = torch.from_numpy(x_np).to(self.device).long()
#            y = torch.from_numpy(y_np).to(self.device).long()
            # idem on transfert dans le GPU
            x = torch.as_tensor(x_np, device = self.device)
            y = torch.as_tensor(y_np, device = self.device)

            del x_np,y_np
            
            yield x, y
            self.step += 1

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

# -----------------------------------------------------------------------------
# 3.  CONTINUOUS TRAINING - version finale avec chargement plusieurs datasets
# -----------------------------------------------------------------------------

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
        self.cult_data = train_params.get('cult_data', False)
        self.litt_data = train_params.get('litt_data', False)
        self.cult_ratio = train_params.get('cult_ratio', 0.0)
        self.litt_ratio = train_params.get('litt_ratio', 0.0)
        self.wiki_ratio = 1. - self.cult_ratio - self.litt_ratio
        self.ema_decay = train_params.get('ema_decay', 0) # 0 pour désactiver
        # Gestion du Device / préférence sur MPS /
#        self.dtype = torch.bfloat16
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"🚀 Device: {self.device}")
        # Instanciation du modèle
        self.model = model_class(vocab_size=len(tokenizer.vocab), **config)
        # On applique l'initialisation personnalisée sur le modèle neuf
#v6.4        self.model.apply(self._init_weights)
        self.model.to(self.device, dtype=torch.float32) #, dtype=torch.float16 if self.device.type == 'mps' else torch.float32)
        # Check >Dtype
        print(f"Dtype du modèle : {next(self.model.parameters()).dtype}")
        # Initialisation des steps totaux : seront mis à jour par le load_checkpoint
        self.total_steps_done = 0
        self.total_step_cult = 0
        self.total_step_wiki = 0
        self.total_step_litt = 0
        # Chargement du checkpoint si disponible
        self._load_checkpoint()
        # Initialisation de l'EMA si activée
        self._init_EMA()
        # Compilation optionnelle avec torch.compile (PyTorch 2.0+)
        self._init_compile()
        # Optimiseur (On l'initialise ici, mais son état peut être écrasé si on charge un checkpoint)
        self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), 
                    lr=train_params['learning_rate'],
                    weight_decay=train_params.get('weight_decay', 0.1),
                )
        # Chargement de l'état de l'optimiseur si disponible
        # Restauration de l'état de l'Optimiseur
        if hasattr(self, '_saved_optimizer_state') and self._saved_optimizer_state:
            self.optimizer.load_state_dict(self._saved_optimizer_state)
            del self._saved_optimizer_state # Libère la mémoire
            print("✅ État de l'optimiseur restauré.")
        # Liste des fichiers déjà traités (obsolète, maintenant on charge un fichier de token unique) & l'historique
        # self.processed_files = self._load_processed_log()
        self.history = self._load_history()    
        self._setup_data(data_dir)
        # Préparation des providers et de l'itérateurs
        #   On crée l'itérateur persistant ici
        #   Le BackgroundGenerator utilise l'itérateur existant
        self._init_providers()
        self.train_iterator = self._training_mixer()
        # Note : on utilise 'next(self.train_iterator)' SANS le 'iter()' sinon reset à chaque iter
        self.train_queue = BackgroundGenerator(
            lambda: next(self.train_iterator), 
            max_prefetch=2 # peut monter à 5 pour plus de fluidité ... mais on sature le bus ram unifié et ralenti le GPU !
        )
        # Initialisation (une seule fois au début de la classe) passage en float16 sur MPS
        self.scaler = torch.amp.GradScaler(self.device, enabled=True)
        # Initialisation du monitor de log
        self.monitor = Monitor()
        print(f"🚀 Init terminé.")

    # Création du Mixer optimisé
    def _training_mixer(self):
        """
        Mixeur intelligent qui s'adapte aux sources disponibles.
        Ratios cibles : Litté 50% | CulturaX 30% | Wiki 20%
        """
        # 1. Préparation des itérateurs (seulement si le provider existe)
        sources = {}
        if hasattr(self, 'litt_provider') and self.litt_provider:
            sources['litt'] = iter(self.litt_provider)
        if hasattr(self, 'culturaX_provider') and self.culturaX_provider:
            sources['cult'] = iter(self.culturaX_provider)
        sources['wiki'] = iter(self.train_provider)

        if not sources:
            raise RuntimeError("❌ Erreur : Aucun dataset n'est disponible pour le mixer !")

        print(f"🔄 Mixer activé avec les sources : {list(sources.keys())}")

        while True:
            r = np.random.random()
            # --- Logique de cascade avec repli (Fallback) ---            
            if r < self.litt_ratio and 'litt' in sources:
                self.total_step_litt += 1
                yield next(sources['litt'])
            
            elif r < (self.litt_ratio + self.cult_ratio) and 'cult' in sources:
                self.total_step_cult += 1
                yield next(sources['cult'])
            
            elif 'wiki' in sources:
                self.total_step_wiki += 1
                yield next(sources['wiki'])
            
            # SÉCURITÉ : Si la source choisie par 'r' n'existe pas, 
            # on prend la première source disponible dans le dictionnaire
            else:
                fallback_key = list(sources.keys())[0]
                if fallback_key == 'litt': self.total_step_litt += 1
                if fallback_key == 'cult': self.total_step_cult += 1
                if fallback_key == 'wiki': self.total_step_wiki += 1
                yield next(sources[fallback_key])

    def _setup_data(self,data_dir):
        # --- CHARGEMENT DES DONNÉES BINAIRES (Nouveau) ---
        # On utilise memmap pour lire le fichier sur le disque sans charger la RAM
        train_path = os.path.join(data_dir, 'train_wiki.bin')
        val_path = os.path.join(data_dir, 'val_wiki.bin')
        train_cult_path = os.path.join(data_dir, 'train_culturax.bin')
        val_cult_path = os.path.join(data_dir, 'val_culturax.bin')
        train_litt_path = os.path.join(data_dir, 'train_litteraire.bin')
        val_litt_path = os.path.join(data_dir, 'val_litteraire.bin')
        
        if os.path.exists(train_path):
            self.train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
            print(f"🚀 Dataset Train: {len(self.train_data)/1e6:.2f}M tokens.")
            self.val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
            print(f"🚀 Dataset Val: {len(self.val_data)/1e6:.2f}M tokens.")
        else:
            print(f"⚠️ Fichiers binaires introuvables dans {data_dir}")      

        if self.cult_data and os.path.exists(train_cult_path):
            self.train_data_cult = np.memmap(train_cult_path, dtype=np.uint16, mode='r')
            print(f"🚀 Dataset CulturaX: {len(self.train_data_cult)/1e6:.2f}M tokens.")
            self.val_cult = np.memmap(val_cult_path, dtype=np.uint16, mode='r')
            print(f"🚀 Dataset Val: {len(self.val_cult)/1e6:.2f}M tokens.")
        else:
            self.train_data_cult = None
            print("ℹ️ Pas de Dataset CulturaX.")    

        if self.litt_data and os.path.exists(train_litt_path):
            self.train_data_litt = np.memmap(train_litt_path, dtype=np.uint16, mode='r')
            print(f"🚀 Dataset Litteraire: {len(self.train_data_litt)/1e6:.2f}M tokens.")
            self.val_litt = np.memmap(val_litt_path, dtype=np.uint16, mode='r')
            print(f"🚀 Dataset Val: {len(self.val_litt)/1e6:.2f}M tokens.")
        else:
            self.train_data_litt = None
            print(f"ℹ️ Pas de Dataset Littéraire.")

    def _init_providers(self):
        self.train_provider = DeterministicProvider(
            data=self.train_data,
            batch_size=self.params['batch_size'],
            block_size=self.config['block_size'],
            device = self.device,
            start_step=self.total_step_wiki,
            grad_accum_steps=self.params.get('grad_accum_steps', 4),
            seed=1965 # Le Salt fixe
        )
        # On ajoute ici un fichier train supplémentaire pour CulturaX
        if  self.train_data_cult is not None:
            self.culturaX_provider = DeterministicProvider(
                data=self.train_data_cult,
                batch_size=self.params['batch_size'],
                block_size=self.config['block_size'],
                device = self.device,
                start_step=self.total_step_cult,
                grad_accum_steps=self.params.get('grad_accum_steps', 4),
                seed=2013 # Le Salt fixe mais différent
                )
        else: 
            self.culturaX_provider = None
        # On ajoute ici un fichier train supplémentaire pour Littearature
        if  self.train_data_litt is not None:
            self.litt_provider = DeterministicProvider(
                data=self.train_data_litt,
                batch_size=self.params['batch_size'],
                block_size=self.config['block_size'],
                device = self.device,
                start_step=self.total_step_litt,
                grad_accum_steps=self.params.get('grad_accum_steps', 4),
                seed=1990 # Le Salt fixe mais différent
                )
        else: 
            self.litt_provider = None


    def _init_EMA(self):
        self.ema_model = None
        if self.ema_decay > 0:
            import copy
            # On copie le modèle APRES le load_checkpoint pour avoir les poids restaurés
            self.ema_model = copy.deepcopy(self.model)
            self.ema_model.eval()
            # On gèle les paramètres
            for p in self.ema_model.parameters():
                p.requires_grad = False
            print(f"🚀 EMA activé (decay: {self.ema_decay})")
            if hasattr(self, '_saved_ema_state'):
                self.ema_model.load_state_dict(self._saved_ema_state)
                del self._saved_ema_state
                print("✅ Poids EMA restaurés depuis le fichier.")

    def _init_compile(self):
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
        return {'train_loss': [],'train_loss_wiki': [],'train_loss_cult': [],'train_loss_lit': [], 'val_loss': [], 'val_loss_wiki': [], 'val_loss_cult': [], 'val_loss_litt': [], 'steps': [], 'time_elapse': [], 'time_model':[],'time_batch': [], 'time_bacwd': [], 'time_optim': [], 'time_init': [], 'time_eval': []}

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
            print("✅ MÉMOIRE OK : M1 gérera l'entraînement confortablement.")

    def _load_checkpoint(self):
        """Charge les poids. Permet de changer les hyperparams d'entraînement (LR) mais garde les poids."""
        if os.path.exists(self.ckpt_path):
            print(f"📥 Chargement du checkpoint : {self.ckpt_path}")
            batch_size = self.params.get('batch_size',32)
            # Chargement sur CPU d'abord pour sécurité
            ckpt = torch.load(self.ckpt_path, map_location=self.device) # à verifier ...
            
            # 1. Chargement des poids du modèle et de l'EMA si existant
            # SUite à la modif du model (suppression du mask triangulaire et utilisation SPDA)
            #self.model.load_state_dict(ckpt['model'])
            self.model.load_state_dict(ckpt['model'], strict=False)
            if 'ema_model' in ckpt:
                self._saved_ema_state = ckpt['ema_model']

            # 2. On NE charge PAS l'optimizer si on veut changer le learning rate manuellement
            # pour une continuité parfaite de l'optimizer, décommentez la ligne suivante :
            # On stocke l'état de l'optimiseur pour le charger PLUS TARD
            self._saved_optimizer_state = ckpt.get('optimizer')
            # self.optimizer.load_state_dict(ckpt['optimizer'])
            # récupératoin du nombre de step effectués
            self.total_steps_done = ckpt.get('total_steps_done', 0)
            self.total_step_wiki = ckpt.get('total_step_wiki', 0) // batch_size
            self.total_step_cult = ckpt.get('total_step_cult', 0) // batch_size
            self.total_step_litt = ckpt.get('total_step_litt', 0) // batch_size
            print(f"📈 Reprise : Wiki à {self.total_step_wiki} | CulturaX à {self.total_step_cult} | {batch_size}")

            # 3. SYNCHRONISATION DU SCHEDULER (Important !)
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
    
    def _save_checkpoint(self, n_versions=5):
        # print(f"💾 Sauvegarde du checkpoint...")
        print(f" | 💾", end="")
        batch_size = self.params.get('batch_size', 32)
        # Récupération des poids "propres" (sans le wrapper de compilation)
        model_to_save = self.model._orig_mod if self.is_compiled else self.model
        ckpt = {
            'model': model_to_save.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps_done': self.total_steps_done, # Crucial pour le Scheduler
            'total_step_wiki': self.total_step_wiki * batch_size,
            'total_step_cult': self.total_step_cult * batch_size,
            'total_step_litt': self.total_step_litt * batch_size,
            'vocab_size': len(self.tokenizer.vocab),
            'params': self.params,
        }
        # On n'ajoute l'EMA que s'il existe
        if self.ema_model is not None:
            ckpt['ema_model'] = self.ema_model.state_dict()
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
            'vocab_size': len(self.tokenizer.vocab),
            'model_ema': self.ema_model.state_dict() if self.ema_model is not None else None
        }
        torch.save(light_ckpt, light_model_path)
        # print(f"✨ Modèle d'inférence prêt : {os.path.basename(light_model_path)}")
        print(f" | ✨")

    def get_batch_from_source(self, data):
        """
        Version optimisée de la fonction pour piocher dans n'importe quel memmap.
        Correction apporté pour éviter les fuites mémoire entre Numpy et Pytorch
        """
        # 1. Générer tous les indices d'un coup
        ix = np.random.randint(0, len(data) - self.config['block_size'], (self.params['batch_size'],))
        
        # 2. Grille d'indices (la "magie" NumPy)
        offsets = np.arange(self.config['block_size'])
        indices = ix[:, None] + offsets 
        # np.array fait une copy, on coupe la liaison avec le disque SSD
        x_np = np.array(data[indices], dtype=np.int64)
        y_np = np.array(data[indices + 1], dtype=np.int64)

        # Conversion en Tenseur (on reste en long pour la CrossEntropy)
        # Note: On passe en .long() car la CrossEntropy ne prend pas le uint16
#        x = torch.from_numpy(x_np).to(self.device).long()
#        y = torch.from_numpy(y_np).to(self.device).long()
#        x = torch.tensor(x_np, device = self.device)
#        y = torch.tensor(y_np, device = self.device)
        x = torch.as_tensor(x_np, device = self.device)
        y = torch.as_tensor(y_np, device = self.device)

        del x_np, y_np #

        return x, y

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

    def train_bin(self):
        """
        Entraînement continu sur fichier binaire.
        - Sauvegarde et évalue tous les 'eval_interval' steps.
        - Gère la reprise parfaite de l'optimizer.
        - Inclus tous les metrics monitoring
        - Gère plusieurs datasets (3)
        - Gradient normalisé pour éviter les NaN
        - Shuffle déterministe sur chaque dataset, et gère la reprises en fonction des batch_size et grad_accum
        - Shuffle recaculer à chaque changement d'epochs dataset par datasets
        - inclus les timings 
        """
        self.model.train()
        
        # --- PARAMÈTRES ---
        # Fréquence des points sur le graphique (ex: toutes les 500 mises à jour)
        eval_interval = self.params.get('eval_interval', 1000) 
        eval_iters = self.params.get('eval_iters', 30) 
        save_interval = self.params.get('save_interval', 1000)
        monitor_interval = self.params.get('monitor_interval', 10)
        
        batch_size = self.params['batch_size']
        grad_accum = self.params.get('grad_accum_steps', 1)
        current_grad_norm = []
        
        # Combien de steps on vise au total (ex: 100 000)
        max_steps = self.params.get('lr_decay_iters', 100_000)

        print(f"📦 Démarrage Training sur 3Go (Mode Continu)")
        print(f"   ▶️ Reprise au step : {self.total_steps_done}")
        print(f"   📉 Evaluation tous les : {eval_interval} steps")
        print(f"   💾 Sauvegarde tous les : {save_interval} steps")

        self._log_memory_status(v=True)

        # --- BOUCLE INFINIE (Pilotée par les steps) ---
        # On ne boucle pas sur 'epoch', on boucle jusqu'à 'max_steps'
        start_time = time.time()
        t_bacwd=0
        t_batch=0
        t_optim=0
        t_model=0
        t_init=0
        t_eval=0
        purge = 0
        t_monitor=time.time()
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
#                loss.backward()  # en v6.4 on calcule en direct
                t3 = time.time() # Temps backward pass (souvent le plus long)
                t_bacwd += t3 - t2
            
            # 3. Step Optimizer
            # ancienne version float32 remise en selle avec bfloat16 sur MPS
            """
            t4_start = time.time()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            current_grad_norm.append(float(norm)) # valeur pour le monitoring
            self.optimizer.step()
            """
            # On redescend les gradients avant le clipping
            t4_start = time.time()
            self.scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            current_grad_norm.append(float(norm))
            # Step final
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # fin de scaler""""            
            t4 = time.time() # Temps mise à jour optimizer
            # print(f"Data: {t1-t0:.3f}s | Fwd: {t2-t1:.3f}s | Bwd: {t3-t2:.3f}s | Opt: {t4-t3:.3f}s | Step: {t4-start_step:.3f}s")
            t_optim += t4 - t4_start
            # On incrémente le compteur GLOBAL
            self.total_steps_done += 1
            # Mise à jour EMA si activé
            if self.ema_model is not None:
                self.update_ema()            
            # --- EVALUATION & PLOT (Fréquent) ---
            t5 = time.time()
            if (self.total_steps_done % monitor_interval == 0) | (self.total_steps_done<5):
                self.monitor.log(self.total_steps_done, accum_loss, lr, monitor_interval, t5-t_monitor, self.total_step_wiki, self.total_step_cult, self.total_step_litt, purge, current_grad_norm,self.params)
                t_monitor=time.time()
                current_grad_norm = []
                purge=0
                
            if (self.total_steps_done % eval_interval == 0) | (self.total_steps_done<6):
                elapsed = time.time() - start_time
                # Calcul du Loss Val (Le vrai juge)
                losses, status, pression, mem_rss = self.estimate_loss_bin_v6(eval_iters=eval_iters)
                purge=1
                swap = psutil.swap_memory().used / (1_048_576)
                if self.total_steps_done % save_interval == 0:
                    print(f"step {self.total_steps_done}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e} ({elapsed:.2f}s) | {status} Po: {pression:.1f}% SW: {swap:.0f}Mo MPS: {torch.mps.current_allocated_memory() / 1_048_576:.0f}Mo", end="")
                else:
                    print(f"step {self.total_steps_done}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.2e} ({elapsed:.2f}s) | {status} Po: {pression:.1f}% SW: {swap:.0f}Mo MPS: {torch.mps.current_allocated_memory() / 1_048_576:.0f}Mo")
                # Mise à jour historique
                nb_step = self.total_steps_done - previous_step
                previous_step = self.total_steps_done
                self.history['train_loss'].append(losses['train'])
                self.history['train_loss_wiki'].append(losses['t_wiki'])
                self.history['train_loss_cult'].append(losses['t_cult'])
                self.history['train_loss_litt'].append(losses['t_litt'])
                self.history['val_loss'].append(losses['val'])
                self.history['val_loss_wiki'].append(losses['v_wiki'])
                self.history['val_loss_cult'].append(losses['v_cult'])
                self.history['val_loss_litt'].append(losses['v_litt'])
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
            if (self.total_steps_done % save_interval == 0) | (self.total_steps_done == 1):
                self._save_checkpoint(n_versions=self.params.get('n_version',5))
            t_eval += time.time() - t5
            
        self.monitor.stop()
        print("✅ Fin de l'entraînement (Max Steps atteint).")

    def estimate_loss_bin_v6(self, eval_iters=30):
        """
        Fonction helper pour estimer le loss sans dropout (mode eval).
        eval_iters: nombre de batchs pour moyenner et avoir un score stable.
        ici en V6 on va calculer par rapport aux 3 datasets
        """
        out = {}
        self.model.eval() # Désactive Dropout
        # on fixe les datasets
        # Dictionnaire des sources de validation à tester
        val_sources = {
            'wiki': self.val_data,
            'cult': getattr(self, 'val_cult', None), 
            'litt': getattr(self, 'val_litt', None)
            }
        train_sources = {
            'wiki': self.train_data,
            'cult': getattr(self, 'train_data_cult', None),
            'litt': getattr(self, 'train_data_litt', None)
            }

        # 1. Purge préventive avant l'effort d'évaluation
        gc.collect()           # Libère la RAM CPU (NumPy/Tensors CPU)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # 2. Utilisation de inference_mode (plus rapide que no_grad sur Metal)
        with torch.inference_mode(): # ce code crash !!
            with torch.autocast(device_type='mps', dtype=torch.float16):
#            with torch.no_grad():
            #2.1/ le train pqr source new version
            # on migre t_losses sur le CPU avec un numpy_array
                for name, data_source in train_sources.items():
                    if data_source is not None:
#                        t_losses = torch.zeros(eval_iters)
                        t_losses = []
                        for k in range(eval_iters):
                            X, Y = self.get_batch_from_source(data_source)
                            _, loss = self.model(X, Y)
#                            t_losses[k] = loss.item()
                            t_losses.append(loss.item())
#                        out[f"t_{name}"] = t_losses.mean().item()
                        out[f"t_{name}"] = sum(t_losses) / len(t_losses)

            # 2.2 /Validation détaillée par domaine
                for name, data_source in val_sources.items():
                    if data_source is not None:
#                        v_losses = torch.zeros(eval_iters)
                        v_losses = []
                        for k in range(eval_iters):
                            # On utilise ta logique "magie numpy" directement sur le source
                            X, Y = self.get_batch_from_source(data_source)
                            _, loss = self.model(X, Y)
#                            v_losses[k] = loss.item()
                            v_losses.append(loss.item())
#                        out[f'v_{name}'] = v_losses.mean().item()
                        out[f'v_{name}'] = sum(v_losses) / len(v_losses)

        # Calcul de la Loss globale pondérée pour le graphe principal
            out['train'] = (self.wiki_ratio * out.get('t_wiki', 0) +
                            self.cult_ratio * out.get('t_cult', 0) + 
                            self.litt_ratio * out.get('t_litt', 0))

            out['val']   = (self.wiki_ratio * out.get('v_wiki', 0) + 
                            self.cult_ratio * out.get('v_cult', 0) + 
                            self.litt_ratio * out.get('v_litt', 0))
       
        self.model.train() # Réactive Dropout
            
        # 3. Purge préventive avant l'effort d'évaluation
        gc.collect()           # Libère la RAM CPU (NumPy/Tensors CPU)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        status, pression, mem_rss = self._log_memory_status()

        return out, status, pression, mem_rss
    
    def _log_memory_status(self, v=False):
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

    @torch.no_grad()
    def update_ema(self):
        """
        Met à jour les poids de ema_model en direction de model.
        Formule : ema = ema + (1 - beta) * (train - ema)
        C'est mathématiquement équivalent à : ema = beta * ema + (1 - beta) * train
        """
        # On utilise lerp_ (Linear Interpolation in-place)
        # lerp(start, end, weight) -> start + weight * (end - start)
        # Pour nous : weight = (1 - ema_decay)
        weight = 1.0 - self.ema_decay
        for ema_param, train_param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.lerp_(train_param.data, weight)

# -----------------------------------------------------------------------------
# 5. MONITORING CLASS - sauvergarde en continue x steps du training
# -----------------------------------------------------------------------------

class Monitor:
    def __init__(self,file = 'model/monitor.log'):
        self.file = file
        self.queue = queue.Queue()
        self.active = True
        # Lancement du thread de monitoring
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"🚀 Monitor activé : écriture dans {self.file}")

    def _run(self):
        while self.active:
            data = self.queue.get()
            if data is None: 
                break
            # Écriture au format JSONL (une ligne par entrée)
            # C'est plus robuste pour les interruptions brutales
            with open(self.file, "a", encoding="utf-8") as f:
                f.write(json.dumps(data) + "\n")
            self.queue.task_done()
            
    def log(self, step, loss, lr, inter, elapse, dataset1, dataset2, dataset3, purge, array_grad_norm, params):
        # Capture des stats système
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()    
        data = {
            "status": "RUNNING",
            "last_update": time.strftime("%H:%M:%S"),
            "step": step,
            "loss": round(float(loss), 4),
            "lr": f"{lr:.2e}",
            "inter": inter,
            "elapse": round(float(elapse),2),
            "ram": mem.percent,
            "swap": round(swap.used / (1024**3), 2),
            "dataset1": dataset1,
            "dataset2": dataset2,
            "dataset3": dataset3,
            "purg": purge,
            "grad_norm": [ round(n, 4) for n in array_grad_norm],
            "params":params
            }
        self.queue.put(data)        

    def stop(self):
        """Ferme proprement le thread."""
        self.active = False
        self.queue.put(None)
        self.thread.join(timeout=2)

class GenerateGPT:
    def __init__(self, tokenizer, ckpt_path, default_model='model', verbose=False):
        self.tokenizer = tokenizer
        self.ckpt_path = ckpt_path
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.default_model = default_model
        self.verbose = verbose
        
    def load_for_inference(self):
        if not os.path.exists(self.ckpt_path):
            print("❌ Aucun modèle trouvé !")
            return None, None
        if self.verbose:
            print(f"Loading {self.ckpt_path} on {self.device}...")
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        
        self.config = checkpoint['config']
        
        # On recrée le modèle avec la config exacte de l'entraînement
        self.model = GPTLanguageModel(vocab_size=len(self.tokenizer.vocab), **self.config)
        self.model.load_state_dict(checkpoint[self.default_model])
        self.model.to(self.device)
        self.model.eval() # TRES IMPORTANT : désactive le Dropout
        
        # return model, config

    def generate_text(self, prompt, max_new_tokens=100, temperature=0.8, top_k=40, rep_penalty=1.2):
        # 1. Encodage
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        # 2. Génération
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context si trop long
                idx_cond = input_tensor[:, -self.model.block_size:]
                
                # Forward
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature # Applique la température

                # On parcourt les tokens déjà générés pour pénaliser les logits et réduire les répétitions
                """ en O(n2)
                for token_id in set(input_tensor[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= rep_penalty
                    else:
                        logits[0, token_id] *= rep_penalty
                """
                # v6.4 (O(n) vectorisé, reste sur le device)
                unique_ids = torch.unique(input_tensor[0])
                scores = logits[0, unique_ids]
                # Divise si positif, multiplie si négatif (même sémantique)
                penalty = torch.where(scores > 0, 
                                      torch.tensor(rep_penalty, device=self.device), 
                                      torch.tensor(1.0 / rep_penalty, device=self.device))
                logits[0, unique_ids] = scores / penalty
            
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
