import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import glob
import random
import json


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
    
# Supposons que vos classes GPTLanguageModel et tokenizer sont importées ou définies au dessus
# from model import GPTLanguageModel 

class ContinuousTrainer:
    def __init__(self, model_class, tokenizer, config, train_params, data_root, 
                 ckpt_path='ckpt.pth', log_file='processed_log.txt',history_path='history.json'):
        
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
        self.model = model_class(vocab_size=len(tokenizer.vocab), **config).to(self.device)
        
        # Optimiseur (On l'initialise ici, mais son état peut être écrasé si on charge un checkpoint)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=train_params['learning_rate'])
        self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), 
                    lr=train_params['learning_rate'],
                    weight_decay=train_params.get('weight_decay', 0.1)
                )
        # Liste des fichiers déjà traités & l'historique
        self.processed_files = self._load_processed_log()
        self.history = self._load_history()    
        # Chargement d'un checkpoint existant (si demandé)
        self.load_checkpoint()

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
        return {'train_loss': [], 'val_loss': [], 'steps': []}

    def _save_history(self):
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f)

    def _mark_file_as_done(self, file_path):
        """Ajoute un fichier à la liste des traités"""
        with open(self.log_file, 'a') as f:
            f.write(file_path + "\n")
        self.processed_files.add(file_path)

    def load_checkpoint(self):
        """Charge les poids. Permet de changer les hyperparams d'entraînement (LR) mais garde les poids."""
        if os.path.exists(self.ckpt_path):
            print(f"📥 Chargement du checkpoint : {self.ckpt_path}")
            # Chargement sur CPU d'abord pour sécurité
            ckpt = torch.load(self.ckpt_path, map_location=self.device) # à verifier ...
            
            # 1. Chargement des poids du modèle
            self.model.load_state_dict(ckpt['model'])
            
            # 2. On NE charge PAS l'optimizer si on veut changer le learning rate manuellement
            # pour une continuité parfaite de l'optimizer, décommentez la ligne suivante :
            # self.optimizer.load_state_dict(ckpt['optimizer'])
            
            print(f"✅ Modèle restauré. (Fichiers déjà traités : {len(self.processed_files)})")
        else:
            print("✨ Aucun checkpoint trouvé, démarrage à zéro.")

    def save_checkpoint(self):
        print(f"💾 Sauvegarde du checkpoint...")
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'vocab_size': len(self.tokenizer.vocab)
        }
        torch.save(ckpt, self.ckpt_path)

    def get_batch(self, data_tensor):
        block_size = self.config['block_size']
        batch_size = self.params['batch_size'] # faut il verifier len(data_tensor) / batch_size
        ix = torch.randint(len(data_tensor) - block_size, (batch_size,))
        x = torch.stack([data_tensor[i:i+block_size] for i in ix])
        y = torch.stack([data_tensor[i+1:i+block_size+1] for i in ix])
        return x.to(self.device), y.to(self.device)

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
            
            # Gradient Accumulation
            self.optimizer.zero_grad()
            accum_loss = 0
            
            for _ in range(grad_accum):
                X, Y = self.get_batch(train_data)
                logits, loss = self.model(X, Y)
                loss = loss / grad_accum
                accum_loss += loss.item()
                loss.backward()
            
            self.optimizer.step()
            total_loss += accum_loss

        avg_loss = total_loss / n_batches
        return avg_loss, train_data, val_data
    
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

    def generate_text(self, prompt, max_new_tokens=100, temperature=0.8):
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
                
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Sampling
                idx_next = torch.multinomial(probs, num_samples=1)
                input_tensor = torch.cat((input_tensor, idx_next), dim=1)

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

    def plot_metrics(self, window_size=5, save_path=None):
        """
        Trace les courbes de loss.
        window_size : taille de la fenêtre pour le lissage (moyenne mobile).
        save_path : chemin pour enregistrer l'image (ex: 'progress.png').
        """
        data = self.load_data()
        if not data: return

        train_loss = data.get('train_loss', [])
        val_loss = data.get('val_loss', [])
        steps = range(len(train_loss))

        if len(train_loss) == 0:
            print("Aucune donnée à tracer.")
            return

        plt.figure(figsize=(12, 6))
        
        # 1. Tracé des données brutes (en pointillés légers)
        plt.plot(steps, train_loss, color='blue', alpha=0.2, label='Train (brut)')
        plt.plot(steps, val_loss, color='red', alpha=0.2, label='Val (brut)')

        # 2. Calcul et tracé des moyennes mobiles (lissage)
        if len(train_loss) >= window_size:
            train_smooth = np.convolve(train_loss, np.ones(window_size)/window_size, mode='valid')
            val_smooth = np.convolve(val_loss, np.ones(window_size)/window_size, mode='valid')
            
            # Ajustement des steps pour le décalage de la moyenne mobile
            smooth_steps = range(window_size - 1, len(train_loss))
            
            plt.plot(smooth_steps, train_smooth, color='blue', linewidth=2, label=f'Train (lissé {window_size}nd)')
            plt.plot(smooth_steps, val_smooth, color='red', linewidth=2, label=f'Val (lissé {window_size}nd)')

        # Configuration du graphique
        plt.title('Évolution du Loss pendant l\'apprentissage (Wiki FR)', fontsize=14)
        plt.xlabel('Nombre de fichiers traités', fontsize=12)
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
