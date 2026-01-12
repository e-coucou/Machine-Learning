import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass

import time
import math,re
#---------------------------------------------------------------
# Hyperparameters
torch.manual_seed(1965)
# data Loading ------------------------------------------------

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel - optimized for M1 """
    # class pour Self-Attention et Masked-Attention
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        
        # Combined linear layers for all heads at once
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
# plus besoin car masque dans la classe        
#        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.scale = head_size ** -0.5

    def forward(self, x, mask=None): # par defaut self Attention
        B, T, C = x.shape
        
        # Compute Q, K, V in one go
        qkv = self.qkv(x)  # (B, T, 3*n_embd)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_size)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Efficient attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)
        
        # Optimized masking / padding
        # mask arrive ici avec la forme (B, 1, T, T) 
        # Il contient le mélange (Causal AND Padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))   
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, num_heads, T, head_size)
        
        # Merge heads
        out = out.permute(0, 2, 1, 3).contiguous()  # (B, T, num_heads, head_size)
        out = out.reshape(B, T, -1)  # (B, T, n_embd)
        
        out = self.proj(out)
        out = self.dropout(out)
        return out
    
class CrossAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.kv_proj = nn.Linear(n_embd, 2 * n_embd, bias=False) # K et V groupés
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = head_size ** -0.5

    def forward(self, x, enc_output, mask=None):
        B, T, C = x.shape      # T est la longueur du résumé
        B, T_enc, _ = enc_output.shape # T_enc est la longueur de l'article
        
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_size).transpose(1, 2)
        kv = self.kv_proj(enc_output).reshape(B, T_enc, 2, self.num_heads, self.head_size).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # padding mask
        if mask is not None:
            # mask shape: (B, 1, 1, T)
            scores = scores.masked_fill(mask == 0, float('-inf'))        

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Pas de masque causal : le décodeur peut regarder TOUT l'article source
        out = torch.matmul(attn_weights, v).permute(0, 2, 1, 3).contiguous().view(B, T, C)

        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ a simple feed-forward neural network - optimized for M1 """
    def __init__(self, n_embd, dropout):
        super().__init__()
        hidden_dim = 4 * n_embd
        
        # Use GELU instead of ReLU (better for transformers + faster on M1)
        self.fc1 = nn.Linear(n_embd, hidden_dim, bias=True)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, n_embd, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    """ Transformer block: communication followed by computation - optimized for M1 """
    def __init__(self, n_embd, num_heads, block_size=None, dropout=0.1):
        super().__init__()
        head_size = n_embd // num_heads
        # Layer norms with eps optimized for M1
        self.ln1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-5)
        self.sa = MultiHeadAttention(num_heads=num_heads, head_size=head_size,n_embd=n_embd,block_size=block_size,dropout=dropout)
        self.ffwd = FeedForward(n_embd=n_embd,dropout=dropout)

    def forward(self, x, mask):
        # Pre-layer norm with residual connections (more stable)
        x = x + self.sa(self.ln1(x), mask=mask) # mask=True par defaut
        x = x + self.ffwd(self.ln2(x))
        return x

class BlockEncoder(nn.Module):
    def __init__(self, n_embd, num_heads, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        # Utilisation de la nouvelle classe optimisée sans masque
        self.ln1 = nn.LayerNorm(n_embd, eps=1.e-5)
        self.ln2 = nn.LayerNorm(n_embd, eps=1.e-5)
        self.sa = MultiHeadAttention(num_heads=num_heads, head_size=head_size, n_embd=n_embd, dropout=dropout)
        self.ffwd = FeedForward(n_embd=n_embd, dropout=dropout) # classe FFwd habituelle

    def forward(self, x, mask=None):
        # L'encodeur traite tout le contexte d'un coup (Self-Attention bidirectionnelle non maskée)
        x = x + self.sa(self.ln1(x), mask=mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class BlockDecoder(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout):
        super().__init__()
        head_size = n_embd // num_heads
        
        # 1. Self-Attention (Masquée) : Pour regarder le passé du résumé
        self.sa = MultiHeadAttention(num_heads=num_heads, head_size=head_size, 
                                      n_embd=n_embd, block_size=block_size, dropout=dropout)
        
        # 2. Cross-Attention : Pour regarder l'article source
        self.ca = CrossAttention(num_heads=num_heads, head_size=head_size, 
                                 n_embd=n_embd, dropout=dropout)
        
        # 3. Feed Forward : Le réseau de neurones de traitement
        self.ffwd = FeedForward(n_embd, dropout)
        
        # Layer Normalizations
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # x : (B, T, C) - Le résumé en cours de génération
        # enc_output : (B, T_article, C) - La sortie de l'encodeur
        
        # Étape 1 : Analyser ce qu'on a déjà écrit
        x = x + self.sa(self.ln1(x), mask=tgt_mask) # mask = True par defaut
        
        # Étape 2 : Chercher les infos pertinentes dans l'article (Cross-Attention)
        # On passe 'x' comme Query et 'enc_output' comme Key/Value
        x = x + self.ca(self.ln2(x), enc_output, mask=src_mask)
        
        # Étape 3 : Réflexion et projection
        x = x + self.ffwd(self.ln3(x))
        
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Empilement des blocs d'encodeur
        self.blocks = nn.Sequential(*[
            BlockEncoder(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.register_buffer('pos_idx', torch.arange(block_size))

    def forward(self, idx, mask=None):
        B, T = idx.shape
        
        # Embeddings + Positions
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(self.pos_idx[:T]) # (T, n_embd)
        
        x = tok_emb + pos_emb
        x = self.blocks(x, mask=mask)
        x = self.ln_f(x)
        return x # Sortie de l'encodeur : (B, T, n_embd)

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads, n_layer, dropout):
        super().__init__()
        # 1. L'Encodeur (traite l'article source)
        self.encoder = Encoder(vocab_size=vocab_size, n_embd=n_embd, block_size=block_size, num_heads=num_heads, n_layer=n_layer, dropout=dropout)
        
        # 2. Le Décodeur (génère le résumé)
        self.decoder_token_embedding = nn.Embedding(vocab_size, n_embd)
        self.decoder_position_embedding = nn.Embedding(block_size, n_embd)
        self.decoder_blocks = nn.ModuleList([
            BlockDecoder(n_embd=n_embd, num_heads=num_heads, block_size=block_size, dropout=dropout) for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.block_size = block_size
        self.register_buffer('pos_idx', torch.arange(block_size))

    def forward(self, source_idx, target_idx, targets=None):
        # source_idx : IDs de l'article (B, T_src)
        # target_idx : IDs du résumé partiel (B, T_tgt)
        
        B, T_tgt = target_idx.shape
        # ETAPE 0 :  Génération du masque de l'article
        src_mask = self.make_src_mask(source_idx)
        tgt_mask = self.make_tgt_mask(source_idx)

        # ÉTAPE 1 : Encodage de l'article (une seule fois pour toute la séquence)
        # On obtient la "mémoire" de l'article
        enc_output = self.encoder(source_idx,mask=src_mask) # (B, T_src, n_embd)
        
        # ÉTAPE 2 : Préparation du décodeur
        tok_emb = self.decoder_token_embedding(target_idx)
        pos_emb = self.decoder_position_embedding(self.pos_idx[:T_tgt])
        x = tok_emb + pos_emb
        
        # ÉTAPE 3 : Passage dans les blocs de décodeur avec Cross-Attention
        for block in self.decoder_blocks:
            x = block(x, enc_output, src_mask=src_mask, tgt_mask=tgt_mask ) # On injecte la mémoire de l'encodeur ici
            
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T_tgt, vocab_size)

        # Calcul de la perte si on est en train
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def make_src_mask(self, src):
        # src: (B, T_src)
        # On crée un masque qui est à True pour les vrais mots et False pour le padding
        # Le format (B, 1, 1, T) permet de diffuser le masque sur toutes les têtes d'attention
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask # (B, 1, 1, T_src)

    def make_tgt_mask(self, tgt):
        # 1. Masque de padding (B, 1, 1, T_tgt)
        padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        
        # 2. Masque causal (1, 1, T_tgt, T_tgt)
        T_tgt = tgt.size(1)
        causal_mask = torch.tril(torch.ones((T_tgt, T_tgt), device=tgt.device)).bool()
        
        # 3. On combine les deux : il faut que ce soit un mot ET que ce soit dans le passé
        tgt_mask = padding_mask & causal_mask
        return tgt_mask     

# Décoder Only ------------------------------------------------
class BigramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, device, n_embd, num_heads, n_layers, block_size, dropout):
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        # Embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Transformer blocks
        #self.blocks = nn.Sequential(*[Block(n_embd=n_embd, num_heads=num_heads, block_size=block_size, dropout=dropout) for _ in range(n_layers)])
        self.blocks = nn.ModuleList([Block(n_embd=n_embd, num_heads=num_heads, block_size=block_size, dropout=dropout) for _ in range(n_layers)])        
        self.ln_final = nn.LayerNorm(n_embd, eps=1e-5)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # On enregistre le masque causal maximum une seule fois
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embeddings
        token_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=self.device))  # (T,C)
        
        # Fusion embeddings
        x = token_emb + pos_emb  # (B,T,C)
        
        # Forward through transformer blocks
        mask = self.make_mask(idx)
        for block in self.blocks:
            x = block(x, mask)  # (B,T,C)
        
        # Final layer norm
        x = self.ln_final(x)  # (B,T,C)
        
        # Logits
        logits = self.lm_head(x)  # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def make_mask(self, tgt):
        # Masque causal (1, 1, T_tgt, T_tgt)
        T_tgt = tgt.size(1)
        # supprimer car plus rapide dans metal M1/M4
        #causal_mask = torch.tril(torch.ones((T_tgt, T_tgt), device=self.device)).bool()
        return self.tril[:T_tgt, :T_tgt].unsqueeze(0).unsqueeze(0) # retour (1,1,T,T)
    
    def generate(self, idx, max_new_tokens):
        """Generate new tokens efficiently on M1"""
        with torch.no_grad():  # Désactiver les gradients pour la génération
            for _ in range(max_new_tokens):
                # Crop à la dernière séquence de block_size tokens
                idx_cond = idx[:, -self.block_size:]
                
                # Forward pass
                logits, _ = self(idx_cond)
                
                # Focus sur le dernier timestep
                logits = logits[:, -1, :]  # (B,C)
                
                # Softmax et sampling
                probs = F.softmax(logits, dim=-1)  # (B,C)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
                
                # Append au contexte
                idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        
        return idx

# Model -------------------------------------------------------
class Model():
    def __init__(self, text_encode, token,n_embd, num_heads, n_layers, block_size, dropout,batch_size, max_iters,eval_interval,eval_iters,learning_rate,fileout='data/test.pth'):
        self.token = token
        self.vocab_size = len(token.vocab)
        self.fileOut = fileout
        self.config = {
            'n_embd': n_embd,
            'num_heads': num_heads,
            'n_layers': n_layers,
            'block_size': block_size,
            'dropout': dropout
        }
        self.parameters = {
            'batch_size': batch_size,
            'max_iters': max_iters,
            'eval_interval': eval_interval,
            'eval_iters': eval_iters,
            'learning_rate': learning_rate
        }
        self.block_size = block_size
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters

        # Configuration du device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            # Note: set_float32_matmul_precision n'affecte que CUDA, 
            # mais on le garde si vous migrez un jour sur NVIDIA.
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")

        # Conversion des données en tensor directement sur le device
        self.data = torch.tensor(text_encode, dtype=torch.long, device=self.device)
        
        # Split Train/Val
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

        # Initialisation du modèle
        # Assurez-vous que BigramLanguageModeler est défini avant
        self.model = BigramLanguageModeler(self.vocab_size, self.device, **self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def get_batch(self, split):
        """
        Version optimisée : On évite la boucle Python 'for i in ix' 
        en utilisant l'indexation vectorisée de PyTorch.
        """
        data = self.train_data if split == 'train' else self.val_data
        # Générer des indices de départ
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,),device=self.device)
        
        # Création des séquences de manière vectorisée
        # On crée une grille d'indices : chaque ligne est un range de block_size
        offsets = torch.arange(self.block_size, device=self.device)
        ix_grid = ix.view(-1, 1) + offsets # Shape (self.batch_size, block_size)
        
        x = data[ix_grid]
        y = data[ix_grid + 1] # Décalage de 1 pour la cible
        
        return x, y
    
    def get_batch_v1(self, split):
        data = self.train_data if split == 'train' else self.val_data
        
        # 1. On génère les indices sur CPU (plus rapide pour les petits nombres aléatoires)
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        
        # 2. On extrait les données (le CPU est très efficace pour le slicing en mémoire unifiée)
        # On utilise la liste de compréhension qui, bizarrement, est très rapide ici 
        # car elle évite de créer de gros tensors intermédiaires d'indices.
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        
        # 3. On s'assure que le résultat est sur le device
        # (Si data est déjà sur MPS, x et y le seront aussi automatiquement)
        return x, y

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters, device=self.device)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def train(self):
        print(f"🚀 Starting training on {self.device}...")
        start_time = time.time()
        self.train_losses, self.val_losses, self.steps_recorded = [], [], []
        self.model.train()
        max_iters = self.parameters.get('max_iters')

        for steps in range(max_iters):
            # Évaluation périodique
            if (steps % self.eval_interval == 0) or (steps == self.max_iters - 1):
                losses = self.estimate_loss()
                elapsed = time.time() - start_time
                
                if steps > 0:
                    time_per_step = elapsed / steps
                    remaining = (self.max_iters - steps) * time_per_step
                    print(f"Step {steps}/{self.max_iters}: train loss {losses['train']:.4f}, "
                          f"val loss {losses['val']:.4f} | ETA: {remaining:.1f}s")
                else:
                    print(f"Step {steps}/{self.max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                self.train_losses.append(losses['train'])
                self.val_losses.append(losses['val'])
                self.steps_recorded.append(steps)

            # --- BOUCLE D'ENTRAÎNEMENT ---
            xb, yb = self.get_batch('train')

            # Utilisation de Mixed Precision (Autocast) si possible pour MPS
            # Note: Utile principalement sur les modèles lourds
            logits, loss = self.model(xb, yb)
            
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time:.2f}s")
        
        self.save_model()

    def save_model(self):
        print('----- Saving model -----')
        checkpoint = {
            # .cpu() assure que les poids sont sauvegardés de manière universelle
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(), # permet de relancer un apprentissage ou faire du fine tuning
            'vocab_size': self.vocab_size,
            'config': self.config,
            'parameters': self.parameters,
            'history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'steps_recorded': self.steps_recorded
            }
        }
        torch.save(checkpoint, self.fileOut)
        print(f"Model saved to {self.fileOut}")

    def load_model(self, path=None):
        path = path if path else self.fileOut
        print(f"----- Loading model from {path} -----")
        
        # 1. Charger le fichier sur le CPU (Sécurité totale pour la compatibilité)
        checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        
        # 2. Restaurer la configuration et le vocabulaire
        self.vocab_size = checkpoint['vocab_size']
        self.config = checkpoint['config']
        self.parameters = checkpoint['parameters']
        self.block_size = self.config.get('block_size')
        self.batch_size = self.parameters.get('batch_size')
        self.max_iters = self.parameters.get('max_iters')
        self.eval_interval = self.parameters.get('eval_interval')
        self.eval_iters = self.parameters.get('eval_iters')
        
        # 3. Recréer le modèle et l'envoyer immédiatement sur le device (MPS ou CPU)
        # On passe les arguments de config pour reconstruire l'architecture exacte
        self.model = BigramLanguageModeler(self.vocab_size, self.device, **self.config).to(self.device)
        
        # 4. Charger les poids dans le modèle
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 5. Réinitialiser l'optimiseur sur les paramètres du nouveau modèle (déjà sur MPS)
        # Note : 'learning_rate' doit être défini dans ton code
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # 6. Charger l'état de l'optimiseur
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # --- OPTIMISATION CRUCIALE POUR MAC M1 ---
            # L'état de l'optimiseur chargé est sur CPU. On doit forcer chaque tenseur 
            # interne (moments Adam) à migrer sur le GPU (MPS).
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        # 7. Restaurer l'historique dans l'objet
        if 'history' in checkpoint:
            self.train_losses = checkpoint['history'].get('train_losses', [])
            self.val_losses = checkpoint['history'].get('val_losses', [])
            self.steps_recorded = checkpoint['history'].get('steps_recorded', [])
        
        # 8. Mettre en mode évaluation par défaut (plus sûr)
        self.model.eval()
        
        print(f"✅ Model loaded and successfully migrated to {self.device}")
        return checkpoint.get('history', None)

    def genere(self, prompt=None,new_token=1000):

        self.model.eval() # Toujours mettre en eval pour la génération
        with torch.no_grad():
            if prompt is None:
                # Pas de contexte: commence avec un token vide
                context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            else:
                # Encode le prompt en tokens
                prompt_tokens = self.token.encode(prompt)  # Retourne une liste de tokens
                context = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        
            generated_indices = self.model.generate(context, max_new_tokens=new_token)[0].tolist()
            generated_text = self.token.decode(generated_indices, self.token.vocab)
            print(generated_text)


    def get_model_size(self):
        """
        Calcule le nombre de paramètres et la taille en mémoire.
        """
        # Nombre total de paramètres
        params_count = sum(p.numel() for p in self.model.parameters())
        
        # Paramètres entraînables uniquement
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Taille estimée en MegaOctets (en float32, 1 param = 4 octets)
        size_mb = (params_count * 4) / (1024**2)
        
        print("-" * 30)
        print(f"📊 Model Statistics:")
        print(f"   Total Parameters: {params_count:,}")
        print(f"   Trainable:        {trainable_params:,}")
        print(f"   Estimated Size:   {size_mb:.2f} MB")
        print("-" * 30)
        return
    
    def plot_losses(self, history=None):
        """
        Trace les courbes de perte Train et Val.
        Si history n'est pas fourni, utilise l'historique interne du modèle.
        """
        if history is None:
            # On suppose que vous avez stocké ces listes pendant l'entraînement
            train_loss = getattr(self, 'train_losses', [])
            val_loss = getattr(self, 'val_losses', [])
            steps = getattr(self, 'steps_recorded', [])
        else:
            train_loss = history['train_losses']
            val_loss = history['val_losses']
            steps = history['steps_recorded']

        if not train_loss:
            print("Aucune donnée d'historique à tracer.")
            return

        plt.figure(figsize=(10, 6))
        
        # Tracer les lignes originales (en transparence)
        plt.plot(steps, train_loss, label='Train Loss', color='skyblue', alpha=0.4)
        plt.plot(steps, val_loss, label='Val Loss', color='salmon', alpha=0.4)
        
        # Ajouter une moyenne mobile pour voir la tendance (lissage)
        if len(train_loss) > 10:
            window = 5
            train_smooth = np.convolve(train_loss, np.ones(window)/window, mode='valid')
            val_smooth = np.convolve(val_loss, np.ones(window)/window, mode='valid')
            # On ajuste les steps pour le mode 'valid'
            steps_smooth = steps[window-1:]
            
            plt.plot(steps_smooth, train_smooth, color='blue', label='Train Loss (Lissée)')
            plt.plot(steps_smooth, val_smooth, color='red', label='Val Loss (Lissée)')

        plt.title('Évolution de la Perte (Loss) pendant l\'entraînement')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.show()

class Model_():
    def __init__(self, model, tokenizer, train_data, val_data, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.config = config # Contient batch_size, lr, etc.
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.device = config.device
        self.history = {'train_loss': [], 'val_loss': []}

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        
        # Génération des indices de départ pour le batch
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        
        # On extrait les séquences directement
        # Puisque data est déjà un tenseur, on utilise le slicing PyTorch
        x = torch.stack([data[i : i + self.config.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.config.block_size + 1] for i in ix])
        
        # On s'assure qu'ils sont au bon format (Long) et sur le bon device (MPS)
        return x.to(self.device).long(), y.to(self.device).long()
    
    def train(self):
        self.model.train()
        start_time = time.time()
        
        for iter in range(self.config.max_iters):
            # 1. Récupération des données
            xb, yb = self.get_batch('train')

            # 2. Forward pass & Loss
            logits, loss = self.model(xb, yb)

            # 3. Backward pass (Optimisé M1/M4)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            # 4. Logs
            if iter % self.config.eval_interval == 0:
                self.history['train_loss'].append(loss.item())
                print(f"Iter {iter}: loss {loss.item():.4f}")

    def plot_losses(self):
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.legend()
        plt.show()

class T_model():
    def __init__(self, model, tokenizer, train_data, val_data, config):
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.val_data = val_data
        self.config = config # Contient batch_size, lr, etc.
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.device = config.device
        self.history = {'train_loss': [], 'val_loss': []}

    def get_batch(self, split):
        # On imagine que tu as chargé des listes d'articles et de résumés
        articles = self.train_articles if split == 'train' else self.val_articles
        resumes = self.train_resumes if split == 'train' else self.val_resumes
        
        ix = torch.randint(len(articles), (self.config.batch_size,))
        
        # On prépare les tenseurs
        # x_enc : L'article complet (Source)
        # x_dec : Le résumé décalé pour l'entrée du décodeur (Cible Input)
        # y     : Le résumé décalé pour la perte (Cible Target)
        x_enc = torch.stack([articles[i] for i in ix])
        x_dec = torch.stack([resumes[i][:, :-1] for i in ix]) # Tout sauf le dernier mot
        y     = torch.stack([resumes[i][:, 1:] for i in ix])  # Tout sauf le premier mot
        
        return x_enc.to(self.device), x_dec.to(self.device), y.to(self.device)

    def train(self):
        self.model.train()
        for iter in range(self.config.max_iters):
            # On récupère les 3 éléments (Article, Résumé_Input, Résumé_Cible)
            src, tgt_in, tgt_out = self.get_batch('train')

            # Forward pass : le Transformer reçoit l'article et le début du résumé
            # targets=tgt_out permet de calculer la CrossEntropy en interne
            logits, loss = self.model(src, tgt_in, targets=tgt_out)

            # Optimisation (Backpropagation)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if iter % self.config.eval_interval == 0:
                print(f"Step {iter}: Loss {loss.item():.4f}")

    def plot_losses(self):
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.legend()
        plt.show()