import tools.ai_token as tk
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.optimizers import AdamW
import time


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.dim = config.n_embd
        
        # Projections linéaires pour Q, K, V
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias= False)
        self.dropout = nn.Dropout(config.dropout)
        self.att_dropout = nn.Dropout(config.dropout)
        # Calcule de la constante
        self.scale = (config.n_embd // config.n_head) ** -0.5
        # Pré-calcul du masque causal (pour une longueur max définie dans config)
        # On utilise un buffer pour qu'il soit sur le bon device
        # max_t = config.max_seq_len 
        # mask = nn.MultiHeadAttention.create_additive_causal_mask(max_t)
        # self.causal_mask = mask #dans le buffer parametre non entrainable

    def __call__(self, x, mask=None, cache=None):
        B, T, C = x.shape  # Batch, Sequence Length, Embedding Dim

        # Calcul de Q, K, V
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Séparation des têtes
        # (B, T, C) -> (B, T, n_head, C // n_head) -> (B, n_head, T, C // n_head)
        q = q.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        # --- GESTION DU CACHE ---
        if cache is not None:
            prev_k, prev_v = cache
            # On concatène les nouvelles clés/valeurs aux anciennes
            k = mx.concatenate([prev_k, k], axis=2)
            v = mx.concatenate([prev_v, v], axis=2)
        
        new_cache = (k, v) # On retourne le cache mis à jour
        # ------------------------        
        # Masquage causal (évite de voir les mots suivants)
        # On utilise le slicing dynamique du masque pré-calculé
        # mask_ = self.causal_mask[:T, :T]
        # Scaled Dot-Product Attention
        #att = (q @ k.transpose(0, 1, 3, 2)) * self.scale # mx.sqrt(q.shape[-1]
        att = (q @ k.swapaxes(-1, -2)) * self.scale
        if mask is not None:
            att = att + mask
        att = att - mx.max(att, axis=-1, keepdims=True)
        att = mx.softmax(att, axis=-1)
        att = self.att_dropout(att)
        # Boucle avec l'entrée
        y = att @ v # (B, n_head, T, T) x (B, n_head, T, head_size)
        # Recombinaison des têtes
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)

        return y, new_cache
    
class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        # On projette vers une dimension intermédiaire (souvent 4 * dim, ou 2/3 * 4 * dim pour Llama)
        hidden_dim = int(8/3 * config.n_embd) 
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        # SwiGLU : (SiLU(W1x) * W2x) * W3
        # L'opération mx.silu est très optimisée sur M1
        return self.dropout(self.w3(nn.silu(self.w1(x)) * self.w2(x)))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.h_dim = config.n_embd * 4 # hidden dimension
        # Dans les Transformers, on projette souvent vers 4x la dimension d'entrée
        self.net = [
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(), # GELU est plus performant que ReLU pour les Transformers
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout)
        ]
        self.net = nn.Sequential(*self.net)

    def __call__(self, x):
        return self.net(x)

class DecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.norm2 = nn.RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config)

    def __call__(self, x, mask=None, cache=None):
        # Connexion résiduelle 1 : Attention
        # Pre-norm : on normalise AVANT l'opération
        y , new_cache = self.attn(self.norm1(x), mask=mask, cache=cache)
        y = x + y
        
        # Connexion résiduelle 2 : Feed-Forward
        y = y + self.ffn(self.norm2(y))
        
        return y, new_cache
    
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = [DecoderBlock(config) for _ in range(config.n_layers)]
        self.n1 = nn.RMSNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # On crée le masque géant ici
        self.full_mask = nn.MultiHeadAttention.create_additive_causal_mask(config.block_size)

    def __call__(self, x, cache=None):
        B, T = x.shape
        offset = cache[0][0].shape[2] if cache is not None else 0
        # On prépare le masque pour toutes les couches d'un coup
        current_mask = self.full_mask[offset : offset + T, : offset + T]
        pos = mx.arange(offset, offset + T, dtype=mx.int32)
        tok_emb = self.wte(x)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb
        new_cache = []
        
        for i, block in enumerate(self.blocks):
            layer_cache = cache[i] if cache is not None else None
            # x, _ = block(x=x, mask=current_mask,cache=None)  # (B,T,C)
            x, layer_cache = block(x=x, mask=current_mask,cache=layer_cache)  # (B,T,C)
            new_cache.append(layer_cache)
        
        y = self.n1(x)
        # Logits
        logits = self.lm_head(y)  # (B,T,vocab_size)
        
        return logits, new_cache
    
    def generate(self, x, max_new_tokens, temperature=1.0):
        # x est de forme (B, L)
        for _ in range(max_new_tokens):
            # On respecte la fenêtre contextuelle (block_size)
            x_cond = x if x.shape[1] <= self.config.block_size else x[:, -self.config.block_size:]
            
            # Inférence : on ne récupère que les logits
            # Pas besoin de gradients ici (vitesse max)
            logits, _ = self(x_cond)
            
            # On prend le dernier jeton et on applique la température
            logits = logits[:, -1, :] / temperature
            
            # Échantillonnage
            next_token = mx.random.categorical(logits) # (B,)
            
            # Concaténation
            x = mx.concatenate([x, next_token[:, None]], axis=1)
            
        return x    

def loss_fn(model, x, y):
    logits, _ = model(x)
    logits = logits.reshape(-1, logits.shape[-1])
    return mx.mean(nn.losses.cross_entropy(logits, y.reshape(-1)))

class TransformerTrainer:
    def __init__(self, model, optimizer, dataset_obj, config):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset_obj
        self.config = config
        
        # On prépare la fonction de gradient
        self.loss_and_grad_fn = nn.value_and_grad(self.model, loss_fn)

    def get_batch(self, split="train"):
        data = self.dataset.train_data if split == "train" else self.dataset.val_data
        ix = mx.random.randint(0, len(data) - self.config.block_size, (self.config.batch_size,))
        ix_list = ix.tolist()
        x = mx.stack([data[i : i + self.config.block_size] for i in ix_list])
        y = mx.stack([data[i + 1 : i + self.config.block_size + 1] for i in ix_list])
        return x, y

    def run(self, max_iters, eval_interval=100):
        print(f"🚀 Mode Impératif (Sans compilation) - Stabilité Max")
        
        for i in range(max_iters):
            x, y = self.get_batch("train")
            
            # Calcul direct (non compilé)
            loss, grads = self.loss_and_grad_fn(self.model, x, y)
            
            # Mise à jour directe
            self.model.update(self.optimizer.apply_gradients(grads, self.model))
            
            # On évalue pour vider la file d'attente du GPU
            mx.eval(loss, self.model.parameters())
            
            if i % eval_interval == 0:
                print(f"Step {i:4d} | Loss: {loss.item():.4f}")
    
class MakeDataset:
    def __init__(self, text_encode, train_ratio=0.9):
        # 1. Conversion en array MLX 
        # C'est ici que la donnée "entre" dans l'écosystème MLX/GPU
        # text_encode est un tableau d'entiers 
        full_data = mx.array(text_encode, dtype=mx.int32)
        
        # 2. Split (Le slicing MLX est une "vue" : zéro copie mémoire)
        n = int(train_ratio * len(full_data))
        self.train_data = full_data[:n]
        self.val_data = full_data[n:]
        
        print(f"Dataset créé : Train ({len(self.train_data)} tokens), Val ({len(self.val_data)} tokens)")

    def get_info(self):
        # Vérification du "device" (M1 utilise l'architecture unifiée par défaut)
        return f"Mémoire utilisée par train_data : {self.train_data.nbytes / 1024**2:.2f} MB"

class Config:
    n_embd: int = 128
    n_layers: int = 2
    n_head: int = 4
    vocab_size: int = 5000 
    block_size: int = 256
    dropout: float = 0.15
    learning_rate: float = 3e-4
    batch_size: int = 32
    max_iters: int = 2000
    eval_iterval: int = 500




