import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math,re
#---------------------------------------------------------------
# Vérification de l'installation de PyTorch et de la disponibilité de MPS
print(f"PyTorch version: {torch.__version__}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
# Vérifier d'où vient PyTorch
#print(f"PyTorch location: {torch.__file__}")
#---------------------------------------------------------------
# variables globales
fileName = 'data/corpus_Moliere.txt'
fileOut = 'bigram_moliere_model.pth'
addedToken = 4000  # Nombre de tokens à ajouter au vocabulaire de base (bytes 0-255)
#-- 
# Hyperparameters
torch.manual_seed(1965)
batch_size = 64 #64 # how many independent sequences will we process in parallel
block_size = 256 #256  # what is the maximum context length for predictions
max_iters = 5000 # number of training iterations
eval_interval = 500 # interval for evaluating the loss
eval_iters = 200 # number of iterations for loss estimation
learning_rate = 3e-4 # learning rate for the optimizer
n_embd = 128 #384 # embedding dimension
num_heads = 4 #6 # number of attention heads
n_layers = 3 #6 # number of transformer blocks
dropout = 0.2 # dropout rate
#---------------------------------------------------------------
#device = 'cuda' if torch.cuda.is_available() else 'cpu' /for NVIDIA GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")
# device = torch.device("cpu") # force CPU for compatibility
#---------------------------------------------------------------
# Load and preprocess the text data
#nettoyage du texte
def preprocess_text(_text):
    """Nettoie un fichier texte pour l'entraînement NLP"""
        
    # Stats avant nettoyage
    original_length = len(_text)
    original_lines = _text.count('\n')
    
    # Nettoyage
    _text = re.sub(r'[ \t]+', ' ', _text)              # Espaces multiples
    _text = re.sub(r'\n{3,}', '\n\n', _text)           # Lignes vides
    _text = re.sub(r'(?m)^[ \t]+|[ \t]+$', '', _text)  # Espaces début/fin ligne
    _text = re.sub(r'\s+([.,;:!?])', r'\1', _text)     # Espace avant ponctuation
    _text = _text.strip()

    # 1. Remplacements intelligents
    _text = _text.replace('—', '-')
    _text = _text.replace('«', '"').replace('»', '"')
    _text = _text.replace('[', '(').replace(']', ')')
    _text = _text.replace('{', '(').replace('}', ')')
    _text = _text.replace(''', "'").replace(''', "'")
    _text = _text.replace('…', '...')
    _text = _text.replace('«', '"').replace('»', '"')
    
    # 2. Filtrer caractères
    allowed = set(
        'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789 .,;:!?\'\"-\n'
        'àâçèéêëîïôùûÀÂÇÉÈÊËÎÏÔÙÛœŒ'
        '()'
    )
    _text = ''.join(c for c in _text if c in allowed)
    
    # 3. Normaliser espaces
    _text = re.sub(r' +', ' ', _text)
    _text = re.sub(r'\n{3,}', '\n\n', _text)
    _text = re.sub(r'(?m)^[ ]+|[ ]+$', '', _text)
        
    _text = _text.strip()    

    # Stats après nettoyage
    cleaned_length = len(_text)
    cleaned_lines = _text.count('\n')
    
    print(f"Nettoyage terminé:")
    print(f"  Caractères: {original_length:,} → {cleaned_length:,} ({cleaned_length/original_length*100:.1f}%)")
    print(f"  Lignes: {original_lines:,} → {cleaned_lines:,}")
    
    return _text
def getStats(ids):
    count = {}
    for pair in zip(ids, ids[1:]):
        count[pair] = count.get(pair, 0) + 1
    return count
def merge(ids, pair, idx):
    i = 0
    merged = []
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged.append(idx)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged
#---------------------------------------------------------------
# Read the text file
print('-----Loading and preprocessing text-----')
with open(fileName, 'r', encoding='utf-8') as f:
    _text = f.read()
text = preprocess_text(_text)
print(f"Longueur du texte après nettoyage: {len(text):,} caractères")
tokens = text.encode("utf-8") #raw byte
ids = list(tokens)
merges = {}
start_time = time.time()
print('-----Building BPE tokenizer-----')
for i in range(addedToken):
    s = getStats(ids)
    pair = max(s, key=s.get)
    if not s:
        break
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx
elapsed = time.time() - start_time
print(f"✅ BPE tokenizer built in {elapsed:.2f}s")
print('  Nombre de tokens avant BPE :', len(tokens))
print('  Nombre de tokens après BPE :', len(ids))
print(f'  -> compression : {len(tokens)/len(ids):.2f}x')
vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0,p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
def decode(ids,vocab):
    tokens = b''.join(vocab[idx] for idx in ids)
    return tokens.decode('utf-8', errors='replace')
def encode(text,vocab):
    tokens = list(text.encode("utf-8")) #raw byte
    while len(tokens) > 1:
        stats = getStats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))
        if not pair in merges:
            break # No more merges available
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens
#---------------------------------------------------------------
# Create character-level vocabulary
#chars = sorted(list(set(text)))
chars = sorted(list(set(ids)))
vocab_size = len(chars)
print('Nombre de caractères uniques :', vocab_size)
#print('Liste des caractères uniques :', ''.join(chars))
# Create mappings from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# Encoding and decoding functions
"""
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
"""
# split the data into training and validation sets
print('-----Preparing data-----')
start_time = time.time()
#data = torch.tensor(encode(text, vocab), dtype=torch.long)
data = torch.tensor(ids, dtype=torch.long)
elapsed = time.time() - start_time
print(f"✅ Data encoded in {elapsed:.2f}s")
#print(data.shape, data.dtype, data)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
# data Loading ------------------------------------------------
# Function to get a batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
# Estimate loss on train and val sets no gradient needed
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,head_size)
        q = self.query(x) # (B,T,head_size)
        # compute attention scores ("affinities")
        weights = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # mask upper triangular part with -inf (for softmax)
        weights = F.softmax(weights, dim=-1) # (B,T,T)
        weights = self.dropout(weights)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size)
        out = weights @ v  # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # concatenate outputs of all heads
        out = self.proj(out) # linear projection
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    """ a simple feed-forward neural network """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 *n_embd), # expand to 4*n_embd?
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # project back to n_embd
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, num_heads):
        # n_embd: embedding dimension, num_heads: number of attention heads
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Model Definition ------------------------------------------------
class BigramLanguageModeler(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads=num_heads) for _ in range(n_layers)])
#        self.sa_heads = MultiHeadAttention(num_heads, n_embd // num_heads) # 4 heads of self-attention
#        self.ffwd = FeedForward(n_embd) # feed-forward layer
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)  # (B,T,C)
        positional_emb = self.positional_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = token_emb + positional_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
#        x = self.sa_heads(x)  # apply one head self-attention (B,T,C)
#        x = self.ffwd(x)      # apply feed-forward layer (B,T,C)
        x = self.ln_final(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size) C=vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # targets shape (-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)
        return idx
# Training Loop ------------------------------------------------
# Instantiate the model and optimizer
print('-----Initializing model-----')
model = BigramLanguageModeler().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
total_params = sum(p.numel() for p in model.parameters())
print(f"  Nombre total de paramètres dans le modèle - (En millions) : {total_params/1e6:.2f}M ")
start_time = time.time()
# Initialiser les listes pour stocker l'historique
train_losses = []
val_losses = []
steps_recorded = []
print('-----Starting training-----')
for steps in range(max_iters):
    if (steps % eval_interval == 0) | (steps == max_iters - 1):
        losses = estimate_loss() # dictionnary with train and val losses
        elapsed = time.time() - start_time
        
        # Estimation du temps restant
        if steps > 0:
            time_per_step = elapsed / steps
            remaining_steps = max_iters - steps
            eta = time_per_step * remaining_steps
            print(f"  step {steps}/{max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | {elapsed:.1f}s elapsed, ETA: {eta:.1f}s")
        else:
            print(f"  step {steps}/{max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Sauvegarder les valeurs
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        steps_recorded.append(steps)
    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(f"\n✅ Training completed in {time.time() - start_time:.2f}s")
# Save the model
print('-----Saving model-----')
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'vocab': vocab,
    'stoi': stoi,
    'itos': itos,
    'n_embd': n_embd,           # ← IMPORTANT
    'num_heads': num_heads,           # ← IMPORTANT
    'n_layers': n_layers,         # ← IMPORTANT
    'block_size': block_size,   # ← IMPORTANT
    'dropout': dropout,         # ← IMPORTANT
    'stoi': stoi,
    'itos': itos,
    # Ajouter l'historique
    'train_losses': train_losses,
    'val_losses': val_losses,
    'steps_recorded': steps_recorded,
    'final_step': steps,
}, fileOut)
print('Model saved to ', fileOut)
# Generate some text (facultatif car model sauvegardé après entrainement)

print('-----Generating text-----')
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist(), vocab))
