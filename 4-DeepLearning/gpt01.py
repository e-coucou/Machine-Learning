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
print(f"PyTorch location: {torch.__file__}")
#---------------------------------------------------------------
# Hyperparameters
torch.manual_seed(1965)
batch_size = 64 #32 # how many independent sequences will we process in parallel
block_size = 256 #8  # what is the maximum context length for predictions
max_iters = 7000 # number of training iterations
eval_interval = 500 # interval for evaluating the loss
eval_iters = 200 # number of iterations for loss estimation
learning_rate = 3e-4 # learning rate for the optimizer
n_embd = 384 #32 # embedding dimension
num_heads = 6 #4 # number of attention heads
n_layers = 6 #3 # number of transformer blocks
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
with open('data/bible.txt', 'r', encoding='utf-8') as f:
    _text = f.read()
text = preprocess_text(_text)
print(f"Longueur du texte après nettoyage: {len(text):,} caractères")
# Create character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Nombre de caractères uniques :', vocab_size)
print('Liste des caractères uniques :', ''.join(chars))
# Create mappings from characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
# split the data into training and validation sets
data = torch.tensor(encode(text), dtype=torch.long)
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
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
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

# Instantiate the model and optimizer
model = BigramLanguageModeler().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# Training loop
start_time = time.time()
# Initialiser les listes pour stocker l'historique
train_losses = []
val_losses = []
steps_recorded = []

for steps in range(max_iters):

    if (steps % eval_interval == 0) | (steps == max_iters - 1):
        losses = estimate_loss() # dictionnary with train and val losses
        elapsed = time.time() - start_time
        
        # Estimation du temps restant
        if steps > 0:
            time_per_step = elapsed / steps
            remaining_steps = max_iters - steps
            eta = time_per_step * remaining_steps
            print(f"step {steps}/{max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | {elapsed:.1f}s elapsed, ETA: {eta:.1f}s")
        else:
            print(f"step {steps}/{max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

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
}, 'bigram_full_model.pth')
print('Model saved to bigram_full_model.pth')
# Generate some text (facultatif car model sauvegardé après entrainement)
"""
print('-----Generating text-----')
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
"""