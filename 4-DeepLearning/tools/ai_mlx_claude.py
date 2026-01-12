import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW


class SimpleAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        self.q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        self.scale = (self.head_dim ** -0.5)

    def __call__(self, x):
        B, T, C = x.shape
        
        q = self.q(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        
        # Attention
        att = (q @ k.swapaxes(-1, -2)) * self.scale  # (B, n_head, T, T)
        att = att - mx.max(att, axis=-1, keepdims=True)  # Stabilisation
        att = mx.softmax(att, axis=-1)
        
        # Output
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = self.proj(y)
        
        return y


class SimpleMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.n_embd * 4
        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=False)

    def __call__(self, x):
        return self.w2(nn.relu(self.w1(x)))


class SimpleBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = SimpleAttention(config)
        self.mlp = SimpleMLP(config)

    def __call__(self, x):
        # ✅ Résidus avec scaling
        x = x + self.attn(x) * 0.1
        x = x + self.mlp(x) * 0.1
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.block_size, config.n_embd)
        
        # ✅ Très petite initialisation
        self.embed.weight = mx.random.normal((config.vocab_size, config.n_embd)) * 0.02
        self.pos_embed.weight = mx.random.normal((config.block_size, config.n_embd)) * 0.02
        
        self.blocks = [SimpleBlock(config) for _ in range(config.n_layers)]
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.head.weight = mx.random.normal((config.vocab_size, config.n_embd)) * 0.02

    def __call__(self, x):
        B, T = x.shape
        
        tok = self.embed(x)
        pos = self.pos_embed(mx.arange(T))
        x = tok + pos
        
        for block in self.blocks:
            x = block(x)
        
        logits = self.head(x)
        return logits


def loss_fn(model, x, y):
    logits = model(x)
    logits = logits.reshape(-1, logits.shape[-1])
    loss = mx.mean(nn.losses.cross_entropy(logits, y.reshape(-1)))
    return loss


class MakeDataset:
    def __init__(self, text_encode, train_ratio=0.9):
        full_data = mx.array(text_encode, dtype=mx.int32)
        n = int(train_ratio * len(full_data))
        self.train_data = full_data[:n]
        self.val_data = full_data[n:]
        print(f"Dataset : Train {len(self.train_data)} | Val {len(self.val_data)}")


class Config:
    n_embd: int = 64
    n_layers: int = 2
    n_head: int = 2
    vocab_size: int = 5000
    block_size: int = 128
    dropout: float = 0.0
    learning_rate: float = 5e-5
    batch_size: int = 16
    max_iters: int = 1000


class Trainer:
    def __init__(self, model, optimizer, dataset, config):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.config = config
        self.loss_and_grad = nn.value_and_grad(self.model, loss_fn)

    def get_batch(self, split="train"):
        data = self.dataset.train_data if split == "train" else self.dataset.val_data
        ix = mx.random.randint(0, len(data) - self.config.block_size, (self.config.batch_size,))
        ix_list = ix.tolist()
        x = mx.stack([data[i:i+self.config.block_size] for i in ix_list])
        y = mx.stack([data[i+1:i+self.config.block_size+1] for i in ix_list])
        return x, y

    def run(self, max_iters):
        print(f"Training (LR={self.config.learning_rate})\n")
        
        for step in range(max_iters):
            x, y = self.get_batch()
            loss, grads = self.loss_and_grad(self.model, x, y)
            
            loss_val = loss.item()
            
            # Check NaN
            if mx.isnan(loss).item():
                print(f"❌ NaN at step {step}")
                break
            
            # Clip gradients
            grad_norm = 0
            for g in grads.values():
                grad_norm += mx.sum(g ** 2).item()
            grad_norm = grad_norm ** 0.5
            
            if grad_norm > 1.0:
                scale = 1.0 / (grad_norm + 1e-8)
                grads = {k: v * scale for k, v in grads.items()}
            
            self.model.update(self.optimizer.apply_gradients(grads, self.model))
            mx.eval(loss, self.model.parameters())
            
            if step % 100 == 0:
                print(f"Step {step:4d} | Loss: {loss_val:.6f} | GradNorm: {grad_norm:.4f}")