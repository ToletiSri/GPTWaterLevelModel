import torch
import torch.nn as nn
from torch.nn import functional as F
import config as cfg

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(cfg.block_size, cfg.block_size)))

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTWaterLevelModel(nn.Module):

    def __init__(self, TotalWaterLevels=700):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(TotalWaterLevels, cfg.n_embd)
        self.position_embedding_table = nn.Linear(2, cfg.n_embd) #.double()  # 2 - latitude, longitude
        self.blocks = nn.Sequential(*[Block(cfg.n_embd, n_head=cfg.n_head) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd) # final layer norm
        self.lm_head = nn.Linear(cfg.n_embd, TotalWaterLevels)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

   

    def forward(self, idx, targets=None):
        #print(idx.shape)
        B, T, _ = idx.shape
        idx_wl = idx[:,:,2] 
        # idx_wl and targets are both (B,T) tensor of integers
        #print(idx_wl.shape)
        idx_latLong = idx[:, :, :2]         
        # change water level from 1.99 to 6.3, i.e get indices. Min water level - 1, max water level 7.99
        idx_wl = idx_wl*100 - 300       

        # Ensure idx_wl is in the range of indices
        min_wl, max_wl = 0, 799  # Mapping from original range (2 to 8.99)
        idx_wl = torch.round(idx_wl).long()
        idx_wl = torch.clamp(idx_wl, min=min_wl, max=max_wl)        

        # Ensure idx_latLong is of type double
        #idx_latLong = idx_latLong.double()
        #print(idx_latLong.shape, idx_latLong.dtype)

        tok_emb = self.token_embedding_table(idx_wl) # (B,T) --> (B,T,C)
        pos_emb = self.position_embedding_table(idx_latLong) # (B,T,2) --> (B,T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x.float()) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None            
        else:
            targets = targets*100 - 300
            targets = torch.round(targets).long()
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, currWaterLevel, max_next_values, next_coordinates):
        # idx is (B, T) array of indices in the current context
        for i in range(max_next_values):
            # crop idx to the last block_size tokens
            idx_cond = currWaterLevel[:, -cfg.block_size:,:]
            #print(idx_cond.shape)
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            #print(logits.shape)            
            logits = logits[:, -1, :] # becomes (B, C)
            #print(logits.shape)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Scaling predicted water level to fit the given range            
            water_level_next = (idx_next + 300)/100
            #print(water_level_next)
            #print(water_level_next.shape)
            
            # Extract the next co=ordinate at index i to get tensor N of shape [1, 1, 2]
            N = next_coordinates[:, i:i+1, :]
            water_level_next = torch.cat((N, water_level_next.unsqueeze(2)),dim=2)
            #print(water_level_next)
            #print(water_level_next.shape)
            #print(currWaterLevel.shape)
            # append sampled index to the running sequence
            currWaterLevel = torch.cat((currWaterLevel, water_level_next), dim=1) # (B, T+1, 3)
        return currWaterLevel
