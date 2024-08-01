import torch

# hyperparameters
batch_size = 8 # how many independent sequences will we process in parallel?
block_size = 16 # what is the maximum context length for predictions?
max_iters = 1000
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 256
n_head = 6
n_layer = 6
dropout = 0.2
TotalLatitudes = 180000
TotalLongitudes = 360000
# ------------
