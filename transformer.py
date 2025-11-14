import pandas as pd
from datasets import load_dataset
#import torch
import torch #using pytorch
import torch.nn as nn
from torch.nn import functional as F


# Load the dataset
dataset = load_dataset("ismaildlml/Jarvis-MCU-Dialogues")
df = dataset['train'].to_pandas()

#print(df.head()) Just checking if I have successfully imported the dataset from hf (hugging face)

#parameters
block_size = 64 #max context length for predictions
batch_size = 64 #no. independent sequences we proccess in parallel
max_iter = 8000
eval_interval = 400
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'  # Use Apple MPS
#device = 'cuda' if torch.cuda.is_available() else 'cpu' #allows to run on GPU (uses SIMD/parallel proccessing)
eval_iter = 100
no_embed=256 #no. embeddings
no_heads = 8
n_layer = 4
dropout = 0.15
#---------------

torch.manual_seed(1337)

# Combine both Tony Stark and Jarvis dialogue into a single text corpus
all_text = []
for idx, row in df.iterrows():
    all_text.append(row['Tony Stark'] + '\n')  # Tony Stark's line
    all_text.append(row['Jarvis'] + '\n')      # Jarvis's response

full_text = ''.join(all_text)


chars = sorted(list(set(full_text)))
#taking text...  Text is a sequence of characters in python.
#So when I call the set constructor on the text, im going to get the set of all the characters that occur in text
#then i call list on that set, to get an arbitary ordering. After this I sort this ordering.

vocab_size = len(chars) #possible elements of our sequences as our vocab size

#------ Encoder and Decoder -------
#this tokeniser is very very very simple, visit openAIs Tiktoken or Google's sentencepeice
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #encoder: takes in string and outputs integer
#decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take in integers and outputs string

#decoder for outside vocab range
def decode(l):
    # Handle both tensor and list inputs
    if torch.is_tensor(l):
        l = l.tolist()  # Convert tensor to list
    # Filter out any tokens that are out of vocab range
    valid_tokens = [i for i in l if i in itos]
    return ''.join([itos[i] for i in valid_tokens])


# --------- Tokenising the entire dataset----------
data = torch.tensor(encode(full_text))

#splitting data into train and validation split
n = int(0.9*len(data)) #first 90% will be train, rest validation
#splitting string
train_data = data[:n]
val_data = data[n:]


train_data[:block_size+1]



# ---------- data loader ----------
def get_batch(split):
    #generating a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data #gets us our data array
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    #for cuda device
    x, y = x.to(device), y.to(device)
    return x, y

#
@torch.no_grad() #context manager: everything inside this func we do not want back propogation
def estimate_loss():
    out={}
    model.eval()
    for split in['train', 'val']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
#calculating mean of loss over multiple iterations




# ------- single head self attention --------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(no_embed, head_size, bias=False)
        self.query = nn.Linear(no_embed, head_size, bias=False)
        self.value = nn.Linear(no_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # tril allows tokens communcation from all proceeding tokens before it

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        # affinities
        k = self.key(x)  # (B,T,16)
        q = self.query(x)  # (B,T,16)
        weights = q @ k.transpose(-2, -1)  * C**-0.5 # (B,T,16) @ (B,16,T) ---> (B,T,T)
        #C^-0.5 allows for scaled attention

        # using matrices: (batch matrix multiply to do weighted aggregation)
        # weights = torch.zeros((T,T)) #affinity between tokens
        #decoder block
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # only allowing tokens to talk to previous tokens
        weights = F.softmax(weights, dim=-1) # (B,T,C)
        weights = self.dropout(weights)

        #aggregating values
        v = self.value(x)
        out = weights @ v
        return out

#---- Multi Headed Self Attention ------
class MultiHeadAttention(nn.Module): #multiple heads of self-attention running in parallel
    def __init__(self, no_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(no_heads)])
        self.proj = nn.Linear(no_embed, no_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  #concatenate all outputs over channel dimension
        out = self.proj(out) #projection is linear transformation of outcome of this layer
        return out

class FeedForward(nn.Module):
    def __init__(self, no_embed): #once improve try implement dropout
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(no_embed, 4 * no_embed), #linear layer
            nn.ReLU(), #non-linearity
            nn.Linear(4 * no_embed, no_embed), #projection layer going back to residual pathway
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

# ---- Transformer Block: comunication followed by computation ------
class Block(nn.Module):
    def __init__ (self, no_embed, no_heads):
        super().__init__()
        head_size = no_embed // no_heads
        self.sa = MultiHeadAttention(no_heads, head_size)
        self.ffwd = FeedForward(no_embed)
        self.layerNorm1 = nn.LayerNorm(no_embed) #using layerNorms from PyTorch
        self.layerNorm2 = nn.LayerNorm(no_embed) #look at attention is all you need architecture

    def forward(self,x):
        # ----- residual connections ------
        #applying layerNorm on x before feeding into self-attention and feedforward
        x = x + self.sa(self.layerNorm1(x)) #forking off here (to do communication)
        x = x + self.ffwd(self.layerNorm2(x)) #forking off here (to do communication)
        return x

# ------ Language Model --------
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        #each token directly reads off logits for next token from lookup table
        self.token_embedding_table= nn.Embedding(vocab_size, no_embed)
        self.position_embedding_table= nn.Embedding(block_size, no_embed)
        #self.sa_head = MultiHeadAttention(4, no_embed//4) #4 heads of 8-dimensional self attention
        #self.ffwd=FeedForward(no_embed)
        """
        self.blocks = nn.Sequential(
            Block(no_embed, no_heads=4),
            Block(no_embed, no_heads=4),
            Block(no_embed, no_heads=4),
            nn.LayerNorm(no_embed), #look at attention is all you need architecture
        )
        """
        self.blocks = nn.Sequential(*[Block(no_embed, no_heads = no_heads) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(no_embed) # layer norm
        self.langMod_head = nn.Linear(no_embed, vocab_size)

    #passing index into token embedding table
    def forward(self, idx, targets=None):
        #idx and targets are both (B,T) tensor of integers

        B,T = idx.shape
        # Pytorch will arrange all of this into batch by time by channel tensor.
        # Batch = 4, Time = 8, Channel = vocab_size
        tok_embed = self.token_embedding_table(idx) #(B,T,C)
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) #integers of T -> -1 (T,C)
        x = tok_embed + pos_embed # (B,T,C) encoding tensors
        #x = self.sa_head(x) #feeding tensor to self attention head
        #x = self.ffwd(x) #(B,T,C)
        x = self.blocks(x) #(B,T,C)
        logits = self.langMod_head(x) #(B,T,vocab_size) #feeding tensor into decoder, giving us the logits

        if targets is None:
            loss = None

        else:
            #however cross entropy require (B,C,T) so we're going to restructure our logits
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            #restructure of targets
            targets = targets.view(B*T)

            #evaluating loss function by using cross entropy (from PyTorch)
            loss = F.cross_entropy(logits, targets) #measures quality of logits/prediction with respect to targets.

        return logits, loss


    def generate(self, idx, max_new_tokens):
        #idx is (B,T)
        for j in range(max_new_tokens):
            #if idx is more than block size than our pos embedding table will run out of scope
            idx_cond = idx[:,-block_size:] #this is cropping idx to the last block_size token
            #get predictions
            logits, loss = self(idx_cond)
            #focus only on last time step
            logits = logits[:, -1, :] #Becoming (B,C)
            #applying softmax to get probabilities
            probs = F.softmax(logits, dim=-1) #(B,C)
            #sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #appending sampled index to running seq
            idx=torch.cat((idx, idx_next), dim=1) #(B, T+1)
        return idx[0]

model = BigramLanguageModel()
#for cuda
for_cuda = model.to(device)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')


#PyTorch Optimiser
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# --------- Training Loop ----------
for steps in range(max_iter):

    #every once in a while eval loss on train and val sets
    if steps % eval_interval == 0:
        loss = estimate_loss()
        print(f"step: {steps}, loss: {loss}")
        print(f"training loss: {loss['train']:.4f}, val loss: {loss['val']:.4f}")
    #this code above links back to estimate_loss func

    #sampling batch of data
    xb,yb = get_batch('train')

    #eval of loss
    logits, loss = model(xb, yb)
    optimiser.zero_grad(set_to_none=True) #zeroing out gradients from previous step
    loss.backward() #get grad from all paramters
    optimiser.step() #using grad to update said paramters

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))





