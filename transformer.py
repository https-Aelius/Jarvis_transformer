#from pyexpat import model

import pandas as pd
from datasets import load_dataset
import torch #using pytorch

# Load the dataset
dataset = load_dataset("ismaildlml/Jarvis-MCU-Dialogues")
df = dataset['train'].to_pandas()

#print(df.head()) Just checking if I have successfully imported the dataset from hf (hugging face)

#parameters
block_size = 8 #max context length for predictions
batch_size = 4 #no. independent sequences we proccess in parallel
max_iter = 10000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' #allows to run on GPU (uses SIMD/parallel proccessing)
eval_iter = 200
#---------------

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

torch.manual_seed(1337)

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


#import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #each token directly reads off logits for next token from lookup table
        self.token_embedding_table= nn.Embedding(vocab_size, vocab_size)

    #passing index into token embedding table
    def forward(self, idx, targets=None):
        #idx and targets are both (B,T) tensor of integers

        # Pytorch will arrange all of this into batch by time by channel tensor.
        # Batch = 4, Time = 8, Channel = vocab_size
        logits = self.token_embedding_table(idx) #(B,T,C)

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
            #get predictions
            logits, loss = self(idx)
            #focus only on last time step
            logits = logits[:, -1, :] #Becoming (B,C)
            #applying softmax to get probabilities
            probs = F.softmax(logits, dim=-1) #(B,C)
            #sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #appending sampled index to running seq
            idx=torch.cat((idx, idx_next), dim=1) #(B, T+1)
        return idx[0]

model = BigramLanguageModel(vocab_size)
#for cuda
for_cuda = model.to(device)


#PyTorch Optimiser
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
batch_size = 32

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

print("Loss is ", loss.item())
idx1 = torch.zeros((1,1), dtype=torch.long, device=device) #batch = 1, time = 1, (1 by 1 tensor) Datatype is int.
result = model.generate(idx1, max_new_tokens=500)
print("results are")
print(decode(result))
