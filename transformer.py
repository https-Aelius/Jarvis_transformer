import pandas as pd
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("ismaildlml/Jarvis-MCU-Dialogues")
df = dataset['train'].to_pandas()

#print(df.head()) Just checking if I have successfully imported the dataset from hf (hugging face)

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

#this tokeniser is very very very simple, visit openAIs Tiktoken or Google's sentencepeice
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] #encoder: takes in string and outputs integer
decode = lambda l: ''.join([itos[i] for i in l]) #decoder: take in integers and outputs string

#Tokenising the entire dataset:
import torch #using pytorch
data = torch.tensor(encode(full_text))

#splitting data into train and validation split
n = int(0.9*len(data)) #first 90% will be train, rest validation
#splitting string
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

torch.manual_seed(1337)
batch_size = 4 #no. independent sequences we proccess in parallel
block_size = 8 #max context length for predictions
def get_batch(split):
    #generating a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data #gets us our data array
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
xb, yb = get_batch('train')

print(xb)


