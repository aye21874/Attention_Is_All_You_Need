import torch
import bpe_tokenizer as D
import string
from datasets import load_dataset
import torch
import torch.nn as nn
from decoder import decoder_stack
from encoder import encoder_stack
from torch.utils.data import DataLoader


print(torch.__version__)

ds = load_dataset("cfilt/iitb-english-hindi")

english_characters = list(string.ascii_lowercase) + list(string.ascii_uppercase)

punctuation_list = list(string.punctuation)

char_to_keep = english_characters + punctuation_list + [' ']

def custom_filter(example):

    for word in example['translation']['en']:
        if word not in char_to_keep:
            return False
        

    for word in example['translation']['hi']:
        if not ((ord(u'\u0900') <= ord(word) <= ord(u'\u097F') ) or (word in list(string.punctuation)) or (word == ' ')):
            return False
        
    # removed sentences greater than 90th percentile     
    if len(example['translation']['en']) > 161:
        return False
    
    if len(example['translation']['hi']) > 115:
        return False

    return True


ds_filtered = ds.filter(custom_filter)

max_tokens = 200

all_tokens = D.bpe_en_obj.base_vocab + ['<unk>', '<pad>']
word2idx_en = {}

for ind, ele in enumerate(all_tokens):
    word2idx_en[ele] = ind

all_tokens = D.bpe_hin_obj.base_vocab + ['<unk>', '<pad>', '<eos>', '<start>']
word2idx_hin = {}

for ind, ele in enumerate(all_tokens):
    word2idx_hin[ele] = ind

def tokenize_en(x):

    res = D.bpe_en_obj.tokenize(x)
    while len(res) < max_tokens:
        res.append('<pad>')

    
    return torch.tensor([word2idx_en[ele] for ele in res])

def tokenize_hin(x):

    res = D.bpe_hin_obj.tokenize(x)
    key = 0
    
    while len(res) < max_tokens:

        if not key:
            res.insert(0, '<start>')
            res.append('<eos>')
            key = 1
            continue

        res.append('<pad>')
    
    return torch.tensor([word2idx_hin[ele] for ele in res])


training_ds = ds_filtered['train']
iterable_dataset = training_ds.to_iterable_dataset()

def custom_mapper(x):
    
    en_tok = tokenize_en(x['translation']['en'])
    hi_tok = tokenize_hin(x['translation']['hi'])
    tar_tok = torch.roll(hi_tok, shifts=-1, dims=-1)
    tar_tok[-1] = 201
    return {'translation': {'en' : en_tok , 'hi' : hi_tok, 'tar': tar_tok}}

tokenized_iterable_dataset = iterable_dataset.map(lambda input: custom_mapper(input))

# training_ds = ds_filtered['train'].map(custom_mapper)  # No to_iterable_dataset()
# dataloader = DataLoader(training_ds, batch_size=32)

if torch.cuda.is_available():
    device = torch.device("cuda")
    # print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
else:
    device = torch.device("cpu")
    # print("No GPU available, using CPU.")

class Transformer_MT(nn.Module):

    def __init__(self):

        super().__init__()

        self.device = device
        self.enc = encoder_stack(4, 4, 512).to(self.device)
    
    def forward(self, enc_input, dec_input):

        enc_output = self.enc(enc_input)
        self.dec = decoder_stack(4, 4, 512).to(self.device)

        output = self.dec(dec_input, enc_output)

        output = output.reshape(-1, 204)
    
        return output
    
model = Transformer_MT()
learning_rate = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss = torch.nn.CrossEntropyLoss()
dataloader = DataLoader(tokenized_iterable_dataset, batch_size=32)

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    batch_size = 32
    # size = 
    for batch, X in enumerate(dataloader):
        # print(batch, X['translation']['en'][0], X['translation']['tar'][0])

        optimizer.zero_grad()

        inputs_1 = X['translation']['en'].to(device)
        inputs_2 = X['translation']['hi'].to(device)
        model_output = model(inputs_1, inputs_2)

        target = X['translation']['tar'].reshape(-1)
        target = target.to(device)
        
        loss = loss_fn(model_output, target)
        
        # print(loss)
        # print(model_output.shape)

        loss.backward()
        optimizer.step()
        # print(f'current batch{batch}')    

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}, current: {current:>7f}" )
            # torch.mps.empty_cache()

model = model.to(device)

epochs = 10

for t in range(epochs):

    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dataloader, model, loss, optimizer)

    torch.save({
            'epoch': t,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, f'checkpoint_{t}.pth')
    
print("Done!") 