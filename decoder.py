import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class CHA(nn.Module):
    def __init__(self, d_model, h, max_tokens, enc_input):
        super().__init__()

        self.queries = nn.Parameter(torch.randn(d_model, d_model))
        self.keys = nn.Parameter(torch.randn(d_model, d_model))
        self.values = nn.Parameter(torch.randn(d_model, d_model))
        self.p_mat = nn.Parameter(torch.randn(d_model, d_model))
        self.h = h
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.y = enc_input

    def forward(self, x):
        
        q = x @ self.queries # 5 * 200 * 512  broadcasting of multiplication takes place
        k = self.y @ self.keys # keys and values are coming from encoder
        v = self.y @ self.values

        q = q.reshape(-1, self.max_tokens, self.h, int(self.d_model/self.h))  # https://dzone.com/articles/reshaping-pytorch-tensors
        k = k.reshape(-1, self.max_tokens, self.h, int(self.d_model/self.h))   
        v = v.reshape(-1, self.max_tokens, self.h, int(self.d_model/self.h))           

        attn_scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)   # https://pytorch.org/docs/main/generated/torch.matmul.html

        attn_scores = attn_scores / math.sqrt(self.d_model / self.h) 

        attn_scores = F.softmax(attn_scores, dim = -1)

        # masking operation with j > i set to 0

        # print(attn_scores.shape, v.shape)   
        result = attn_scores @ v.permute(0, 2, 1, 3)

        result = torch.cat([torch.squeeze(ele) for ele in torch.split(result, 1, dim = 1)], dim = -1)

        result = result @ self.p_mat  # broadcasting takes place

        return result

class MMHA(nn.Module):
    def __init__(self, d_model, h, max_tokens):
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.queries = nn.Parameter(torch.randn(d_model, d_model))
        self.keys = nn.Parameter(torch.randn(d_model, d_model))
        self.values = nn.Parameter(torch.randn(d_model, d_model))
        self.p_mat = nn.Parameter(torch.randn(d_model, d_model))
        self.h = h
        self.d_model = d_model
        self.max_tokens = max_tokens

        # create a n by n matrix
        matrix = torch.full((self.max_tokens, self.max_tokens), -float('inf'))
        self.mask = torch.triu(matrix, diagonal=1).to(self.device)

    def forward(self, x):
        
        q = x @ self.queries # 5 * 200 * 512  broadcasting of multiplication takes place
        k = x @ self.keys
        v = x @ self.values

        q = q.reshape(-1, self.max_tokens, self.h, int(self.d_model/self.h))  # https://dzone.com/articles/reshaping-pytorch-tensors
        k = k.reshape(-1, self.max_tokens, self.h, int(self.d_model/self.h))   
        v = v.reshape(-1, self.max_tokens, self.h, int(self.d_model/self.h))           

        attn_scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)   # https://pytorch.org/docs/main/generated/torch.matmul.html

        attn_scores = attn_scores / math.sqrt(self.d_model / self.h) 

        # add attention mask
        attn_scores = attn_scores + self.mask.to(self.device)
        attn_scores = F.softmax(attn_scores, dim = 3)

        # print(attn_scores.shape, v.shape)   
        result = attn_scores @ v.permute(0, 2, 1, 3)

        result = torch.cat([torch.squeeze(ele) for ele in torch.split(result, 1, dim = 1)], dim = -1)

        result = result @ self.p_mat  # broadcasting takes place

        return result

class First_Decoder(nn.Module):

    def custom(self, idx, x):
    
        if (idx % 2) == 0:
            const = np.pow(10000, idx/self.d_model)
            return torch.sin(x / const)

        else:
            const = np.pow(10000, (idx - 1)/self.d_model)
            return torch.cos(x / const)
        
    def __init__(self, d_model, h, enc_input):
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.d_model = d_model
        self.h = h
        self.max_tokens = 200
        self.emb = nn.Embedding(204, d_model)
        self.enc_input = enc_input


        pos_matrix = torch.tensor(list(range(self.max_tokens)))
        self.pos_matrix = pos_matrix.repeat(self.d_model, 1)

        self.pos_embeddings = (torch.stack([self.custom(idx, x) for idx, x in enumerate(self.pos_matrix)]).T).to(self.device)

        self.MMHA = MMHA(d_model, h, self.max_tokens)  # make this into cross headed attention
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(28*28, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 10),
        # )

        self.CHA = CHA(d_model, h, self.max_tokens, enc_input)

        self.layer_norm = nn.LayerNorm(self.d_model)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        )

    def forward(self, x):

        x = self.emb(x)

        x = torch.add(x,self.pos_embeddings[:self.max_tokens, :])

        mha_out = self.MMHA(x)  # make a masked multi headed attention

        x = x + mha_out # adding residual connection

        x = self.layer_norm(x)

        # add the cross attention from the encoder layer here
        cha_out = self.CHA(x)
        
        x = x + cha_out

        x = self.layer_norm(x)

        relu_out = self.linear_relu_stack(x)

        x = x + relu_out
        
        x = self.layer_norm(x)

        return x

class N_Decoder(nn.Module):

    def custom(self, idx, x):
    
        if (idx % 2) == 0:
            const = np.pow(10000, idx/self.d_model)
            return torch.sin(x / const)

        else:
            const = np.pow(10000, (idx - 1)/self.d_model)
            return torch.cos(x / const)
        
    def __init__(self, d_model, h, enc_input):
        super().__init__()

        self.d_model = d_model
        self.h = h
        self.max_tokens = 200
        self.emb = nn.Embedding(204, d_model)
        self.enc_input = enc_input


        pos_matrix = torch.tensor(list(range(self.max_tokens)))
        self.pos_matrix = pos_matrix.repeat(self.d_model, 1)
        
        # self.pos_embeddings = (torch.stack([self.custom(idx, x) for idx, x in enumerate(self.pos_matrix)]).T).to(self.device)

        self.MMHA = MMHA(d_model, h, self.max_tokens)  # make this into cross headed attention

        self.CHA = CHA(d_model, h, self.max_tokens, enc_input)

        self.layer_norm = nn.LayerNorm(self.d_model)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        )

    def forward(self, x):

        mha_out = self.MMHA(x)  # make a masked multi headed attention

        x = x + mha_out # adding residual connection

        x = self.layer_norm(x)

        # add the cross attention from the encoder layer here
        cha_out = self.CHA(x)
        
        x = x + cha_out

        x = self.layer_norm(x)

        relu_out = self.linear_relu_stack(x)

        x = x + relu_out
        
        x = self.layer_norm(x)

        return x
    

class decoder_stack(nn.Module):

    def __init__(self, n, h, d_model):
        super().__init__()
        self.n = n
        self.h = h
        self.d_model = d_model
        self.final_linear = nn.Linear(d_model, 204)
        self.final_softmax = nn.Softmax(dim = -1)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        else:
            self.device = torch.device("cpu")
            print("No GPU available, using CPU.")

    def forward(self, x, enc_input):

        self.first_dec = First_Decoder(self.d_model, self.h, enc_input).to(self.device)
        self.dec_stack = torch.nn.Sequential(*torch.nn.ModuleList([N_Decoder(self.d_model, self.h, enc_input).to(self.device) for i in range(self.n)]))

        x = self.first_dec(x)
        x = self.dec_stack(x)
        x = self.final_linear(x)
        x = self.final_softmax(x)

        return x
