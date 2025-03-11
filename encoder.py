import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# def custom(idx, x):
    
#     if (idx % 2) == 0:
#         const = np.pow(10000, idx/d_model)
#         return torch.sin(x / const)

#     else:
#         const = np.pow(10000, (idx - 1)/d_model)
#         return torch.cos(x / const)
    
    
# pos_embeddings = torch.stack([custom(idx, x) for idx, x in enumerate(pos_matrix)]).T

class MHA(nn.Module):
    def __init__(self, d_model, h, max_tokens):
        super().__init__()

        self.queries = nn.Parameter(torch.randn(d_model, d_model))
        self.keys = nn.Parameter(torch.randn(d_model, d_model))
        self.values = nn.Parameter(torch.randn(d_model, d_model))
        self.p_mat = nn.Parameter(torch.randn(d_model, d_model))
        self.h = h
        self.d_model = d_model
        self.max_tokens = max_tokens

    def forward(self, x):
        
        q = x @ self.queries # 5 * 200 * 512  broadcasting of multiplication takes place
        k = x @ self.keys
        v = x @ self.values

        q = q.reshape(-1, self.max_tokens, self.h, int(self.d_model/self.h))  # https://dzone.com/articles/reshaping-pytorch-tensors
        k = k.reshape(-1, self.max_tokens, self.h, int(self.d_model/self.h))   
        v = v.reshape(-1, self.max_tokens, self.h, int(self.d_model/self.h))           

        attn_scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)   # https://pytorch.org/docs/main/generated/torch.matmul.html

        attn_scores = attn_scores / math.sqrt(self.d_model / self.h) 

        attn_scores = F.softmax(attn_scores, dim = 3)

        # print(attn_scores.shape, v.shape)   
        result = attn_scores @ v.permute(0, 2, 1, 3)

        result = torch.cat([torch.squeeze(ele) for ele in torch.split(result, 1, dim = 1)], dim = 2)

        result = result @ self.p_mat  # broadcasting takes place

        return result
    

# a = torch.randn(3, 2, 4)
# making a Encoder NN 

class First_Encoder(nn.Module):

    def custom(self, idx, x):
    
        if (idx % 2) == 0:
            const = np.pow(10000, idx/self.d_model)
            return torch.sin(x / const)

        else:
            const = np.pow(10000, (idx - 1)/self.d_model)
            return torch.cos(x / const)
        
    def __init__(self, d_model, h):
        super().__init__()

        self.d_model = d_model
        self.h = h
        self.max_tokens = 200
        self.emb = nn.Embedding(202, d_model)



        pos_matrix = torch.tensor(list(range(self.max_tokens)))
        self.pos_matrix = pos_matrix.repeat(self.d_model, 1)
        self.pos_embeddings = torch.stack([self.custom(idx, x) for idx, x in enumerate(self.pos_matrix)]).T

        self.MHA = MHA(d_model, h, self.max_tokens)
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(28*28, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 10),
        # )

        self.layer_norm = nn.LayerNorm(self.d_model)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        )

    def forward(self, x):

        x = self.emb(x)

        x = torch.add(x,self.pos_embeddings[:self.max_tokens, :])

        mha_out = self.MHA(x)

        x += mha_out # adding residual connection

        x = self.layer_norm(x)

        relu_out = self.linear_relu_stack(x)

        x += relu_out

        x = self.layer_norm(x)

        return x
    
class N_Encoder(nn.Module):

    def custom(self, idx, x):
    
        if (idx % 2) == 0:
            const = np.pow(10000, idx/self.d_model)
            return torch.sin(x / const)

        else:
            const = np.pow(10000, (idx - 1)/self.d_model)
            return torch.cos(x / const)
        
    def __init__(self, d_model, h):
        super().__init__()

        self.d_model = d_model
        self.h = h
        self.max_tokens = 200

        self.MHA = MHA(d_model, h, self.max_tokens)
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(28*28, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 10),
        # )

        self.layer_norm = nn.LayerNorm(self.d_model)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512)
        )

    def forward(self, x):

        mha_out = self.MHA(x)

        x += mha_out # adding residual connection

        x = self.layer_norm(x)

        relu_out = self.linear_relu_stack(x)

        x += relu_out

        x = self.layer_norm(x)

        return x
    

class encoder_stack(nn.Module):

    def __init__(self, n, h, d_model):
        super().__init__()
        self.n = n
        self.h = h
        self.d_model = d_model
        self.first_enc = First_Encoder(self.d_model, self.h)
        self.enc_stack = torch.nn.Sequential(*torch.nn.ModuleList([N_Encoder(self.d_model, self.h) for i in range(self.n)]))


    def forward(self, x):

        x = self.first_enc(x)
        x = self.enc_stack(x)

        return x
