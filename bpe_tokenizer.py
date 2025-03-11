from datasets import load_dataset
import torch
import string
from bpe import BPE

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

    return True


ds_filtered = ds.filter(custom_filter)


## Getting the Hindi tokenizer ready
start_unicode = 0x0900  # Start of Devanagari block
end_unicode = 0x097F  # End of Devanagari block


base_vocab = []
for codepoint in range(start_unicode, end_unicode + 1):
    char = chr(codepoint)
    base_vocab.append(char)

base_vocab = base_vocab + punctuation_list + [' ']

corpus = ds_filtered['train']['translation']
n = 200

bpe_hin_obj = BPE(base_vocab)
bpe_hin_obj.bpe_algo(corpus, n, lang = 'hi')


# Getting the English tokenizer ready

corpus = ds_filtered['train']['translation']
base_vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
              'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 
              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '!', '"', 
              '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', 
              '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ']

n = 200

bpe_en_obj = BPE(base_vocab)
bpe_en_obj.bpe_algo(corpus, n)












