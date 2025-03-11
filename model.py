from datasets import load_dataset

ds = load_dataset("cfilt/iitb-english-hindi")

vocab = []

for sen in ds['train']['translation']:
    word_list = sen['en'].split(' ')
    vocab.extend(word_list)


print(len(vocab))

