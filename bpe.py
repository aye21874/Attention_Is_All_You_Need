from collections import defaultdict


class BPE:

    # Keep a merge rules list for tokenization
    


    def __init__(self, vocab) -> None:
        self.base_vocab = vocab
        self.merge_rules = []

    def merge(self, cnt, base_vocab):
        
        counter = defaultdict(int)

        # fill the counter with couplet values
        for ele, val in cnt.items():
            
            i = 0
            while i + 1 < len(ele):

                couple = ele[i] + ele[i + 1]

                counter[couple] += val

                i += 1

        # get the pair with the max value
        sorted_counter = dict(sorted(counter.items(), key=lambda item: item[1], reverse= True))
        max_pair = list(sorted_counter.items())[0]

        # add to the base_vocab
        base_vocab = base_vocab + [max_pair[0]]

        # merge rule learnt
        self.merge_rules += [max_pair[0]]

        # make merges to the cnt dict
        # modify dict while iterating over it: https://stackoverflow.com/questions/5384914/how-to-delete-items-from-a-dictionary-while-iterating-over-it

        for ele, val in list(cnt.items()):

            i = 0
            temp_ele = list(ele)
            while i + 1 < len(ele):
                couple = temp_ele[i] + temp_ele[i + 1]
                if couple == max_pair[0]:

                    del cnt[ele]

                    new_key =  temp_ele[:i] + [couple]

                    if i+2 < len(ele):
                        new_key += temp_ele[i+2:]

                    cnt[tuple(new_key)] = val
                    break

                i += 1

        return cnt, base_vocab

    def bpe_algo(self, corpus, n, lang = 'en'):

        # get the length of the base vocab
        x = len(self.base_vocab)
        cnt = defaultdict(int)


        for sen in corpus:
            for word in sen[lang].split():
                cnt[tuple(word)] += 1

        
        # keep growing base vocab until desired size is reached
        while x < n:
            cnt, self.base_vocab = self.merge(cnt, self.base_vocab)
            x += 1

        # print(cnt) 

        return self.base_vocab

    # corpus = {'train': {'translation':[{'en':'hug'}, {'en':'hug'}, {'en':'hug'}, {'en':'pug'}, {'en':'pug'}, {'en':'hugs'}, {'en':'bun'}]}}
    # base_vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
    #             'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 
    #             'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '!', '"', 
    #             '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', 
    #             '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', ' ']
    # n = 87

    # vocab = bpe_algo(corpus, base_vocab, n)
    # print(vocab)

    def tokenize(self, sen):

        '''Given the input sentence tokenize it into the vocabulary given and also apply the merge rules'''
    
        # splitting and adding unkown tokens
        res = []
        for word in sen:

            if word not in self.base_vocab:
                res.append('<unk>')

            else:
                res.append(word)
        
        if len(res) < 2:
            return res
        # merging using merging rules    
        key = True
        while key:

            i = 0
            while i + 1 < len(res):
                
                couple = res[i] + res[i + 1]
                if couple in self.merge_rules:
                    left = res[:i]
                    right = res[i+2:] if i+2 < len(res) else [] 
                    res =  left + [couple] + right 
                    key = True
                    break
                
                i += 1
                key = False

        return res

        
