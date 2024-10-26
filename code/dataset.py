import torch
from torch.utils.data import Dataset
from Bio import SeqIO
import pandas as pd
import transformers
from torch.utils.data import Sampler
import numpy as np

def read_fasta(path):
    sequences = list(SeqIO.parse(path, format="fasta"))
    res = []
    for sequence in sequences:
        res.append(str(sequence.seq))
    return res

#maxlength=200

def tokenizer_new(seq):
    table = ["[PAD]","[CLS]","[SEP]","A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
    tokens = [0]*202
    attention_mask = [0]*202
    
    for i,x in enumerate(seq[0:len(seq)]):
        if i < 200:
            tokens[i+1] = table.index(x)
            # tokens[i]=table.index
            # table_id.append(table.index(x))
            attention_mask[i] = 1
    # print("t",table_id,'token_id',tokens)
    tokens[0]=table.index("[CLS]")
    tokens[i+2]=table.index("[SEP]")
    attention_mask[i+1]=1
    attention_mask[i+2]=1
    return tokens, attention_mask

def add_spaces(df_col):
    return " ".join(df_col)

class PlantAMPDataset(Dataset):
    def __init__(self,path,tokenizer=None):
        self.tokenized_seqs = []
        self.attention_masks = []
        self.labels = []
        
        sequences_pd=pd.read_csv(path)
        for id,df in sequences_pd.iterrows():
            # tokens,attention_mask=tokenizer(df["Sequence"])
            # Making a list of sequences from the df:
            seqs_list =df.Sequence
            # Adding spaces in between sequence letters (amino acids):
            seqs_spaced = add_spaces(seqs_list)
            # print(len(seqs_spaced))
            if tokenizer==None:
                tokens,attention_mask=tokenizer_new(df["Sequence"])
            else:
                 # ID list tokenized:
                if type(tokenizer) == transformers.models.t5.tokenization_t5.T5Tokenizer:
                    inputs = tokenizer(seqs_spaced, add_special_tokens=True, padding = 'max_length', max_length = 201)
                else:
                    inputs = tokenizer(seqs_spaced, add_special_tokens=True, padding = 'max_length', max_length = 202)
                # Retrieving the input IDs and mask for attention as tensors:
                tokens = inputs['input_ids']
                attention_mask = inputs['attention_mask']

                
            self.tokenized_seqs.append(tokens)
            self.attention_masks.append(attention_mask)
            if(df["Label"]==1):
                # self.labels.append([1, 0])
                # self.labels.append([0, 1])
                self.labels.append(df["Label"])
            else:
                # self.labels.append([0 ,1])
                # self.labels.append([1 ,0])
                self.labels.append(df["Label"])
            

    def __getitem__(self, index):
        item = {}
        item['input_ids'] = torch.tensor(self.tokenized_seqs[index])
        item['attention_mask'] = torch.tensor(self.attention_masks[index])
        item['labels'] = torch.as_tensor(self.labels[index]).long()
        # torch.Tensor(self.labels[index])   
        return item

    def __len__(self):
        return len(self.labels)
    
class LabelBalancedBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.labels =  torch.as_tensor(data_source.labels).long() # 所有样本的标签
        self.batch_size = batch_size # 每次在同一个标签中取多少个作为一组
        self.unique_labels = torch.unique(self.labels) # 标签集
        self.label_indices = {label.item(): (self.labels == label).nonzero().view(-1).tolist() 
                              for label in self.unique_labels} # 标签字典
        self.n_normal = self.batch_size // 2
        self.n_outlier = self.batch_size - self.n_normal
        
        
        
    @staticmethod
    def random_generator(idx_list):
        while True:
            random_list = np.random.permutation(idx_list)
            for i in random_list:
                yield i
 
    def __iter__(self):
        for label, idx in self.label_indices.items(): # 对每个标签和对应的样本
#             for i in range(0, len(idx), self.batch_size): # 每个标签分批采样(batch_size一组)
#                  batch = idx[i : i + self.batch_size]
#                  yield batch
            if label==0:
                self.normal_generator = self.random_generator(idx)
            else:
                self.outlier_generator = self.random_generator(idx)
        for _ in range(self.__len__()):
            batch = []
            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            for _ in range(self.n_outlier):
                batch.append(next(self.outlier_generator))
            yield np.random.permutation(batch)

    def __len__(self): # batch 的个数
        n = 0
        for label, idx in self.label_indices.items():
            n += (len(idx) + self.batch_size - 1) // self.batch_size
        return n
    