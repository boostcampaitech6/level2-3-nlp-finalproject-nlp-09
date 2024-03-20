import pickle as pickle
import os
import pandas as pd
import torch

class RE_Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 class."""
    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def select(self, indices):
        return ({key: val[indices] for key, val in self.pair_dataset.items()},
                          self.labels[indices])
        
def preprocessing_dataset(dataset):
  """ 처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
  out_dataset = pd.DataFrame({'context':dataset['context'], 'main':dataset['main'], 'detail':dataset['detail']})
  return out_dataset

def load_data(dataset_dir):
  """ csv 파일을 경로에 맡게 불러 옵니다. """
  dataset = pd.read_csv(dataset_dir)
  dataset = preprocessing_dataset(dataset)
  
  return dataset

def tokenized_dataset(context, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  tokenized_context = tokenizer(context,return_tensors="pt", padding=True, truncation=True, max_length=256, add_special_tokens=True,)
  return tokenized_context