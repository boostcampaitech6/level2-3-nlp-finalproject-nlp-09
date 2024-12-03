import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import json
import sklearn
import random
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TrainingArguments
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import pickle as pickle

def tokenized_dataset(context, tokenizer):
  """ tokenizer에 따라 sentence를 tokenizing 합니다."""
  tokenized_context = tokenizer(context,return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=True,)
  return tokenized_context

class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    return item

  def __len__(self):
    return len(self.labels)