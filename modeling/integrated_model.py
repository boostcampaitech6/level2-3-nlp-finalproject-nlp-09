import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import random
import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, TrainingArguments,EarlyStoppingCallback

class IntegratedModel(nn.Module):
    def __init__(self,models):
        super(IntegratedModel, self).__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, input_ids, attention_mask = None):
        outputs = []
        for model in self.models:
            output = model(input_ids = input_ids, attention_mask = attention_mask)
            outputs.append(output)
        outputs = torch.stack(outputs, dim = 0)
        return torch.mean(outputs, dim = 0)

models = []

for i in range(5):
    model = AutoModelForSequenceClassification.from_pretrained(f"HUGGING_FACE_MODEL_{i}")
    models.append(model)
    
total_model = IntegratedModel(models)

    