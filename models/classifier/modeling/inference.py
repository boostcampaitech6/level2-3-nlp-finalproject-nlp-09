from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

from train import set_seed

def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(category,label):
    num_label = []
    with open(f'{category}_to_num.pkl', 'rb') as f: # ****** 여기에 dict_label_to_num.pkl 파일 위치 입력 ******
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
  
    return num_label


def load_test_dataset(dataset_dir, tokenizer):
    """
    test dataset을 불러온 후,
    tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)
    test_label = list(map(int,test_dataset['label'].values))
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return test_dataset['id'], tokenized_test, test_label

def main(args):
    set_seed(42)
    """
    지정된 형식과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device('cuda:0' torch.cuda.is_available() else 'cpu')
    # load tokenizer
    Tokenizer_NAME = "klue/roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)
    
    # load my model
    MODEL_NAME = args.model_dir
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.parameters
    model.to(device)
    
    # load test dataset
    test_dataset_dir = 'data/testdata.csv'
    test_id,test_dataset,test_label = load_test_dataset(test_dataset_dir,tokenizer)
    RE_test_dataset = RE_Dataset(test_dataset,test_label)
    
    # predict answer
    pred_answer, output_prob = inference(model, RE_test_dataset,device)
    pred_answer = num_to_label('detail',pred_answer)
    
    # make csv file with predicted answer
    output = pd.DataFrame({'id':test_id,'pred_label':pred_answer, 'probs':output_prob})
    output.to_csv("./result.csv", index = False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장
    
    print("---Finish!---")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # model dir
    parser.add_argument('--model_dir',type = str, default = "best_model")
    args = parser.parse_args()
    print(args)
    main(args)
    