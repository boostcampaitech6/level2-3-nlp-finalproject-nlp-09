import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import json
import sklearn
import random
import os

from load_data import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments,EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import pickle as pickle
import wandb

def set_seed(seed:int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def klue_re_micro_f1_for_main(preds, labels):
    label_list = ['기쁨', '슬픔', '싫어함(상태)', '분노', '미움(상대방)', '두려움', '수치심', '욕망', '사랑', '중립']
    label_indices = list(range(len(label_list)))
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc_for_main(probs, labels):
    labels = np.eye(10)[labels]

    score = np.zeros((10,))
    for c in range(10):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def klue_re_micro_f1_for_detail(preds, labels):
    label_list = ['만족감', '무기력', '즐거움', '답답함', '타오름', '불쾌', '자랑스러움', '절망', '치사함', '걱정', '부끄러움', '궁금함', '놀람', '아쉬움', '싫증', '공감', '감동', '냉담', '경멸',
                  '매력적', '반가움', '불만', '실망', '미안함', '다정함', '공포', '억울함', '난처함', '날카로움', '불신감', '동정(슬픔)', '불편함', '아픔', '고마움', '호감', '귀중함', '기대감', '고통',
                  '수치심', '초조함', '원망', '위축감', '후회', '욕심', '시기심', '안정감', '너그러움', '외면', '그리움', '허망', '편안함', '신명남', '비위상함', '반감', '죄책감', '아른거림', '외로움',
                  '서먹함', '자신감', '두근거림', '심심함', '갈등', '신뢰감', '열정적인']
    label_indices = list(range(len(label_list)))
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc_for_detail(probs, labels):
    labels = np.eye(64)[labels]

    score = np.zeros((64,))
    for c in range(64):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics_for_main(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1_for_main(preds, labels)
  auprc = klue_re_auprc_for_main(probs, labels)
  acc = accuracy_score(labels, preds)

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def compute_metrics_for_detail(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1_for_detail(preds, labels)
  auprc = klue_re_auprc_for_detail(probs, labels)
  acc = accuracy_score(labels, preds)

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(category,label):
    num_label = []
    with open(f'{category}_to_num.pkl', 'rb') as f: # ****** 여기에 dict_label_to_num.pkl 파일 위치 입력 ******
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
  
    return num_label

  
def train():
    set_seed(42)
    # load model and tokenizer
    MODEL_NAME = 'klue/roberta-large'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = load_data('../main_aihub300.csv')
    
    main = label_to_num(main, train_dataset['main'].values)
    detail = label_to_num(detail, train_dataset['detail'].values)
    context = tokenized_dataset(train_dataset['context'].to_list(), tokenizer)
    
    # make dataset for pytorch
    RE_main_dataset = RE_Dataset(context, main)
    RE_detail_dataset = RE_Dataset(context, detail)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # setting hyperparameter
    model_config_for_main =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config_for_main.num_labels = len(train_dataset['main'].unique())

    model_config_for_detail =  AutoConfig.from_pretrained(MODEL_NAME)
    model_config_for_detail.num_labels = len(train_dataset['detail'].unique())
    
    training_args_main = TrainingArguments(
        output_dir='./results_main',          # output directory
        #save_steps=500,                 # model saving step.
        num_train_epochs=20,              # total number of training epochs
        learning_rate=1e-5,               # learning_rate
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        evaluation_strategy='epoch', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        logging_steps=1,              # log saving step.
        #eval_steps = 500,            # evaluation step.
        save_strategy = 'epoch',
        load_best_model_at_end = True,
        save_total_limit=1,
        )
    
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        )
    
    # Kfold
    kf = KFold(n_splits= 5, shuffle= True, random_state= 42)
    
    
    # main_label

    for fold, (train_index, val_index) in enumerate(kf.split(RE_main_dataset)):
        print(f"Fold : {fold}")
        
        # model load
        model_for_main =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config_for_main)
        model_for_main.to(device)
        
        # data load
        train_dataset_fold = RE_main_dataset.select(train_index)
        val_dataset_fold = RE_main_dataset.select(val_index)
        
        # Trainer
        trainer_for_main = Trainer(
        model=model_for_main,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args_main,                  # training arguments, defined above
        train_dataset=train_dataset_fold,         # training dataset
        eval_dataset=val_dataset_fold,             # evaluation dataset
        compute_metrics=compute_metrics_for_main,         # define metrics function
        callbacks = [early_stopping],
        )
        
        # 모델 훈련
        trainer_for_main.train()
        
        # 모델 저장
        model_for_main.save_pretrained(f"pts/best_model_main_{fold}")
      
    # detail label
    training_args_detail = TrainingArguments(
        output_dir='./results_detail',          # output directory
        #save_steps=500,                 # model saving step.
        num_train_epochs=20,              # total number of training epochs
        learning_rate=1e-5,               # learning_rate
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        evaluation_strategy='epoch', # evaluation strategy to adopt during training
                                    # `no`: No evaluation during training.
                                    # `steps`: Evaluate every `eval_steps`.
                                    # `epoch`: Evaluate every end of epoch.
        logging_steps=1,              # log saving step.
        #eval_steps = 500,            # evaluation step.
        save_strategy = 'epoch',
        load_best_model_at_end = True,
        save_total_limit=1,
        )
    
    for fold, (train_index, val_index) in enumerate(kf.split(RE_main_dataset)):
        print(f"Fold : {fold}")
        
        # model load
        model_for_main =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config_for_main)
        model_for_main.to(device)
        
        # data load
        train_dataset_fold = RE_main_dataset.select(train_index)
        val_dataset_fold = RE_main_dataset.select(val_index)
        
        # Trainer
        trainer_for_main = Trainer(
        model=model_for_main,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args_main,                  # training arguments, defined above
        train_dataset=train_dataset_fold,         # training dataset
        eval_dataset=val_dataset_fold,             # evaluation dataset
        compute_metrics=compute_metrics_for_main,         # define metrics function
        callbacks = [early_stopping],
        )
        
        # 모델 훈련
        trainer_for_main.train()
        
        # 모델 저장
        model_for_main.save_pretrained(f"pts/best_model_main_{fold}")
      
    
def main():
    train()
    
if __name__ == "main":
    main()
            
        