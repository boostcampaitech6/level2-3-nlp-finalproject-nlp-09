import os
import json
from glob import glob
from tqdm.auto import tqdm

import pandas as pd
from datasets import Dataset

EOS_TOKEN = '</끝>'


def read_json(file):
    """_summary_
        load json for DatasetB (감성대화셋)
    Args:
        file (str or PathLike): file_path of json

    Returns:
        Dict, List: Python builtin class
    """
    with open(file, 'rb') as f:
        data = json.load(f)
    return data


def formatting_prompts_func(examples):
    """_summary_
        formatting func for trainer
    Args:
        examples (Union[dict, Dataset]): examples of row

    Returns:
        List[str]: formatted prompts
    """
    output_texts = []
    for conversation in examples['conversation']:
        texts = []
        for line in conversation:
            text = f"{line['role']}: {line['content']}{EOS_TOKEN if line['role']!='user' else ''}"
            texts.append(text)
        output_texts.append("\n".join(texts))
    return output_texts


class DatasetA:
    def __init__(self, path:str ='./data/dataset.parquet'):
        """_summary_
            AIHUB 감성 대화 셋 loader & class
        Args:
            path (str): parquet file. Defaults to './data/dataset.parquet'.
        """
        self.path = path
        self.dataset = self.load_dataset()

    def load_dataset(self):
        train = pd.read_parquet(self.path)
        user_columns = ['사람문장1', '사람문장2', '사람문장3']
        conversations = []
        for _, row in tqdm(train.iterrows(), desc="Data Formatting", total=len(train)):
            for usr_col in user_columns:
                if row[usr_col] is not None:
                    user_chat_dict = {
                        "role" : "user",
                        "content" : row[usr_col]
                    }
                    sys_chat_dict = {
                        "role" : "assistant",
                        "content" : row[usr_col.replace("사람문장", "시스템문장")]
                    }
                    conversations.extend([user_chat_dict, sys_chat_dict])
        
        dataset = {
            "conversation" : conversations
        }

        dataset = Dataset.from_dict(dataset)
        return dataset


class DatasetB:    
    def __init__(self, file_paths: str ='./data/*/*/*.json'):
        """_summary_
            AIHUB 감성대화셋 loader & class
        Args:
            file_paths (str, optional): Files to get json by glob method. Defaults to './data/*/*/*.json'.
        """
        self.file_paths = file_paths
        self.files = glob(self.file_paths)
        self.dataset = self.make_trainset()

    def extract_train_data(self, data):
        relation = data['info']['relation']
        situation = data['info']['situation']
        behavior = data['info']['listener_behavior']
        conversation = [
            {
                'role' : '나' if x['role'] == 'speaker' else '친구',
                'content' : x['text'].replace('감정화자','너').replace('공감화자', '나')
            } for x in data['utterances']
        ]

        return relation, situation, behavior, conversation

    def make_trainset(self):
        relations = []
        situations = []
        behaviors = []
        conversations = []
        for file in tqdm(self.files, desc="Data Formatting",):
            data = read_json(file)
            if data['info']['relation'] in ['친구']:
                relation, situation, behavior, conversation = self.extract_train_data(data)
                relations.append(relation)
                situations.append(situation)
                behaviors.append(behavior)
                conversations.append(conversation)

        output = {
            'relation' : relations,
            'situation' : situations,
            'behavior' : behaviors,
            'conversation' : conversations
        }
        
        dataset = Dataset.from_dict(output)
        # dataset = Dataset.from_pandas(dataset.to_pandas().sample(50)) # for sampling
        return dataset