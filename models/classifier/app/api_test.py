# POST 구현
from typing import List,Dict,Union
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch   
from inference import *
from preprocessing import *
from konlpy.tag import Kkma, Komoran, Hannanum, Okt, Mecab
 
router = APIRouter()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Model Load
# model_main = AutoModelForSequenceClassification.from_pretrained("bunoheb/emotion_classifier_main")
# model_main.to(device)
model = AutoModelForSequenceClassification.from_pretrained("m2af/klue-roberta-large-detail_emotion-classifier-basic")
model.to(device)
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

# num_to_label
# with open(r'/home/kgy/level2-3-nlp-finalproject-nlp-09/active_learning/num_to_main.pkl', 'rb') as f:
#     num_to_main = pickle.load(f)
with open(r'/home/kgy/level2-3-nlp-finalproject-nlp-09/models/classifier/modeling/num_to_detail.pkl','rb') as f:
    num_to_detail = pickle.load(f)


# predict
class PredictionRequest(BaseModel):
    summary: str
    #chat: list

class PredictionResponse(BaseModel):
    emotion: dict
    #word: list

@router.post("/predict/")
async def predict(request: PredictionRequest) -> PredictionResponse:
    model.eval()
    
    # summary input
    tokenized_content = tokenized_dataset(request.summary, tokenizer)
    outputs = model(
        input_ids = tokenized_content['input_ids'].to(device),
        attention_mask = tokenized_content['attention_mask'].to(device),
        # token_type_ids = tokenized_content['token_type_ids'].to(device),
    )
    logits = outputs[0]
    
    # 확률은 소수점 2자리까지만 표현
    prob = F.softmax(logits, dim = -1).detach().cpu().numpy()
    prob = [float(f"{number:.8f}") for number in prob[0]]
    
    top3_prob = sorted(prob, reverse = True)[:3]
    top3_index = [prob.index(x) for x in top3_prob]
    
    top3_emotion = [num_to_detail[i] for i in top3_index]
    top3_prob = [float(f"{number:.2f}") for number in top3_prob]
       
    inference_result_value = {
        'emotion1':[top3_emotion[0],top3_prob[0]],
        'emotion2':[top3_emotion[1],top3_prob[1]],
        'emotion3':[top3_emotion[2],top3_prob[2]], 
        'otehrs': ["기타",1-sum(top3_prob)]
        
    }
    
    return PredictionResponse(emotion = inference_result_value)

# word 
okt = Okt()

class InputData(BaseModel):
    generation_id: int
    role: str
    content: str
 
class WordListRequest(BaseModel):
    chat: List[InputData]

class WordListResponse(BaseModel):
    word: list
    
@router.post("/word/")
async def word(request: WordListRequest) -> WordListResponse:
    user_contents = " ".join([chat.content for chat in request.chat if chat.role == 'user'])
    
    # 형태소 분석기
    text_pos = okt.pos(user_contents)
    extract_word = list(set([text[0] for text in text_pos if text[1] == "Noun" or text[1] == "Adjective"]))
      
    
    return WordListResponse(
        word = extract_word
    )
    
    