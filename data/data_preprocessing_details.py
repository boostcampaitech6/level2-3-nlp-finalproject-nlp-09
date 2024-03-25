import json
from collections import Counter
import pandas as pd

def change_emotion(text):
  if text == '사나움':
    text = '날카로움'
  if text == '발열':
    text = '타오름'
  if text == '통쾌함':
    text = '즐거움'
  return text

def change_email(text): # 라벨러를 식별할 수 있는 정보 마스킹
  if text == 'User1@gmail.com':
    text = 'A'
  elif text == 'User2@gmail.com':
    text = 'B'
  elif text == 'User3@gmail.com':
    text = 'C'
  elif text == 'User4@gmail.com':
    text = 'D'
  elif text == 'User5@gmail.com':
    text = 'E'
  else:
    text = ''
  return text

def change_name(text):
  if (text=='A') | (text=='C') | (text=='E'):
    return 'F'
  elif (text=='B') | (text=='D'):
    return 'M'
  else:
    return ''

def extract_annotation(data):
  result_list=[]
  for value in data:
      annotation_list = []

      # data 가져오기
      result_dict = value['data']
      try:
          result_dict.pop('Unnamed: 0',None)
      except:
          pass
      
      # annotation 가져오기
      for anno in value['annotations']:
        annotation_dict = dict()
        # email 가져오기
        if len(anno['result'])!=0:
          for res in anno['result']:
            if res['type']=='choices':
              if res['from_name']=='repr':
                main = res['value']['choices'][0]
              elif res['from_name']=='detail':
                detail = res['value']['choices'][0]
        name = change_email(anno['completed_by']['email'])
        gender = change_name(name)
        annotation_dict['annotator']=name
        annotation_dict['gender']=gender
        annotation_dict['emotion']=main+'_'+detail
        annotation_dict = dict(sorted(annotation_dict.items()))
        annotation_list.append(annotation_dict)
      result_dict['annotation']=annotation_list
      result_list.append(result_dict)
  return result_list
    
def main():
    
     # 라벨링 데이터 json 형태로 load(원천데이터)
     # annotated의 정보가 담겨있는 데이터
    file_path = "./annotated_data.json"
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # 원천 데이터 중 필요한 부분만 추출
    # 5명이 라벨러한 데이터 감정 가져오기(성별 포함)
    result = extract_annotation(data)
    df = pd.DataFrame(result)
    
    # 결측치 제거
    df = df.dropna()
    
    # url 정보 제거
    df = df.drop('url', axis=1)
    
    # date 열 이름 변경 : 직관적으로
    df = df.rename(columns={'date':'created_date'})
    
    df.to_csv('./labeled_data_details.csv',encoding='utf-8-sig')
    

if __name__ == "__main__":
    main()