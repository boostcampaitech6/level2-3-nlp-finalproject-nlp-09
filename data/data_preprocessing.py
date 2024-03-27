import json
from collections import Counter
import pandas as pd

def change_emotion(text): # 데이터 통합하는 함수
  if text == '사나움':
    text = '날카로움'
  elif text == '발열':
    text = '타오름'
  elif text == '통쾌함':
    text = '즐거움'
  return text

def extract_annotation(data):
    result_list=[]
    result_dict = {}
    
    for value in data:
        # 데이터 기본 정보 불러오기
        result_dict = value['data']
        try:
            result_dict.pop('Unnamed: 0',None)
        except:
            pass

        emotion_list = []
        for j, anno in enumerate(value['annotations']):

            if (anno['reviews']) and (anno['reviews'][0]['accepted'] == True): # 리뷰가 있을 경우
                if anno['reviews'][0]['previous_annotation_history_result']:
                    for emo in anno['reviews'][0]['previous_annotation_history_result']:
                        if emo['from_name'] == 'repr':
                            main = emo['value']['choices'][0]
                        elif emo['from_name'] == 'detail':
                            detail = emo['value']['choices'][0]
                if ('fixed_annotation_history_result' in anno['reviews'][0]) and (anno['reviews'][0]['fixed_annotation_history_result']):
                    for emo in anno['reviews'][0]['fixed_annotation_history_result']:
                        if emo['from_name'] == 'repr':
                            main = emo['value']['choices'][0]
                        elif emo['from_name'] == 'detail':
                            detail = emo['value']['choices'][0]
                break
            
            else: # 리뷰가 없을 경우
                for emo in anno['result']:
                    if emo['from_name'] == 'repr':
                        main_emotion = emo['value']['choices'][0]
                    elif emo['from_name'] == 'detail':
                        detail_emotion = emo['value']['choices'][0]
                emotion = f'{main_emotion}_{detail_emotion}'
                emotion_list.append(emotion)
        emotion_list = Counter(emotion_list)
        emotion = emotion_list.most_common(1)

        if emotion:
            main, detail = emotion[0][0].split('_')
        detail = change_emotion(detail)
        result_dict['main'] = main
        result_dict['detail'] = detail
        result_dict['agreement'] = value['agreement']

        result_list.append(result_dict)
    return result_list
        
    
def main():
    
     # 라벨링 데이터 json 형태로 load(원천데이터)
    file_path = "./annotated_data.json"
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # 원천 데이터 중 필요한 부분만 추출
    result = extract_annotation(data)
    
    # raw 데이터 형태로 변환 후, 저장
    df = pd.DataFrame(result)
    
    # url 정보 제거
    df = df.drop('url', axis=1)
    
    # date 열 이름 변경 : 직관적으로
    df = df.rename(columns={'date':'created_date'})
    
    
    df.to_csv('./labeled_data.csv',encoding='utf-8-sig')
    

if __name__ == "__main__":
    main()