## DEPENDENCIES
- OS : Ubuntu 20.04
- python : 3.10.6
- nvidia-driver : 535.129.03
- cuda : 12.2
- GPU : Tesla V100 32GB x 1
```python
pip install -r requirements.txt
```

## Model Info
### emotion classifier model
- base model: [klue/roberta-large](https://huggingface.co/klue/roberta-large)
- fine-tuning model: [bunoheb/emotion_classifier_detail](https://huggingface.co/bunoheb/emotion_classifier_detail)
- fine-tuning active learning model: [m2af/klue-roberta-large-detail_emotion-classifier-basic](https://huggingface.co/m2af/klue-roberta-large-detail_emotion-classifier-basic)

## Quick Tour
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from load_data import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = AutoModelForSequenceClassification.from_pretrained("m2af/klue-roberta-large-detail_emotion-classifier-basic")
model.to(device)

tokenizer = AutoTokenizer.from_pretrained("m2af/klue-roberta-large-detail_emotion-classifier-basic")

with open('num_to_detail.pkl','rb') as f:
    num_to_detail = pickle.load(f)

# text
tokenized_content = tokenized_dataset("오늘은 너무 즐거운 날이네요!", tokenizer)
outputs = model(
    input_ids = tokenized_content['input_ids'].to(device),
    attention_mask = tokenized_content['attention_mask'].to(device),
    # token_type_ids = tokenized_content['token_type_ids'].to(device),
    )
label = np.argmax(outputs[0], axis = -1}
print(num_to_detail[label])
```

## Data Settings
- Training Dataset : [m2af/ko-emotion-dataset](https://huggingface.co/datasets/m2af/ko-emotion-dataset)
- Active learning Dataset: [감성 대화 말뭉치](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=86)

