## DEPENDENCIES
- OS : Ubuntu 20.04
- python : 3.8.10
- nvidia-driver : 535.161.07
- cuda : 12.2
- GPU : Tesla V100 32GB x 1
```bash
pip install -r requirements.txt
```

## Model Info
- base model : [OrionStarAI/Orion-14B-Base](https://huggingface.co/OrionStarAI/Orion-14B-Base)
- quantized model : [CurtisJeon/OrionStarAI-Orion-14B-Base-4bit](https://huggingface.co/CurtisJeon/OrionStarAI-Orion-14B-Base-4bit)
- fine-tuned adapter : []()

## Quick Tour
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name  = "CurtisJeon/OrionStarAI-Orion-14B-Base-4bit"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto",
)
model.config.use_cache = True
    
model.load_adapter("./models/best_adaptor", "loaded")
model.set_adapter("loaded")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Generate Sample
outputs = model.generate(**tokenizer("안녕하세요, 반갑습니다.", return_tensors="pt"))
print(outputs)
```


## Dataset Settings
I used two datasets from AIHUB
- DatasetA: [감성 대화 말뭉치](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=86)
  - Convertion to parquet type is needed for this dataset
- DatasetB: [공감형 대화](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71305)
  - Unzip all the files with .json

## Config Settings
Hyperparameters are loaded from `./config.yaml`

## Training
```bash
python ./modeling/train.py
```

## Inference
```
python ./modeling/inference.py
```