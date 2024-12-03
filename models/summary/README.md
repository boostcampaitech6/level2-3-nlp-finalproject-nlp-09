## DEPENDENCIES
- OS : Ubuntu 20.04
- python : 3.8.10
- nvidia-driver : 535.161.07
- cuda : 12.2
- GPU : Tesla V100 32GB x 1

```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```
<!-- 
## Install Docker
- https://docs.docker.com/engine/install/ubuntu/ -->

## Run FastAPI for Depoly
```bash
python ./app/main.py
```

## Model Info
- base model : [OrionStarAI/Orion-14B-Chat](https://huggingface.co/OrionStarAI/Orion-14B-Chat)
- quantized model : [CurtisJeon/OrionStarAI-Orion-14B-Chat-4bit](https://huggingface.co/CurtisJeon/OrionStarAI-Orion-14B-Chat-4bit)

## Quick Tour
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "CurtisJeon/OrionStarAI-Orion-14B-Chat-4bit",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto",
)
model.config.use_cache = True

tokenizer = AutoTokenizer.from_pretrained("CurtisJeon/OrionStarAI-Orion-14B-Chat-4bit", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```
```python
prompt = """[명령어]
아래의 대화문을 읽고 나에 대한 일기를 작성하듯 구체적으로 요약하세요.
한글만 사용하세요. Use Only Korean!!
편안한 말투로 작성하세요.
있는 내용에만 근거하세요. 함부로 추론하면 불이익을 받습니다.
[대화문]
나: 오늘 하루가 슬펐어
나: 그냥 아무것도 하기 싫어
나: 왜 항상 세상은 나를 억까하는 걸까?
[요약문]
오늘은"""
# Generate Sample
outputs = model.generate(**tokenizer(prompt, return_tensors="pt"))
print(tokenizer.decode(outputs[0]))
```