import torch

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria, StoppingCriteriaList, TextStreamer
)

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


def get_bnb_config(config):
    q_config = config['quantization']
    if q_config['use_4bit']:
        compute_dtype = getattr(torch, q_config['bnb_4bit_compute_dtype'])
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=q_config['use_4bit'],
            bnb_4bit_quant_type=q_config['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=q_config['use_nested_quant'],
        )
        return bnb_config
    else:
        return False

def get_model(config):
    t_config = config['train_params']

    model = AutoModelForCausalLM.from_pretrained(
        t_config['model_name'],
        quantization_config=get_bnb_config(config),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=t_config['device_map'],
    )
    model.config.use_cache = True
    
    if t_config["load_adapter"]:
        model.load_adapter(t_config["adapter_path"], "loaded")
        model.set_adapter("loaded")
    
    return model

def get_tokenizer(config):
    t_config = config['train_params']
    tokenizer = AutoTokenizer.from_pretrained(t_config['model_name'], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    return tokenizer


# orion [35824, 50362, 51429]
# polyglot [31, 18, 5568, 33]

def generate_reply(model, tokenizer, query):
    stop_words_ids = [torch.LongTensor([31, 18, 5568, 33]).to('cuda'), torch.LongTensor([2]).to('cuda'),]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)]
    )
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    model.eval()
    reformat_question = f"""당신은 user에게 긍정적이고 친근한 답변을 제공하는 chatbot assistant입니다.
assistant는 사용자의 말을 되풀이합니다. assistant는 user에게 질문을 합니다.

###user: {query}
###assistant: """
    inputs = tokenizer(reformat_question, add_special_tokens=True, return_tensors="pt")

    with torch.no_grad():
        # Generate
        generate_ids = model.generate(
        inputs.input_ids.cuda(),
        max_new_tokens=100,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
        )
    generated_answers = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, skip_prompt=True, clean_up_tokenization_spaces=False)[0]
    generated_answers = generated_answers.replace(reformat_question, "")
    return generated_answers


if __name__ == "__main__":
    from utils import load_config, set_env
    set_env()
    config = load_config('./modeling/config.yaml')
    
    model = get_model(config)
    tokenizer = get_tokenizer(config)
    
    reply = generate_reply(model, tokenizer, "오늘 너무 우울하다...ㅠㅠ")
    print("Generated:", reply)
    print("DONE")