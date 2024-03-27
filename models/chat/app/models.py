import torch
from transformers import (
    pipeline, BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM, 
    TextStreamer,
    StoppingCriteria, StoppingCriteriaList
)
from typing import List, Dict
from assistant_chats import get_random_first_chat, get_random_second_chat

# Chat Prompt
# """도우미는 사용자에게 긍정적이고 친근한 답변을 제공하며 공감합니다. 도우미는 사용자에게 다시 질문을 합니다.
# 대화의 흐름에 맞는 답변이 오면 좋습니다. 도우미는 반말로 대답해야 좋습니다. 도우미는 한국말만 사용해야 합니다! 영어를 사용하면 불이익을 받습니다!

# {history_text}###user: {message}
# ###assistant: """



class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


class ChatPipe:
    def __init__(
        self, model_name: str, 
        adapter_name: str=None, 
        streamer: bool=True
    ):
        """_summary_
            Chat Model Pipeline Class
                - build pipeline for chat model and diary summary
                - create chat templates that are used
        Args:
            model_name (str): model name or model path used for generation task (HuggingFace)
            adapter_name (str, optional): Adapter name or path (LoRA)
            streamer (bool, optional): Whether to use streamer or not. Defaults to True.
        """
        self.model_name = model_name
        self.adapter_name = adapter_name
        self.streamer = streamer
        self.model = None
        self.tokenizer = None
        
        if 'polyglot' in self.model_name:
            self.stop_word = "</끝>"
        else:
            self.stop_word = "</s>"
        
        self.build_pipeline()
        
    def __call__(self, message: str, history: list) -> str:
        # bad_prompt = self.bad_chat_prompt(message)
        # bad_result = self.pipe(bad_prompt, 5)
        bad_result = "좋음"
        if "나쁨" in bad_result:
            return "음... 너가 무슨 말을 하는 지 잘 모르겠어서 그런데 다시 말해줄 수 있을까?"
        else:
            if len(history) < 2:    # 첫 입력에 대해서
                prompt = self.first_chat_prompt(message)
                result = self.pipe(prompt, 10)
                if "평범" in result:
                    return get_random_first_chat()

            prompt = self.chat_prompt(message, history)
            text = self.pipe(prompt)
            return self.post_process(text)
        
    def post_process(self, text: str) -> str:
        text = text.replace(';','')
        text = text.replace(']', "")
        text = text.replace('user', '친구')
        text = text.replace('ooo','')
        text = text.replace('oo','')
        text = text.replace('000','')
        text = text.replace('00','')
        text = text.replace('</끝>', "")
        if '\n' in text:
            text = text.split('\n')[0]
        if '###' in text:
            text = text.split('###')[0]
        while text[0] == '.':
            text = text[1:]
        
        text = text.replace('ooo씨','친구')
        text = text.replace('oo씨','친구')
        text = text.replace('000씨','친구')
        text = text.replace('00씨','친구')
        
        return text.strip()
    
    def build_pipeline(self):
        """_summary_
            Build Text Generation Pipeline with adaptor
        Returns:
            pipeline: Text Generation Pipeline
        """
        compute_dtype = getattr(torch, "float16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        model.config.use_cache = True
        
        if self.adapter_name is not None:
            print("Load Adapter")
            model.load_adapter(self.adapter_name, "loaded")
            model.set_adapter("loaded")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
    
    def pipe(self, text:str, max_new_tokens:int=100):
        streamer = None
        if self.streamer:
            streamer = TextStreamer(self.tokenizer)
        
        if 'Orion' in self.model_name:
            stop_words_ids = [torch.LongTensor([35824, 50362, 51429]).to('cuda'), torch.LongTensor([2]).to('cuda'),]
        elif 'polyglot' in self.model_name:
            stop_words_ids = [torch.LongTensor([31, 18, 5568, 33]).to('cuda'), torch.LongTensor([2]).to('cuda'),]
        else:
            stop_words_ids = [torch.LongTensor([2]).to('cuda'),]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            # Generate
            generate_ids = self.model.generate(
                inputs.input_ids.cuda(),
                max_new_tokens=max_new_tokens,
                # temperature=0.9,
                # eos_token_id=2,
                # pad_token_id=2,
                # top_k=40,
                # top_p=0.95,
                repetition_penalty=1.5,
                # do_sample=True,
                # num_return_sequences=1,
                streamer=streamer,
                stopping_criteria=stopping_criteria,
            )
        generate_ids = generate_ids[:,len(inputs.input_ids[0]):]
        generated_answers = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].replace('</s>','')

        return generated_answers
    
    def first_chat_prompt(self, message: str) -> str:
        text = f""" 문장이 `평범` 한지 `특별`한지 판단하세요.
[s_start] 오늘 별일 없었어 [s_end] > 평범 {self.stop_word}
[s_start] 오늘 공연 보다 왔어! [s_end] > 특별 {self.stop_word}
[s_start] 오늘 그냥 하루종일 집에 있었어 [s_end] > 평범 {self.stop_word}
[s_start] 친구들이랑 게임했어 [s_end] > 특별 {self.stop_word}
[s_start] 딱히 [s_end] > 평범 {self.stop_word}
[s_start] {message} [s_end] >"""
        return text
    
    def bad_chat_prompt(self, message: str) -> str:
        text = f"""문장에 의미가 있으면 `좋음` 없으면 `나쁨`으로 표시하세요.
[s_start] 뷁 [s_end] > 나쁨 {self.stop_word}
[s_start] 오늘 공연 보다 왔어! [s_end] > 좋음 {self.stop_word}
[s_start] ㄲㄴㄷ [s_end] > 나쁨 {self.stop_word}
[s_start] 게임함 [s_end] > 좋음 {self.stop_word}
[s_start] 앙 기모찌 [s_end] > 나쁨 {self.stop_word}
[s_start] 딱히 [s_end] > 좋음 {self.stop_word}
[s_start] {message} [s_end] >"""
        return text
    
    def chat_prompt(self, message: str, history: List[Dict[str, str]]) -> str:
        history_text = ""
        for line in history:
            history_text += f"{'나' if line['role'] == 'user' else '친구'}: {line['content']}{self.stop_word if line['role'] == 'assistant' else ''}\n"
        text = f"""당신은 아래의 대화문을 매끄럽게 완성해야 합니다. 되도록 상대방을 존중하고 공감해주는 말투로 대답하세요. 한국어만 사용하세요. 문장은 짧을수록 좋습니다.

{history_text}나: {message}
친구: """
        return text
    
    def summary_prompt(self, history: List[Dict[str, str]]) -> str:
        history_text = ""
        for line in history:
            if line['role'] == 'user':
                history_text += f"나: {line['content']}\n"
        text = f"""[명령어]
아래의 대화문을 읽고 `나`에 대한 일기를 작성하듯 구체적으로 요약하세요.
있는 내용에만 근거하세요. 함부로 추론하면 불이익을 받습니다.
[대화문]
{history_text}[요약문]
오늘은"""

        return text