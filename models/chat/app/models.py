import torch
from transformers import (
    pipeline, 
    AutoTokenizer,
    AutoModelForCausalLM, 
    TextStreamer,
    StoppingCriteria, StoppingCriteriaList
)
from typing import List, Dict
from assistant_chats import get_random_first_chat, get_random_second_chat


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
        if '\n' in text:
            text = text.split('\n')[0]
        if '###' in text:
            text = text.split('###')[0]
        return text.strip()
    
    def build_pipeline(self):
        """_summary_
            Build Text Generation Pipeline with adaptor
        Returns:
            pipeline: Text Generation Pipeline
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
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
        if self.streamer:
            streamer = TextStreamer(self.tokenizer)
            
        stop_words_ids = [torch.LongTensor([35824, 50362, 51429]).to('cuda'), torch.LongTensor([2]).to('cuda'),]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            # Generate
            generate_ids = self.model.generate(
                inputs.input_ids.cuda(),
                max_new_tokens=max_new_tokens,
                temperature=0.9,
                eos_token_id=2,
                pad_token_id=2,
                top_k=40,
                top_p=0.95,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1,
                streamer=streamer,
                stopping_criteria=stopping_criteria,
            )
        generate_ids = generate_ids[:,len(inputs.input_ids[0]):]
        generated_answers = self.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].replace('</s>','')

        return generated_answers
    
    def first_chat_prompt(self, message: str) -> str:
        text = f""" 문장이 `평범` 한지 `특별`한지 판단하세요.
[s_start] 오늘 별일 없었어 [s_end] > 평범 </s>
[s_start] 오늘 공연 보다 왔어! [s_end] > 특별 </s>
[s_start] 오늘 그냥 하루종일 집에 있었어 [s_end] > 평범 </s>
[s_start] 친구들이랑 게임했어 [s_end] > 특별 </s>
[s_start] 딱히 [s_end] > 평범 </s>

[s_start] {message} [s_end] >"""
        return text
    
    def bad_chat_prompt(self, message: str) -> str:
        text = f"""문장에 의미가 있으면 `좋음` 없으면 `나쁨`으로 표시하세요.
[s_start] 뷁 [s_end] > 나쁨 </s>
[s_start] 오늘 공연 보다 왔어! [s_end] > 좋음 </s>
[s_start] ㄲㄴㄷ [s_end] > 나쁨 </s>
[s_start] 게임함 [s_end] > 좋음 </s>
[s_start] 앙 기모찌 [s_end] > 나쁨 </s>
[s_start] 딱히 [s_end] > 좋음 </s>

[s_start] {message} [s_end] >"""
        return text
    
    def chat_prompt(self, message: str, history: List[Dict[str, str]]) -> str:
        history_text = ""
        for line in history:
            history_text += f"###{line['role']}: {line['content']}{'</s>' if line['role'] == 'assistant' else ''}\n"
        text = f"""당신은 user에게 긍정적이고 친근한 답변을 제공하는 chatbot assistant입니다.
assistant는 사용자의 말을 공감합니다. assistant는 user에게 다시 질문을 합니다. 대화의 흐름에 맞는 답변이 오면 좋습니다. assistant는 반말로 대답해야 좋습니다.

{history_text}
###user: {message}
###assistant: """
        return text
    
    def summary_prompt(self, history: List[Dict[str, str]]) -> str:
        history_text = ""
        for line in history:
            if line['role'] == 'user':
                history_text += f"나: {line['content']}\n"
        text = f"""아래의 대화문을 요약해주세요.

나: 아니 딱히 특별한 일은 없었어...
나: 그냥 평범한 하루 일상이었던 것 같아... 감정도 딱히 뭔가 느껴지는 건 없구
나: 지루하다... 너말대로 일상에 지루함을 느껴서 무기력해진 감이 없지 않아 있는 것 같아
[요약문]
오늘은 평소와 같은 하루 일상이었다. 그래서인지 일상에 지루함을 느껴서 무기력해진 감이 없지 않아 있었다.</s>

나: 오늘 세븐틴 콘서트 보고 왔어!
나: 너무 행복했어! 특히 내 최애가 나한테 인사하는 것 같았어!
나: 얼른 다음 콘서트가 또 열렸으면 좋겠다!!
[요약문]
오늘은 세븐틴 콘서트를 다녀왔다! 최애가 나한테 인사를 하는 것 같은 느낌을 받아 너무 행복했다! 다음 콘서트가 매우 기대된다!</s>


{history_text}
[요약문]
"""

        return text