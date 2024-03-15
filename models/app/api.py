from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, \
    AutoModelForSequenceClassification, AutoModelForCausalLM, TextStreamer

router = APIRouter()

model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")
tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2', max_length=160)
streamer = TextStreamer(tokenizer)
pipe = pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    streamer=streamer,
    device=0,
    min_new_tokens=20,
    max_new_tokens=128,
    early_stopping=True,
    do_sample=True,
    eos_token_id=2,
    repetition_penalty=1.1,
    temperature=0.9,
    top_k=50,
    top_p=0.95,
)


class PredictionRequest(BaseModel):
    generation_id: int
    query: str

class PredictionResponse(BaseModel):
    generation_id: int
    query: str
    answer: str


@router.post("/generate/")
async def generate_text(req: PredictionRequest) -> PredictionResponse:
    try:
        result = pipe(req.query)
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Model is not initialized")
    except ValueError:
        raise HTTPException(status_code=400, detail="Input is not valid")
    except Exception:
        raise HTTPException(status_code=500, detail="Something went wrong")

    # Extract the predicted label and confidence score
    text = result[0]['generated_text']

    return PredictionResponse(
        generation_id=req.generation_id,
        query=req.query,
        answer=text
    )
