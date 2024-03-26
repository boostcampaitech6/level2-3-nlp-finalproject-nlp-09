from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

from models import ChatPipe

os.environ['HF_TOKEN'] = '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'

router = APIRouter()

summary_pipe = ChatPipe(
    "CurtisJeon/OrionStarAI-Orion-14B-Chat-4bit", 
    streamer=True
)

pipe = ChatPipe(
    "CurtisJeon/OrionStarAI-Orion-14B-Chat-4bit", 
    '/home/jhw/level2-3-nlp-finalproject-nlp-09/models/chat/trained/CurtisJeon-OrionStarAI-Orion-14B-Chat-4bit-20240326-014234', 
    streamer=True
)

class ChatRequest(BaseModel):
    generation_id: int
    query: str
    history: list # json list[dict]

class ChatResponse(BaseModel):
    generation_id: int
    query: str
    answer: str
    
class SummaryRequest(BaseModel):
    generation_id: int
    query: list # json list[dict]
    
class SummaryResponse(BaseModel):
    generation_id: int
    query: list
    answer: str


@router.post("/generate/")
async def generate_text(req: ChatRequest) -> ChatResponse:
    text = None
    try:
        text = pipe(req.query, req.history)
            
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Model is not initialized")
    except ValueError:
        raise HTTPException(status_code=400, detail="Input is not valid")
    except Exception as err:
        print(err)
        raise HTTPException(status_code=500, detail="Something went wrong")

    return ChatResponse(
        generation_id=req.generation_id,
        query=req.query,
        answer=text
    )

@router.post("/summary/")
async def summary_text(req: SummaryRequest) -> SummaryResponse:
    try:
        prompt = summary_pipe.summary_prompt(req.query)
        result = summary_pipe.pipe(prompt)
        text = summary_pipe.post_process(result)
        
    except RuntimeError:
        raise HTTPException(status_code=500, detail="Model is not initialized")
    except ValueError:
        raise HTTPException(status_code=400, detail="Input is not valid")
    except Exception as err:
        print(err)
        raise HTTPException(status_code=500, detail="Something went wrong")

    return SummaryResponse(
        generation_id=req.generation_id,
        query=req.query,
        answer=text.strip()
    )