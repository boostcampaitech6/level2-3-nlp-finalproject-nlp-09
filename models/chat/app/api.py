from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os

from models import ChatPipe

router = APIRouter()

pipe = ChatPipe(
    "EleutherAI/polyglot-ko-5.8b", 
    'm2af/EleutherAI-polyglot-ko-5.8b-adapter', 
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
