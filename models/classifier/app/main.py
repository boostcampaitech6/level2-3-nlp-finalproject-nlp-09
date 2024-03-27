from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger
from sqlmodel import SQLModel

from api_test import router
from database import engine

from config import config
from dependencies import load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 데이터베이스 테이블 생성
    logger.info("Creating database tables")
    SQLModel.metadata.create_all(engine)
    
   
    
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(router)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
