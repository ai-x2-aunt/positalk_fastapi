from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai_api import OpenAIHandler  # openai_api.py 임포트

# 환경변수 로드
load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI 핸들러 초기화
openai_handler = OpenAIHandler()

class ChatRequest(BaseModel):
    message: str
    style: str  # 스타일 파라미터 추가

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = openai_handler.get_completion(request.message, request.style)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)} 