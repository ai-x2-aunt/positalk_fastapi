from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://192.168.2.1:3000"],  # React 앱의 주소
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 모델 초기화 함수
def init_model():
    model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # pad_token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    return tokenizer, model

# 전역 변수로 모델과 토크나이저 설정
tokenizer, model = init_model()

class TransformRequest(BaseModel):
    text: str
    style: str

# 메시지 형식 정의
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.post("/generate_text")
async def transform_text(request: TransformRequest):
    try:
        # system 메시지와 user 메시지를 함께 설정
        messages = [
            {
                "role": "system",
                "content": "당신은 모든 말을 존댓말로 바꿔주는 AI 어시스턴트입니다. 사용자의 질문에 정확하고 도움되는 답변을 제공합니다."
            },
            {
                "role": "user",
                "content": request.text
            }
        ]
        
        # 나머지 코드는 동일
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # attention_mask 생성
        attention_mask = torch.ones_like(inputs)
        inputs = inputs.to(model.device)
        attention_mask = attention_mask.to(model.device)

        # 텍스트 생성
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(
            outputs[0][inputs.shape[-1]:], 
            skip_special_tokens=True
        )

        return {"generated_text": generated_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
