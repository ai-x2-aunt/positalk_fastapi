from fastapi import FastAPI, UploadFile
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import List

# Hugging Face 모델과 tokenizer 로드
model_name = "Qwen/Qwen1.5-4B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# text generation pipeline을 사용
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

app = FastAPI()

@app.post("/generate_text/")
async def generate_text(messages: List[dict]):
    try:
        # 입력 받은 메시지로 텍스트 생성
        result = pipe(messages)
        
        # 생성된 텍스트 반환
        return {
            "generated_text": result[0]["generated_text"]
        }

    except Exception as e:
        # 오류 발생 시 오류 메시지 반환
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
