import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import asyncio
from typing import Optional
import concurrent.futures

class HuggingFaceHandler:
    def __init__(self):
        print("=== HuggingFace 모델 초기화 시작 ===")
        self.model_name = "EleutherAI/polyglot-ko-5.8b"
        
        print("1. 토크나이저 로딩 중...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        print("✓ 토크나이저 로딩 완료")
        
        print("2. GPU 메모리 정리 중...")
        torch.cuda.empty_cache()
        print("✓ GPU 메모리 정리 완료")
        
        print("3. 모델 로딩 중... (1-3분 소요)")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✓ 모델 로딩 완료")
        
        self.model_loaded = True
        print("=== 초기화 완료! 서비스 준비됨 ===")
            
        self.model_loaded = True
        self.emojis = ['💕', '✨', '🥺', '😊', '💝', '🌸', '💗', '💖']
        
        # 추론 타임아웃 설정 (초)
        self.inference_timeout = 300   # 60초로 수정

    async def get_completion(self, message: str, style: str) -> Optional[str]:
        if not self.model_loaded:
            print("[상태] 모델이 아직 로드되지 않았습니다")
            return None
            
        try:
            # 스타일별 프롬프트 정의
            style_prompts = {
                'formal': f"""다음 문장을 격식있고 공적인 말투로 변환해주세요.

                규칙:
                1. '-습니다', '-입니다' 등의 격식체 사용
                2. 정중하고 예의바른 톤 유지
                3. 불필요한 존댓말은 제외
                
                입력: "{message}"
                출력:""",
                
                'casual': f"""다음 문장을 친근하고 편안한 말투로 변환해주세요.

                규칙:
                1. '-야', '-어' 등의 반말 사용
                2. 자연스럽고 일상적인 표현 사용
                3. 너무 격식없지 않게 유지
                
                입력: "{message}"
                출력:""",
                
                'polite': f"""다음 문장을 매우 공손하고 예의바른 말투로 변환해주세요.

                규칙:
                1. '-시옵니다', '-드립니다' 등 최상급 존댓말 사용
                2. 겸손하고 정중한 표현 사용
                3. 상대방을 최대한 존중하는 어조
                
                입력: "{message}"
                출력:""",
                
                'cute': f"""다음 문장을 귀엽고 발랄한 말투로 변환해주세요.

                규칙:
                1. "~용", "~얏", "~냥" 같은 귀여운 어미 사용하기
                2. 밝고 긍정적인 톤으로 변환하기
                3. 짧고 간단하게 변환하기
                4. 문장 끝에는 느낌표나 물음표 사용하기
                
                입력: "{message}"
                출력:"""
            }

            prompt = style_prompts.get(style, style_prompts['casual'])  # 기본값은 친근체

            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)

            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']

            outputs = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.generate(
                        **inputs,
                        max_new_tokens=64,  # 출력 길이 증가
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.2  # 반복 방지
                    )
                ),
                timeout=self.inference_timeout
            )

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            # cute 스타일일 때만 이모지 추가
            if style == 'cute':
                emoji_count = random.randint(1, 2)
                selected_emojis = ' ' + ''.join(random.sample(self.emojis, emoji_count))
                response = response + selected_emojis

            return response

        except asyncio.TimeoutError:
            print(f"[타임아웃] {self.inference_timeout}초 초과")
            return None
        except Exception as e:
            print(f"[에러] HuggingFace 모델 오류: {e}")
            return None
    def _generate_response(self, message: str, style: str) -> str:
        # 기존의 동기 처리 코드를 여기로 이동
        if style == 'cute':
            prompt = f"""다음 문장을 귀엽고 발랄한 말투로 변환해주세요...."""
        else:
            prompt = f"""다음 문장을 변환해주세요...."""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
