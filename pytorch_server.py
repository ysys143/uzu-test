#!/usr/bin/env python3
"""
PyTorch Gemma 모델을 FastAPI 서버로 제공
OpenAI 호환 API 형태로 /chat/completions 엔드포인트 제공
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os
import json

# HuggingFace tokenizers 병렬처리 경고 억제
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

app = FastAPI(title="PyTorch Gemma Server", version="1.0.0")

# 전역 모델 및 토크나이저
model = None
tokenizer = None

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.3
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict

def load_model(model_path: str = "./models/gemma-3-1b-it"):
    """모델과 토크나이저를 로딩"""
    global model, tokenizer
    
    print(f"🔥 PyTorch 모델 로딩 시작: {model_path}")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="mps"
    )
    
    load_time = time.time() - start_time
    print(f"✅ PyTorch 모델 로딩 완료: {load_time:.2f}초")

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 모델 로딩"""
    model_path = os.getenv("PYTORCH_MODEL_PATH", "./models/gemma-3-1b-it")
    load_model(model_path)

@app.get("/")
async def health_check():
    """헬스체크 엔드포인트"""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI 호환 채팅 완성 API"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # 메시지를 프롬프트로 변환
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": msg.role, "content": msg.content} for msg in request.messages],
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 토큰화
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("mps")
        
        # 생성
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=request.temperature,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 생성된 토큰만 디코딩
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        
        # OpenAI 호환 응답 형태
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": input_length,
                "completion_tokens": len(generated_tokens),
                "total_tokens": input_length + len(generated_tokens),
                "generation_time": generation_time
            }
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PyTorch Gemma Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--model-path", default="./models/gemma-3-1b-it", help="Model path")
    
    args = parser.parse_args()
    
    # 환경변수 설정
    os.environ["PYTORCH_MODEL_PATH"] = args.model_path
    
    print(f"🚀 PyTorch 서버 시작: {args.host}:{args.port}")
    print(f"📁 모델 경로: {args.model_path}")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info") 