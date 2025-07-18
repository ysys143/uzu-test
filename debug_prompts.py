#!/usr/bin/env python3
"""
각 엔진의 프롬프트 처리 방식 디버그
"""

import subprocess
import time
import requests
import json
import os

def test_pytorch_prompt():
    """PyTorch 프롬프트 처리 확인"""
    print("🔥 PyTorch 프롬프트 처리:")
    
    # HuggingFace tokenizers 병렬처리 경고 억제
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained('./models/gemma-3-1b-it')
    model = AutoModelForCausalLM.from_pretrained(
        './models/gemma-3-1b-it',
        torch_dtype=torch.float16,
        device_map="mps"
    )
    
    prompt = "안녕하세요! 반갑습니다."
    
    # 토크나이저 적용 전후 확인
    print(f"  원본 프롬프트: '{prompt}'")
    
    # Chat template이 있는지 확인
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        print(f"  Chat template: {tokenizer.chat_template}")
        # Messages 형식으로 적용
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        print(f"  Chat template 적용 후: '{formatted_prompt}'")
    else:
        print("  Chat template 없음")
        formatted_prompt = prompt
    
    # 토큰화 확인
    tokens = tokenizer.encode(formatted_prompt, return_tensors="pt")
    print(f"  토큰 수: {len(tokens[0])}")
    print(f"  토큰들: {tokens[0].tolist()[:20]}...")  # 처음 20개만
    
    # 토큰을 다시 디코딩해서 확인
    decoded = tokenizer.decode(tokens[0], skip_special_tokens=False)
    print(f"  디코딩된 텍스트: '{decoded}'")

def test_ollama_prompt():
    """Ollama 프롬프트 처리 확인"""
    print("\n🦙 Ollama 프롬프트 처리:")
    
    prompt = "안녕하세요! 반갑습니다."
    print(f"  원본 프롬프트: '{prompt}'")
    
    # Ollama 명령어로 전송되는 실제 형태
    cmd = ['ollama', 'run', 'gemma2:2b', prompt]
    print(f"  Ollama 명령어: {' '.join(cmd)}")
    
    # Modelfile에서 template 확인
    print("  Modelfile template:")
    with open('models/Modelfile', 'r') as f:
        content = f.read()
        print(f"  {content}")

def test_uzu_prompt():
    """Uzu 프롬프트 처리 확인"""
    print("\n⚡ Uzu 프롬프트 처리:")
    
    prompt = "안녕하세요! 반갑습니다."
    print(f"  원본 프롬프트: '{prompt}'")
    
    # Uzu API 요청 형태
    payload = {
        "model": "gemma-3-1b-it-uzu",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    print(f"  API 요청 형태:")
    print(f"  {json.dumps(payload, ensure_ascii=False, indent=2)}")

def test_llamacpp_prompt():
    """llama.cpp 프롬프트 처리 확인"""
    print("\n🦙 llama.cpp 프롬프트 처리:")
    
    prompt = "안녕하세요! 반갑습니다."
    print(f"  원본 프롬프트: '{prompt}'")
    
    # llama.cpp 명령어
    cmd = [
        'llama-cli',
        '-m', './models/gemma-3-1b-it-gguf-llama/model.gguf',
        '-p', prompt,
        '-n', '10',  # 짧게 테스트
        '--temp', '0.7',
        '-ngl', '99',
        '--no-display-prompt',
        '-no-cnv'
    ]
    print(f"  llama.cpp 명령어: {' '.join(cmd)}")

if __name__ == "__main__":
    print("🔍 프롬프트 처리 방식 디버그")
    print("=" * 50)
    
    # PyTorch 테스트 (시간이 오래 걸릴 수 있음)
    try:
        test_pytorch_prompt()
    except Exception as e:
        print(f"  PyTorch 테스트 실패: {e}")
    
    # 다른 엔진들 프롬프트 형태 확인
    test_ollama_prompt()
    test_uzu_prompt()
    test_llamacpp_prompt()
    
    print("\n✅ 디버그 완료!") 