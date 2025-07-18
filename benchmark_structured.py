#!/usr/bin/env python3
"""
구조화된 프롬프트를 사용한 AI 추론 엔진 벤치마크 스크립트
응답 형태를 표준화하여 공정한 성능 비교를 위함
"""

import json
import time
import subprocess
import requests
import os
import statistics
from benchmark_prompts import get_all_prompts, get_formatted_prompt, SYSTEM_PROMPT

def count_tokens_rough(text):
    """대략적인 토큰 수 계산 (한글 1.5, 영어 0.75 토큰으로 추정)"""
    korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
    other_chars = len(text) - korean_chars
    return int(korean_chars * 1.5 + other_chars * 0.75)

def test_pytorch():
    """PyTorch + Transformers 테스트"""
    print("PyTorch 테스트 중...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = "models/gemma-3-1b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="mps" if torch.backends.mps.is_available() else "auto"
        )
        
        results = []
        all_prompts = get_all_prompts()
        
        for i, prompt in enumerate(all_prompts[:5]):  # 처음 5개로 테스트
            print(f"  프롬프트 {i+1}/5 처리 중...")
            
            # 구조화된 프롬프트 사용
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            if torch.backends.mps.is_available():
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,  # 더 긴 응답 허용
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            end_time = time.time()
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            tokens_generated = len(tokenizer.encode(response))
            duration = end_time - start_time
            tps = tokens_generated / duration if duration > 0 else 0
            
            results.append({
                'prompt': prompt,
                'response': response,
                'tokens': tokens_generated,
                'duration': duration,
                'tps': tps
            })
        
        avg_tps = statistics.mean([r['tps'] for r in results])
        total_tokens = sum([r['tokens'] for r in results])
        
        return {
            'engine': 'PyTorch',
            'avg_tps': avg_tps,
            'total_tokens': total_tokens,
            'results': results
        }
        
    except Exception as e:
        print(f"PyTorch 오류: {e}")
        return None

def test_ollama():
    """Ollama 테스트"""
    print("Ollama 테스트 중...")
    try:
        results = []
        all_prompts = get_all_prompts()
        
        for i, prompt in enumerate(all_prompts[:5]):  # 처음 5개로 테스트
            print(f"  프롬프트 {i+1}/5 처리 중...")
            
            # 구조화된 프롬프트 사용
            full_prompt = get_formatted_prompt(prompt)
            
            start_time = time.time()
            
            response = requests.post('http://localhost:11434/api/generate', 
                json={
                    'model': 'gemma2:9b-instruct-q4_K_M',
                    'prompt': full_prompt,
                    'stream': False,
                    'options': {
                        'num_predict': 500,  # 더 긴 응답 허용
                        'temperature': 0.7
                    }
                }, timeout=120)
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                response_text = data['response']
                tokens_generated = count_tokens_rough(response_text)
                duration = end_time - start_time
                tps = tokens_generated / duration if duration > 0 else 0
                
                results.append({
                    'prompt': prompt,
                    'response': response_text,
                    'tokens': tokens_generated,
                    'duration': duration,
                    'tps': tps
                })
            else:
                print(f"    Ollama API 오류: {response.status_code}")
                return None
        
        avg_tps = statistics.mean([r['tps'] for r in results])
        total_tokens = sum([r['tokens'] for r in results])
        
        return {
            'engine': 'Ollama',
            'avg_tps': avg_tps,
            'total_tokens': total_tokens,
            'results': results
        }
        
    except Exception as e:
        print(f"Ollama 오류: {e}")
        return None

def test_llama_cpp():
    """llama.cpp 테스트"""
    print("llama.cpp 테스트 중...")
    try:
        results = []
        all_prompts = get_all_prompts()
        
        for i, prompt in enumerate(all_prompts[:5]):  # 처음 5개로 테스트
            print(f"  프롬프트 {i+1}/5 처리 중...")
            
            # 구조화된 프롬프트 사용
            full_prompt = get_formatted_prompt(prompt)
            
            start_time = time.time()
            
            result = subprocess.run([
                'llama-cli',
                '-m', 'models/gemma-3-1b-it-gguf/gemma-3-1b-it-Q4_K_M.gguf',
                '-p', full_prompt,
                '-n', '500',  # 더 긴 응답 허용
                '--temp', '0.7',
                '--chat-template', 'gemma',
                '-st'
            ], capture_output=True, text=True, timeout=120)
            
            end_time = time.time()
            
            if result.returncode == 0:
                response_text = result.stdout.strip()
                tokens_generated = count_tokens_rough(response_text)
                duration = end_time - start_time
                tps = tokens_generated / duration if duration > 0 else 0
                
                results.append({
                    'prompt': prompt,
                    'response': response_text,
                    'tokens': tokens_generated,
                    'duration': duration,
                    'tps': tps
                })
            else:
                print(f"    llama.cpp 오류: {result.stderr}")
                return None
        
        avg_tps = statistics.mean([r['tps'] for r in results])
        total_tokens = sum([r['tokens'] for r in results])
        
        return {
            'engine': 'llama.cpp',
            'avg_tps': avg_tps,
            'total_tokens': total_tokens,
            'results': results
        }
        
    except Exception as e:
        print(f"llama.cpp 오류: {e}")
        return None

def test_uzu():
    """Uzu 테스트"""
    print("Uzu 테스트 중...")
    try:
        # Uzu 서버가 실행 중인지 확인
        try:
            test_response = requests.get('http://localhost:8000/health', timeout=5)
        except:
            print("  Uzu 서버 시작 중...")
            # 서버 시작
            env = os.environ.copy()
            env['UZU_MODEL_PATH'] = 'models/gemma-3-1b-it-uzu'
            
            server_process = subprocess.Popen([
                'cargo', 'run', '--bin', 'uzu-cli', '--', 'serve'
            ], cwd='uzu', env=env)
            
            # 서버가 준비될 때까지 대기
            for _ in range(30):
                try:
                    test_response = requests.get('http://localhost:8000/health', timeout=2)
                    if test_response.status_code == 200:
                        break
                except:
                    pass
                time.sleep(1)
            else:
                print("    Uzu 서버 시작 실패")
                return None
        
        results = []
        all_prompts = get_all_prompts()
        
        for i, prompt in enumerate(all_prompts[:5]):  # 처음 5개로 테스트
            print(f"  프롬프트 {i+1}/5 처리 중...")
            
            start_time = time.time()
            
            response = requests.post('http://localhost:8000/v1/chat/completions', 
                json={
                    'model': 'default',
                    'messages': [
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': prompt}
                    ],
                    'max_tokens': 500,  # 더 긴 응답 허용
                    'temperature': 0.7
                }, timeout=120)
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                response_text = data['choices'][0]['message']['content']
                tokens_generated = count_tokens_rough(response_text)
                duration = end_time - start_time
                tps = tokens_generated / duration if duration > 0 else 0
                
                results.append({
                    'prompt': prompt,
                    'response': response_text,
                    'tokens': tokens_generated,
                    'duration': duration,
                    'tps': tps
                })
            else:
                print(f"    Uzu API 오류: {response.status_code}")
                return None
        
        avg_tps = statistics.mean([r['tps'] for r in results])
        total_tokens = sum([r['tokens'] for r in results])
        
        return {
            'engine': 'Uzu',
            'avg_tps': avg_tps,
            'total_tokens': total_tokens,
            'results': results
        }
        
    except Exception as e:
        print(f"Uzu 오류: {e}")
        return None

def run_benchmark():
    """전체 벤치마크 실행"""
    print("=== 구조화된 프롬프트 AI 추론 엔진 벤치마크 ===")
    print(f"프롬프트 수: 5개 (테스트용)")
    print(f"시스템 프롬프트: {SYSTEM_PROMPT[:100]}...")
    print()
    
    # 각 엔진 테스트
    engines = [
        ('pytorch', test_pytorch),
        ('ollama', test_ollama), 
        ('llama_cpp', test_llama_cpp),
        ('uzu', test_uzu)
    ]
    
    all_results = {}
    
    for engine_name, test_func in engines:
        print(f"\n--- {engine_name.upper()} 테스트 ---")
        result = test_func()
        if result:
            all_results[engine_name] = result
            print(f"{result['engine']}: {result['avg_tps']:.2f} TPS (총 {result['total_tokens']} 토큰)")
        else:
            print(f"{engine_name} 테스트 실패")
    
    # 결과 요약
    print("\n=== 벤치마크 결과 요약 ===")
    if all_results:
        # TPS 기준 정렬
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['avg_tps'], reverse=True)
        baseline_tps = sorted_results[-1][1]['avg_tps']  # 가장 느린 것을 기준으로
        
        for i, (engine_name, result) in enumerate(sorted_results):
            speedup = result['avg_tps'] / baseline_tps
            print(f"{i+1}. {result['engine']}: {result['avg_tps']:.2f} TPS ({speedup:.1f}x)")
        
        # 상세 결과 저장
        with open('benchmark_structured_results.json', 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n상세 결과가 benchmark_structured_results.json에 저장되었습니다.")
        
        # 응답 길이 분석
        print("\n=== 응답 길이 분석 ===")
        for engine_name, result in all_results.items():
            token_counts = [r['tokens'] for r in result['results']]
            avg_tokens = statistics.mean(token_counts)
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            print(f"{result['engine']}: 평균 {avg_tokens:.1f} 토큰 (범위: {min_tokens}-{max_tokens})")
    
    else:
        print("모든 엔진 테스트가 실패했습니다.")

if __name__ == "__main__":
    run_benchmark() 