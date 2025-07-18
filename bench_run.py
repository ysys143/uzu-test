#!/usr/bin/env python3
"""
Uzu AI 추론 엔진 다중 실행 성능 벤치마크 스크립트
각 엔진별로 10번 반복 실행하여 통계적으로 유의한 성능 메트릭을 수집합니다.

지원 엔진:
- PyTorch (HuggingFace + MPS)
- Ollama (GGUF)
- llama.cpp (GGUF + Metal)
- Uzu (Native Metal)
"""

# HuggingFace tokenizers 병렬처리 경고 억제
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import time
import subprocess
import json
import statistics
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import logging


class MultiRunBenchmarkRunner:
    def __init__(self, num_runs: int = None, config_file: str = "benchmark_config.json", quick_test: bool = False):
        # 설정 파일 로딩
        self.config = self._load_config(config_file)
        
        # 환경변수와 설정 파일에서 매개변수 로딩
        self.num_runs = num_runs or int(os.getenv('BENCHMARK_NUM_RUNS', self.config['benchmark']['num_runs']))
        self.max_tokens = int(os.getenv('BENCHMARK_MAX_TOKENS', self.config['benchmark']['max_tokens']))
        self.temperature = float(os.getenv('BENCHMARK_TEMPERATURE', self.config['benchmark']['temperature']))
        self.timeout_seconds = int(os.getenv('BENCHMARK_TIMEOUT', self.config['benchmark']['timeout_seconds']))
        self.quick_test = quick_test or os.getenv('BENCHMARK_QUICK_TEST', '').lower() in ('true', '1', 'yes')
        
        # 로그 파일 설정
        logging_config = self.config['logging']
        os.makedirs(logging_config['directory'], exist_ok=True)
        
        # 출력 디렉토리 생성
        os.makedirs('report', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"{logging_config['directory']}/benchmark_detailed_{timestamp}.log"
        self.timestamp = timestamp
        
        # 로깅 설정
        handlers = [logging.FileHandler(self.log_filename, encoding='utf-8')]
        if logging_config.get('console_output', True):
            handlers.append(logging.StreamHandler())
            
        logging.basicConfig(
            level=getattr(logging, logging_config.get('level', 'INFO')),
            format='%(asctime)s - %(message)s',
            handlers=handlers
        )
        self.logger = logging.getLogger(__name__)
        
        # 구조화된 프롬프트 시스템 도입
        from benchmark_prompts import get_all_prompts, SYSTEM_PROMPT
        all_prompts = get_all_prompts()
        
        # 빠른 테스트 모드 확인
        if self.quick_test:
            # 빠른 테스트: 첫 번째 프롬프트만 사용
            self.test_prompts = [all_prompts[0]]  # 파이썬 리스트와 튜플 차이
        else:
            # 정식 테스트: 다양한 길이의 프롬프트 10개 선택 (더 포괄적인 테스트)
            self.test_prompts = [
                # 짧은 프롬프트 (4개)
                all_prompts[0],   # 파이썬 리스트와 튜플 차이
                all_prompts[1],   # HTTP와 HTTPS 차이  
                all_prompts[4],   # Git과 GitHub 차이
                all_prompts[8],   # 머신러닝과 딥러닝 차이
                
                # 중간 프롬프트 (3개)
                all_prompts[18],  # 데이터 처리 라이브러리
                all_prompts[22],  # 소프트웨어 아키텍처 패턴
                all_prompts[25],  # 클라우드 서비스 비교
                
                # 긴 프롬프트 (3개)  
                all_prompts[34],  # 웹 개발 스택
                all_prompts[40],  # 분산 시스템 설계
                all_prompts[45],  # 데이터 엔지니어링 파이프라인
            ]
        
        # 시스템 프롬프트 설정 (환경변수 또는 설정 파일에서 오버라이드 가능)
        system_prompt_override = os.getenv('BENCHMARK_SYSTEM_PROMPT') or self.config['benchmark'].get('system_prompt_override')
        self.system_prompt = system_prompt_override if system_prompt_override else SYSTEM_PROMPT
        
        self.results = {}
        
    def _load_config(self, config_file: str) -> Dict:
        """설정 파일 로딩"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ 설정 파일 로딩 성공: {config_file}")
            return config
        except FileNotFoundError:
            print(f"⚠️  설정 파일을 찾을 수 없습니다: {config_file}, 기본값 사용")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ 설정 파일 파싱 오류: {e}, 기본값 사용")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """기본 설정 반환"""
        return {
            "benchmark": {
                "max_tokens": 500,
                "temperature": 0.3,
                "num_runs": 10,
                "timeout_seconds": 120,
                "system_prompt_override": None
            },
            "engines": {
                "pytorch": {"enabled": True, "device": "mps", "torch_dtype": "float16"},
                "ollama": {"enabled": True, "model_name": "gemma-3-1b-it-bench", "verbose": True},
                "llamacpp": {"enabled": True, "model_path": "./models/gemma-3-1b-it-gguf-llama/model.gguf", "ngl": 99, "chat_template": "gemma"},
                "uzu": {"enabled": True, "model_path": "./models/gemma-3-1b-it-uzu", "port": 51839, "server_timeout": 60}
            },
            "logging": {"directory": "logging", "level": "INFO", "console_output": True},
            "output": {"report_directory": "report", "data_directory": "output"}
        }
        
    def log_response_details(self, engine_name: str, prompt_idx: int, inference_time: float, 
                           tps: float, response_text: str, tokens_count: int = None):
        """응답 상세 정보를 로그 파일에 기록"""
        response_length = len(response_text.split()) if response_text else 0
        
        # 화면에는 간단한 정보만
        print(f"    프롬프트 {prompt_idx + 1}: {inference_time:.3f}초, {tps:.2f} TPS")
        
        # 로그 파일에는 상세 정보
        self.logger.info(f"[{engine_name}] 프롬프트 {prompt_idx + 1}")
        self.logger.info(f"[{engine_name}] 추론 시간: {inference_time:.3f}초")
        if tokens_count:
            self.logger.info(f"[{engine_name}] 생성 토큰: {tokens_count}개")
        self.logger.info(f"[{engine_name}] TPS: {tps:.2f}")
        self.logger.info(f"[{engine_name}] 응답 길이: {response_length}단어, {len(response_text)}자")
        self.logger.info(f"[{engine_name}] 응답 내용: {response_text}")
        self.logger.info(f"[{engine_name}] " + "-" * 60)
        
    def calculate_statistics(self, values: List[float]) -> Dict:
        """통계 계산"""
        if not values:
            return {
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'std': 0,
                'count': 0
            }
            
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'count': len(values)
        }
        
    def test_pytorch_mps_multi_run(self):
        """PyTorch + MPS 다중 실행 테스트"""
        self.logger.info(f"🔥 PyTorch + MPS {self.num_runs}회 반복 테스트 시작...")
        
        # 모델 한 번만 로딩
        print("  모델 로딩 중...")
        load_start = time.time()
        tokenizer = AutoTokenizer.from_pretrained('./models/gemma-3-1b-it')
        model = AutoModelForCausalLM.from_pretrained(
            './models/gemma-3-1b-it',
            torch_dtype=torch.float16,
            device_map="mps"
        )
        load_time = time.time() - load_start
        print(f"  모델 로딩 완료: {load_time:.2f}초")
        
        all_runs = []
        tps_values = []
        inference_times = []
        
        for run_idx in range(self.num_runs):
            print(f"  실행 {run_idx + 1}/{self.num_runs}...")
            run_results = []
            
            for prompt_idx, prompt in enumerate(self.test_prompts):
                print(f"    프롬프트 {prompt_idx + 1} 시작...")
                try:
                    # 추론 시간 측정
                    inference_start = time.time()
                    
                    # 구조화된 프롬프트 사용 (시스템 메시지 + 사용자 메시지)
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("mps")
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=self.max_tokens,
                            do_sample=True,
                            temperature=self.temperature,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # 생성된 토큰만 디코딩 (입력 프롬프트 제외)
                    input_length = inputs['input_ids'].shape[1]
                    generated_tokens = outputs[0][input_length:]
                    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    inference_time = time.time() - inference_start
                    
                    # 생성된 토큰 수 계산
                    output_tokens = len(generated_tokens)
                    tps = output_tokens / inference_time if inference_time > 0 else 0
                    
                    # 응답 상세 로깅
                    self.log_response_details("PyTorch", prompt_idx, inference_time, tps, response, output_tokens)
                    
                    run_results.append({
                        'prompt_idx': prompt_idx,
                        'prompt': prompt,
                        'response': response,
                        'inference_time': inference_time,
                        'tokens_generated': output_tokens,
                        'tps': tps
                    })
                    
                    tps_values.append(tps)
                    inference_times.append(inference_time)
                    
                    # 변수 정리 (메모리 캐시는 나중에 정리)
                    del outputs, inputs, generated_tokens
                    
                except Exception as e:
                    print(f"    프롬프트 {prompt_idx + 1} 처리 중 오류: {e}")
                    continue
                
                print(f"    프롬프트 {prompt_idx + 1} 완료, 다음 프롬프트로...")
            
            # 이번 실행의 평균 TPS
            if run_results:
                run_avg_tps = sum(r['tps'] for r in run_results) / len(run_results)
                all_runs.append({
                    'run_index': run_idx,
                    'avg_tps': run_avg_tps,
                    'tests': run_results
                })
                print(f"    실행 {run_idx + 1} 평균 TPS: {run_avg_tps:.2f}")
            else:
                print(f"    실행 {run_idx + 1}: 모든 프롬프트 실패")
        
        # 통계 계산
        run_avg_tps_values = [run['avg_tps'] for run in all_runs]
        
        self.results['pytorch'] = {
            'engine': 'PyTorch + MPS',
            'num_runs': self.num_runs,
            'load_time': load_time,
            'statistics': {
                'tps': self.calculate_statistics(tps_values),
                'inference_time': self.calculate_statistics(inference_times),
                'run_avg_tps': self.calculate_statistics(run_avg_tps_values)
            },
            'all_runs': all_runs
        }
        
        avg_tps = self.results['pytorch']['statistics']['tps']['mean']
        print(f"✅ PyTorch 완료 - 로딩: {load_time:.2f}초, 전체 평균 TPS: {avg_tps:.2f}")
        
        # 메모리 정리
        del model, tokenizer
        torch.mps.empty_cache()
        
    def test_ollama_multi_run(self):
        """Ollama 다중 실행 테스트"""
        print(f"🦙 Ollama {self.num_runs}회 반복 테스트 시작...")
        
        all_runs = []
        tps_values = []
        inference_times = []
        
        for run_idx in range(self.num_runs):
            print(f"  실행 {run_idx + 1}/{self.num_runs}...")
            run_results = []
            
            for prompt_idx, prompt in enumerate(self.test_prompts):
                # 구조화된 프롬프트 생성
                full_prompt = f"{self.system_prompt}\n\n사용자 질문: {prompt}"
                
                ollama_config = self.config['engines']['ollama']
                cmd = ['ollama', 'run', ollama_config['model_name']]
                if ollama_config.get('verbose', False):
                    cmd.append('--verbose')
                
                start_time = time.time()
                try:
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=self.timeout_seconds,
                        input=full_prompt + '\n'
                    )
                    inference_time = time.time() - start_time
                    
                    if result.returncode == 0:
                        output_lines = result.stdout.strip().split('\n')
                        stderr_lines = result.stderr.strip().split('\n')
                        response = ""
                        eval_rate = 0
                        
                        # stderr에서 성능 정보 파싱
                        for line in stderr_lines:
                            if 'eval rate:' in line:
                                try:
                                    # "eval rate:     77.72 tokens/s" 형식 파싱
                                    parts = line.split('eval rate:')[1].strip()
                                    eval_rate = float(parts.split('tokens/s')[0].strip())
                                except Exception as e:
                                    print(f"      성능 파싱 오류: {line} -> {e}")
                                    pass
                        
                        # stdout에서 응답 텍스트 파싱
                        for line in output_lines:
                            if not any(keyword in line for keyword in ['total duration:', 'load duration:', 'prompt eval', 'eval ', 'rate:']):
                                if line.strip() and not line.startswith('>>>'):
                                    response += line + " "
                        
                        # 응답 상세 로깅
                        self.log_response_details("Ollama", prompt_idx, inference_time, eval_rate, response)
                        
                        run_results.append({
                            'prompt_idx': prompt_idx,
                            'prompt': prompt,
                            'response': response.strip(),
                            'inference_time': inference_time,
                            'tps': eval_rate
                        })
                        
                        tps_values.append(eval_rate)
                        inference_times.append(inference_time)
                    else:
                        print(f"    에러 (프롬프트 {prompt_idx + 1}): {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print(f"    타임아웃 (프롬프트 {prompt_idx + 1})")
            
            if run_results:
                run_avg_tps = sum(r['tps'] for r in run_results) / len(run_results)
                all_runs.append({
                    'run_index': run_idx,
                    'avg_tps': run_avg_tps,
                    'tests': run_results
                })
                print(f"    실행 {run_idx + 1} 평균 TPS: {run_avg_tps:.2f}")
        
        # 통계 계산
        run_avg_tps_values = [run['avg_tps'] for run in all_runs]
        
        self.results['ollama'] = {
            'engine': 'Ollama (GGUF)',
            'num_runs': self.num_runs,
            'load_time': 0,  # Ollama는 사전 로딩됨
            'statistics': {
                'tps': self.calculate_statistics(tps_values),
                'inference_time': self.calculate_statistics(inference_times),
                'run_avg_tps': self.calculate_statistics(run_avg_tps_values)
            },
            'all_runs': all_runs
        }
        
        avg_tps = self.results['ollama']['statistics']['tps']['mean']
        print(f"✅ Ollama 완료 - 전체 평균 TPS: {avg_tps:.2f}")
        
    def test_llamacpp_multi_run(self):
        """llama.cpp 다중 실행 테스트"""
        print(f"🦙 llama.cpp {self.num_runs}회 반복 테스트 시작...")
        
        all_runs = []
        tps_values = []
        inference_times = []
        
        for run_idx in range(self.num_runs):
            print(f"  실행 {run_idx + 1}/{self.num_runs}...")
            run_results = []
            
            for prompt_idx, prompt in enumerate(self.test_prompts):
                # 구조화된 프롬프트 생성
                full_prompt = f"{self.system_prompt}\n\n사용자 질문: {prompt}"
                
                llamacpp_config = self.config['engines']['llamacpp']
                cmd = [
                    'llama-cli',
                    '-m', llamacpp_config['model_path'],
                    '-p', full_prompt,
                    '-n', str(self.max_tokens),
                    '--temp', str(self.temperature),
                    '-ngl', str(llamacpp_config['ngl']),
                    '--no-display-prompt',
                    '--chat-template', llamacpp_config['chat_template'],
                    '-st'  # single-turn mode
                ]
                
                start_time = time.time()
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds
                    )
                    inference_time = time.time() - start_time
                    
                    if result.returncode == 0:
                        output_lines = result.stderr.split('\n')
                        tps = 0
                        
                        # 성능 정보 파싱 - "76.85 tokens per second" 형식 찾기
                        for line in output_lines:
                            if 'tokens per second' in line and 'eval time' in line:
                                try:
                                    # "   13.01 ms per token,    76.85 tokens per second)" 형식 파싱
                                    parts = line.split('tokens per second')[0]
                                    tps = float(parts.split(',')[-1].strip())
                                    break
                                except Exception as e:
                                    print(f"      TPS 파싱 오류: {line} -> {e}")
                                    pass
                        
                        response = result.stdout.strip()
                        
                        # 응답 상세 로깅
                        self.log_response_details("llama.cpp", prompt_idx, inference_time, tps, response)
                        
                        run_results.append({
                            'prompt_idx': prompt_idx,
                            'prompt': prompt,
                            'response': response,
                            'inference_time': inference_time,
                            'tps': tps
                        })
                        
                        tps_values.append(tps)
                        inference_times.append(inference_time)
                    else:
                        print(f"    에러 (프롬프트 {prompt_idx + 1}): 반환코드 {result.returncode}")
                        print(f"    stderr: {result.stderr[:200]}...")
                        print(f"    stdout: {result.stdout[:200]}...")
                        
                except subprocess.TimeoutExpired:
                    print(f"    타임아웃 (프롬프트 {prompt_idx + 1}) - 120초 초과")
            
            if run_results:
                run_avg_tps = sum(r['tps'] for r in run_results) / len(run_results)
                all_runs.append({
                    'run_index': run_idx,
                    'avg_tps': run_avg_tps,
                    'tests': run_results
                })
                print(f"    실행 {run_idx + 1} 평균 TPS: {run_avg_tps:.2f}")
        
        # 통계 계산
        run_avg_tps_values = [run['avg_tps'] for run in all_runs]
        
        self.results['llamacpp'] = {
            'engine': 'llama.cpp (GGUF + Metal)',
            'num_runs': self.num_runs,
            'load_time': 0,
            'statistics': {
                'tps': self.calculate_statistics(tps_values),
                'inference_time': self.calculate_statistics(inference_times),
                'run_avg_tps': self.calculate_statistics(run_avg_tps_values)
            },
            'all_runs': all_runs
        }
        
        avg_tps = self.results['llamacpp']['statistics']['tps']['mean']
        print(f"✅ llama.cpp 완료 - 전체 평균 TPS: {avg_tps:.2f}")
        
    def test_uzu_multi_run(self):
        """Uzu 다중 실행 테스트 (서버 모드 사용)"""
        print(f"⚡ Uzu {self.num_runs}회 반복 테스트 시작...")
        print("  Uzu 서버 시작 중...")
        
        # test_uzu_only.py에서 검증된 방식 적용
        import signal
        import requests
        import os
        
        # Uzu 설정 가져오기
        uzu_config = self.config['engines']['uzu']
        
        # 환경변수 설정 (검증된 방식)
        env = os.environ.copy()
        env['ROCKET_PORT'] = str(uzu_config['port'])
        
        # Uzu 서버 시작 (로그는 메인 로그에 통합)
        server_process = subprocess.Popen(
            ['./uzu/target/release/uzu_cli', 'serve', uzu_config['model_path']],
            stdout=subprocess.DEVNULL,  # 서버 출력 숨김
            stderr=subprocess.DEVNULL,   # 서버 에러 숨김
            text=True,
            env=env
        )
        
        # 서버 로그는 파일로 저장되므로 별도 모니터링 불필요
        
        # 서버가 시작될 때까지 대기 (검증된 방식)
        import time
        server_ready = False
        print("  서버 시작 대기 중...")
        
        for i in range(uzu_config['server_timeout']):  # 설정된 시간 대기
            try:
                # 포트 8000에서 확인 (서버가 실제로 사용하는 포트)
                response = requests.get('http://localhost:8000/', timeout=2)
                print(f"  서버 응답: {response.status_code}")
                server_ready = True
                break
            except requests.exceptions.ConnectionError:
                print(f"    연결 대기 중... ({i+1}/{uzu_config['server_timeout']})")
                time.sleep(1)
            except Exception as e:
                print(f"    예외 발생: {e}")
                time.sleep(1)
        
        if not server_ready:
            print("  ❌ 서버 시작 실패")
            print("  서버 프로세스 상태:")
            print(f"    - 반환코드: {server_process.poll()}")
            
            # 서버 로그 확인
            try:
                stdout, stderr = server_process.communicate(timeout=5)
                print(f"    - stdout: {stdout}")
                print(f"    - stderr: {stderr}")
            except:
                print("    - 로그 읽기 실패")
            
            server_process.terminate()
            return
            
        print("  ✅ 서버 시작 성공!")
        
        try:
            all_runs = []
            tps_values = []
            inference_times = []
            
            for run_idx in range(self.num_runs):
                print(f"  실행 {run_idx + 1}/{self.num_runs}...")
                run_results = []
                
                for prompt_idx, prompt in enumerate(self.test_prompts):
                    start_time = time.time()
                    try:
                        # OpenAI 호환 API 호출 (구조화된 프롬프트 사용)
                        payload = {
                            "model": os.path.basename(uzu_config['model_path']),
                            "messages": [
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            "max_tokens": self.max_tokens,
                            "temperature": self.temperature
                        }
                        
                        response = requests.post(
                            'http://localhost:8000/chat/completions',
                            json=payload,
                            timeout=self.timeout_seconds
                        )
                        inference_time = time.time() - start_time
                        
                        if response.status_code == 200:
                            data = response.json()
                            response_text = data['choices'][0]['message']['content']
                            
                            # TPS 계산
                            response_tokens = len(response_text.split()) if response_text else 0
                            tps = response_tokens / inference_time if inference_time > 0 else 0
                            
                            # 응답 상세 로깅
                            self.log_response_details("Uzu", prompt_idx, inference_time, tps, response_text)
                            
                            run_results.append({
                                'prompt_idx': prompt_idx,
                                'prompt': prompt,
                                'response': response_text,
                                'inference_time': inference_time,
                                'tps': tps
                            })
                            
                            tps_values.append(tps)
                            inference_times.append(inference_time)
                        else:
                            print(f"    에러 (프롬프트 {prompt_idx + 1}): HTTP {response.status_code}")
                            print(f"    응답: {response.text[:200]}...")
                            
                    except requests.RequestException as e:
                        print(f"    요청 에러 (프롬프트 {prompt_idx + 1}): {e}")
                    except Exception as e:
                        print(f"    일반 에러 (프롬프트 {prompt_idx + 1}): {e}")
                
                if run_results:
                    run_avg_tps = sum(r['tps'] for r in run_results) / len(run_results)
                    all_runs.append({
                        'run_index': run_idx,
                        'avg_tps': run_avg_tps,
                        'tests': run_results
                    })
                    print(f"    실행 {run_idx + 1} 평균 TPS: {run_avg_tps:.2f}")
        
        finally:
            # 서버 종료
            print("  서버 종료 중...")
            server_process.terminate()
            server_process.wait()
            print("  ✅ 서버 종료 완료")
        
        # 통계 계산
        run_avg_tps_values = [run['avg_tps'] for run in all_runs]
        
        self.results['uzu'] = {
            'engine': 'Uzu (Native Metal)',
            'num_runs': self.num_runs,
            'load_time': 0.5,
            'statistics': {
                'tps': self.calculate_statistics(tps_values),
                'inference_time': self.calculate_statistics(inference_times),
                'run_avg_tps': self.calculate_statistics(run_avg_tps_values)
            },
            'all_runs': all_runs,
            'note': 'Simulated results based on previous benchmark'
        }
        
        avg_tps = self.results['uzu']['statistics']['tps']['mean']
        print(f"✅ Uzu 완료 - 전체 평균 TPS: {avg_tps:.2f}")
        
    def generate_report(self):
        """결과 리포트 생성"""
        timestamp = datetime.now().isoformat()
        
        # 콘솔 출력
        report_header = "\n" + "="*80 + "\n"
        report_header += "🎯 Uzu AI 추론 엔진 다중 실행 성능 벤치마크 최종 결과\n"
        report_header += f"실행 횟수: {self.num_runs}회, 실행 시간: {timestamp}\n"
        report_header += "="*80
        
        print(report_header)
        
        # 통계 테이블
        table_header = f"{'엔진':<15} {'평균TPS':<10} {'TPS범위':<15} {'표준편차':<10} {'상대성능':<10}"
        table_separator = "-" * 75
        
        print(table_header)
        print(table_separator)
        
        baseline_tps = self.results.get('pytorch', {}).get('statistics', {}).get('tps', {}).get('mean', 1)
        
        table_rows = []
        for engine, data in self.results.items():
            engine_name = data.get('engine', engine)
            tps_stats = data.get('statistics', {}).get('tps', {})
            
            avg_tps = tps_stats.get('mean', 0)
            min_tps = tps_stats.get('min', 0)
            max_tps = tps_stats.get('max', 0)
            std_tps = tps_stats.get('std', 0)
            relative = avg_tps / baseline_tps if baseline_tps > 0 else 0
            
            tps_range = f"{min_tps:.1f}-{max_tps:.1f}"
            
            row = f"{engine_name:<15} {avg_tps:<10.2f} {tps_range:<15} {std_tps:<10.2f} {relative:<10.1f}x"
            print(row)
            table_rows.append(row)
        
        # Markdown 리포트 생성
        md_content = self._generate_markdown_report(timestamp, table_header, table_rows, baseline_tps)
        
        # Markdown 파일 저장 (report/ 디렉토리에)
        quick_suffix = "_quick" if self.quick_test else ""
        md_file = f'report/benchmark_report_{self.num_runs}runs{quick_suffix}_{self.timestamp}.md'
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        # 상세 JSON 결과 저장
        detailed_results = {
            'benchmark_info': {
                'timestamp': timestamp,
                'num_runs': self.num_runs,
                'num_prompts': len(self.test_prompts),
                'max_tokens': self.max_tokens,
                'test_prompts': self.test_prompts
            },
            'results': self.results,
            'summary': {}
        }
        
        # 요약 통계 생성
        for engine, data in self.results.items():
            detailed_results['summary'][engine] = {
                'engine_name': data.get('engine', engine),
                'avg_tps': data.get('statistics', {}).get('tps', {}).get('mean', 0),
                'tps_std': data.get('statistics', {}).get('tps', {}).get('std', 0),
                'tps_range': [
                    data.get('statistics', {}).get('tps', {}).get('min', 0),
                    data.get('statistics', {}).get('tps', {}).get('max', 0)
                ],
                'relative_performance': data.get('statistics', {}).get('tps', {}).get('mean', 0) / baseline_tps if baseline_tps > 0 else 0
            }
        
        # JSON 파일 저장 (output/ 디렉토리에)
        json_file = f'output/benchmark_results_{self.num_runs}runs{quick_suffix}_{self.timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 상세 결과가 {json_file}에 저장되었습니다.")
        print(f"📋 벤치마크 리포트가 {md_file}에 저장되었습니다.")
        print(f"📈 총 {sum(len(data.get('all_runs', [])) for data in self.results.values())}회의 개별 실행 결과가 포함되었습니다.")
        print(f"📝 상세 로그: {self.log_filename}")
        
    def _get_system_info(self) -> Dict[str, str]:
        """시스템 정보 수집"""
        import platform
        import subprocess
        
        system_info = {}
        
        try:
            # macOS 하드웨어 정보
            if platform.system() == "Darwin":
                result = subprocess.run(
                    ['system_profiler', 'SPHardwareDataType'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'Model Name:' in line:
                            system_info['model'] = line.split('Model Name:')[1].strip()
                        elif 'Model Identifier:' in line:
                            system_info['model_id'] = line.split('Model Identifier:')[1].strip()
                        elif 'Chip:' in line:
                            system_info['processor'] = line.split('Chip:')[1].strip()
                        elif 'Total Number of Cores:' in line:
                            system_info['cores'] = line.split('Total Number of Cores:')[1].strip()
                        elif 'Memory:' in line:
                            system_info['memory'] = line.split('Memory:')[1].strip()
            
            # OS 버전
            result = subprocess.run(['sw_vers'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'ProductVersion:' in line:
                        system_info['os_version'] = line.split('ProductVersion:')[1].strip()
                    elif 'BuildVersion:' in line:
                        system_info['build_version'] = line.split('BuildVersion:')[1].strip()
            
            # Python 버전
            import sys
            system_info['python_version'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
        except Exception as e:
            print(f"시스템 정보 수집 오류: {e}")
        
        return system_info
        
    def _generate_markdown_report(self, timestamp: str, table_header: str, table_rows: List[str], baseline_tps: float) -> str:
        """Markdown 리포트 생성"""
        system_info = self._get_system_info()
        
        md_content = f"""# Uzu AI 추론 엔진 성능 벤치마크 결과

## 시스템 정보
- **모델**: {system_info.get('model', 'Unknown')} ({system_info.get('model_id', 'Unknown')})
- **프로세서**: {system_info.get('processor', 'Unknown')}
- **CPU 코어**: {system_info.get('cores', 'Unknown')}
- **메모리**: {system_info.get('memory', 'Unknown')}
- **운영체제**: macOS {system_info.get('os_version', 'Unknown')} ({system_info.get('build_version', 'Unknown')})
- **Python**: {system_info.get('python_version', 'Unknown')}

## 벤치마크 정보
- **실행 시간**: {timestamp}
- **반복 횟수**: {self.num_runs}회
- **테스트 프롬프트**: {len(self.test_prompts)}개
- **최대 토큰**: {self.max_tokens}개
- **총 실행 횟수**: {len(self.test_prompts) * self.num_runs * len(self.results)}회

## 성능 요약

| 엔진 | 평균 TPS | TPS범위 | 표준편차 | 상대 성능 |
|------|----------|----------|----------|----------|
"""
        
        # 테이블 데이터 변환
        for engine, data in self.results.items():
            engine_name = data.get('engine', engine)
            tps_stats = data.get('statistics', {}).get('tps', {})
            
            avg_tps = tps_stats.get('mean', 0)
            min_tps = tps_stats.get('min', 0)
            max_tps = tps_stats.get('max', 0)
            std_tps = tps_stats.get('std', 0)
            relative = avg_tps / baseline_tps if baseline_tps > 0 else 0
            
            tps_range = f"{min_tps:.1f}-{max_tps:.1f}"
            
            md_content += f"| {engine_name} | {avg_tps:.2f} | {tps_range} | {std_tps:.2f} | {relative:.1f}x |\n"
        
        # 상세 통계 추가
        md_content += "\n## 상세 통계\n\n"
        
        for engine, data in self.results.items():
            engine_name = data.get('engine', engine)
            tps_stats = data.get('statistics', {}).get('tps', {})
            inference_stats = data.get('statistics', {}).get('inference_time', {})
            
            md_content += f"### {engine_name}\n\n"
            md_content += f"**TPS (Tokens Per Second)**\n"
            md_content += f"- 평균: {tps_stats.get('mean', 0):.2f}\n"
            md_content += f"- 중간값: {tps_stats.get('median', 0):.2f}\n"
            md_content += f"- 최소값: {tps_stats.get('min', 0):.2f}\n"
            md_content += f"- 최대값: {tps_stats.get('max', 0):.2f}\n"
            md_content += f"- 표준편차: {tps_stats.get('std', 0):.2f}\n\n"
            
            md_content += f"**추론 시간 (초)**\n"
            md_content += f"- 평균: {inference_stats.get('mean', 0):.3f}\n"
            md_content += f"- 중간값: {inference_stats.get('median', 0):.3f}\n"
            md_content += f"- 최소값: {inference_stats.get('min', 0):.3f}\n"
            md_content += f"- 최대값: {inference_stats.get('max', 0):.3f}\n"
            md_content += f"- 표준편차: {inference_stats.get('std', 0):.3f}\n\n"
        
        # 테스트 프롬프트 목록 추가
        md_content += "## 테스트 프롬프트\n\n"
        for i, prompt in enumerate(self.test_prompts, 1):
            md_content += f"{i}. \"{prompt}\"\n"
        
        # 실행별 결과 요약
        md_content += "\n## 실행별 평균 TPS\n\n"
        for engine, data in self.results.items():
            engine_name = data.get('engine', engine)
            md_content += f"### {engine_name}\n\n"
            
            all_runs = data.get('all_runs', [])
            if all_runs:
                md_content += "| 실행 | 평균 TPS |\n"
                md_content += "|------|----------|\n"
                for run in all_runs:
                    run_idx = run.get('run_index', 0) + 1
                    avg_tps = run.get('avg_tps', 0)
                    md_content += f"| {run_idx} | {avg_tps:.2f} |\n"
                md_content += "\n"
        
        # 결론
        md_content += "## 결론\n\n"
        
        # 성능 순위
        sorted_engines = sorted(
            [(engine, data.get('statistics', {}).get('tps', {}).get('mean', 0)) 
             for engine, data in self.results.items()], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        md_content += "**성능 순위 (평균 TPS 기준)**\n\n"
        for rank, (engine, avg_tps) in enumerate(sorted_engines, 1):
            engine_name = self.results[engine].get('engine', engine)
            relative = avg_tps / baseline_tps if baseline_tps > 0 else 0
            md_content += f"{rank}. **{engine_name}**: {avg_tps:.2f} TPS ({relative:.1f}x)\n"
        
        # 안정성 분석
        md_content += "\n**성능 안정성 (표준편차 기준)**\n\n"
        sorted_stability = sorted(
            [(engine, data.get('statistics', {}).get('tps', {}).get('std', 0)) 
             for engine, data in self.results.items()], 
            key=lambda x: x[1]
        )
        
        for rank, (engine, std) in enumerate(sorted_stability, 1):
            engine_name = self.results[engine].get('engine', engine)
            md_content += f"{rank}. **{engine_name}**: 표준편차 {std:.2f} TPS\n"
        
        quick_suffix = "_quick" if self.quick_test else ""
        md_content += f"\n---\n\n*벤치마크 실행 시간: {timestamp}*\n"
        md_content += f"*JSON 데이터: output/benchmark_results_{self.num_runs}runs{quick_suffix}_{self.timestamp}.json*\n"
        md_content += f"*상세 로그: {self.log_filename}*\n"
        
        return md_content
        
    def run_all_tests(self):
        """모든 테스트 실행"""
        self.logger.info(f"🚀 Uzu AI 추론 엔진 {self.num_runs}회 반복 벤치마크 시작!")
        self.logger.info(f"테스트 프롬프트 수: {len(self.test_prompts)}")
        self.logger.info(f"최대 토큰 수: {self.max_tokens} (구조화된 응답에 충분한 길이)")
        self.logger.info(f"온도 설정: {self.temperature} (일관된 응답)")
        
        # 활성화된 엔진만 카운트
        enabled_engines = [name for name, config in self.config['engines'].items() if config.get('enabled', True)]
        self.logger.info(f"활성화된 엔진: {', '.join(enabled_engines)}")
        self.logger.info(f"총 예상 실행 횟수: {len(self.test_prompts) * self.num_runs * len(enabled_engines)}회")
        self.logger.info(f"상세 로그 파일: {self.log_filename}")
        self.logger.info("")
        
        engines_to_test = [
            ('pytorch', self.test_pytorch_mps_multi_run),
            ('ollama', self.test_ollama_multi_run),
            ('llamacpp', self.test_llamacpp_multi_run),
            ('uzu', self.test_uzu_multi_run)
        ]
        
        for engine_name, test_method in engines_to_test:
            # 설정에서 엔진이 활성화되어 있는지 확인
            if not self.config['engines'].get(engine_name, {}).get('enabled', True):
                print(f"⏭️  {engine_name} 엔진 비활성화됨, 건너뛰기")
                continue
                
            try:
                test_method()
                print()
            except Exception as e:
                print(f"❌ {engine_name} 테스트 실패: {e}")
                self.logger.error(f"{engine_name} 테스트 오류: {e}", exc_info=True)
                print()
        
        self.generate_report()


if __name__ == "__main__":
    import sys
    
    # 작업 디렉토리 변경
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 명령행 인자 처리
    num_runs = None
    quick_test = False
    
    # 인자 파싱
    args = sys.argv[1:]
    for arg in args:
        if arg.lower() in ('quick', 'q', '--quick', '-q'):
            quick_test = True
        else:
            try:
                num_runs = int(arg)
                if num_runs < 1:
                    raise ValueError("실행 횟수는 1 이상이어야 합니다.")
            except ValueError:
                print(f"오류: 잘못된 인자 '{arg}'")
                print("사용법: python bench_run.py [실행횟수] [quick]")
                print("예시:")
                print("  python bench_run.py 5        # 5회 반복, 전체 프롬프트")
                print("  python bench_run.py 1 quick  # 1회 반복, 프롬프트 1개만")
                print("  python bench_run.py quick    # 기본 반복, 프롬프트 1개만")
                sys.exit(1)
    
    # MultiRunBenchmarkRunner에서 우선순위에 따라 결정됨:
    # 1. 명령행 인자 (num_runs) > 2. 환경변수 > 3. 설정 파일 > 4. 기본값
    runner = MultiRunBenchmarkRunner(num_runs=num_runs, quick_test=quick_test)
    actual_runs = runner.num_runs
    
    test_mode = "빠른 테스트" if quick_test else "정식 벤치마크"
    print(f"{actual_runs}회 반복 {test_mode}를 시작합니다!")
    
    if num_runs:
        print(f"  (명령행 인자로 설정됨)")
    elif os.getenv('BENCHMARK_NUM_RUNS'):
        print(f"  (환경변수 BENCHMARK_NUM_RUNS={os.getenv('BENCHMARK_NUM_RUNS')})")
    else:
        print(f"  (설정 파일 기본값)")
    
    if quick_test:
        print(f"  ⚡ 빠른 테스트 모드: 프롬프트 1개만 사용")
    
    print(f"📊 벤치마크 구성:")
    print(f"  - 프롬프트 수: {len(runner.test_prompts)}개")
    print(f"  - 활성화된 엔진: {len([name for name, config in runner.config['engines'].items() if config.get('enabled', True)])}개")
    print(f"  - 각 엔진당 실행: {actual_runs}회")
    print(f"  - 총 실행 횟수: {len(runner.test_prompts)} × {actual_runs} × {len([name for name, config in runner.config['engines'].items() if config.get('enabled', True)])} = {len(runner.test_prompts) * actual_runs * len([name for name, config in runner.config['engines'].items() if config.get('enabled', True)])}회")
    print()
    runner.run_all_tests() 