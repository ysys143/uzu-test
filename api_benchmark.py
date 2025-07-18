#!/usr/bin/env python3
"""
API 기반 벤치마크 러너
모든 엔진을 서버 형태로 실행하여 순수한 추론 성능만 측정
"""

import time
import requests
import json
import statistics
from datetime import datetime
from typing import Dict, List, Optional
import os
import logging
from server_manager import ServerManager
from benchmark_prompts import get_all_prompts, SYSTEM_PROMPT

class APIBenchmarkRunner:
    def __init__(self, num_runs: int = None, config_file: str = "benchmark_config.json", quick_test: bool = False):
        # 설정 파일 로딩
        self.config = self._load_config(config_file)
        
        # 환경변수와 설정 파일에서 매개변수 로딩
        self.num_runs = num_runs or int(os.getenv('BENCHMARK_NUM_RUNS', self.config['benchmark']['num_runs']))
        self.max_tokens = int(os.getenv('BENCHMARK_MAX_TOKENS', self.config['benchmark']['max_tokens']))
        self.temperature = float(os.getenv('BENCHMARK_TEMPERATURE', self.config['benchmark']['temperature']))
        self.timeout_seconds = int(os.getenv('BENCHMARK_TIMEOUT', self.config['benchmark']['timeout_seconds']))
        self.quick_test = quick_test or os.getenv('BENCHMARK_QUICK_TEST', '').lower() in ('true', '1', 'yes')
        
        # 출력 디렉토리 생성
        os.makedirs('report', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        os.makedirs(self.config['logging']['directory'], exist_ok=True)
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 로그 파일 설정
        logging_config = self.config['logging']
        self.log_filename = f"{logging_config['directory']}/api_benchmark_detailed_{self.timestamp}.log"
        
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
        
        # 시스템 프롬프트 설정
        system_prompt_override = os.getenv('BENCHMARK_SYSTEM_PROMPT') or self.config['benchmark'].get('system_prompt_override')
        self.system_prompt = system_prompt_override if system_prompt_override else SYSTEM_PROMPT
        
        # 프롬프트 선택
        all_prompts = get_all_prompts()
        if self.quick_test:
            self.test_prompts = [all_prompts[0]]  # 첫 번째 프롬프트만
        else:
            self.test_prompts = [
                all_prompts[0], all_prompts[1], all_prompts[4], all_prompts[8],
                all_prompts[18], all_prompts[22], all_prompts[25],
                all_prompts[34], all_prompts[40], all_prompts[45]
            ]
        
        # 서버 관리자 초기화
        self.server_manager = ServerManager(self.config)
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
            "benchmark": {"max_tokens": 500, "temperature": 0.3, "num_runs": 10, "timeout_seconds": 120},
            "servers": {
                "pytorch": {"enabled": True, "port": 8001, "model_path": "./models/gemma-3-1b-it", "startup_timeout": 60, "api_endpoint": "/chat/completions"},
                "ollama": {"enabled": True, "port": 11434, "model_name": "gemma-3-1b-it-bench", "startup_timeout": 30, "api_endpoint": "/api/generate"},
                "llamacpp": {"enabled": True, "port": 8002, "model_path": "./models/gemma-3-1b-it-gguf-llama/model.gguf", "startup_timeout": 30, "api_endpoint": "/completion"},
                "uzu": {"enabled": True, "port": 8000, "model_path": "./models/gemma-3-1b-it-uzu", "startup_timeout": 60, "api_endpoint": "/chat/completions"}
            },
            "logging": {"directory": "logging", "level": "INFO", "console_output": True}
        }
    
    def test_server_api(self, server_name: str, prompt: str) -> Optional[Dict]:
        """개별 서버 API 테스트"""
        server_config = self.config['servers'][server_name]
        base_url = self.server_manager.get_server_url(server_name)
        
        if not base_url:
            self.logger.error(f"{server_name} 서버 URL을 찾을 수 없습니다")
            return None
        
        self.logger.info(f"[{server_name}] API 테스트 시작")
        self.logger.info(f"[{server_name}] 프롬프트: {prompt}")
        
        try:
            if server_name == "pytorch" or server_name == "uzu":
                return self._test_openai_api(server_name, base_url, server_config, prompt)
            elif server_name == "ollama":
                return self._test_ollama_api(server_name, base_url, server_config, prompt)
            elif server_name == "llamacpp":
                return self._test_llamacpp_api(server_name, base_url, server_config, prompt)
            else:
                self.logger.error(f"알 수 없는 서버 타입: {server_name}")
                print(f"❌ 알 수 없는 서버 타입: {server_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"{server_name} API 테스트 오류: {e}", exc_info=True)
            print(f"❌ {server_name} API 테스트 오류: {e}")
            return None
    
    def _test_openai_api(self, server_name: str, base_url: str, config: Dict, prompt: str) -> Dict:
        """OpenAI 호환 API 테스트 (PyTorch, Uzu)"""
        payload = {
            "model": os.path.basename(config.get('model_path', server_name)),
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}{config['api_endpoint']}",
            json=payload,
            timeout=self.timeout_seconds
        )
        inference_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            response_text = data['choices'][0]['message']['content']
            
            # 토큰 수 계산 (사용 가능한 경우)
            tokens_used = data.get('usage', {}).get('completion_tokens', len(response_text.split()))
            tps = tokens_used / inference_time if inference_time > 0 else 0
            
            # 로그 기록
            self.logger.info(f"[{server_name}] 추론 시간: {inference_time:.3f}초")
            self.logger.info(f"[{server_name}] 생성 토큰: {tokens_used}개")
            self.logger.info(f"[{server_name}] TPS: {tps:.2f}")
            self.logger.info(f"[{server_name}] 응답 길이: {len(response_text.split())}단어, {len(response_text)}자")
            self.logger.info(f"[{server_name}] 응답 내용: {response_text}")
            self.logger.info(f"[{server_name}] " + "-" * 60)
            
            return {
                'success': True,
                'response': response_text,
                'inference_time': inference_time,
                'tokens_generated': tokens_used,
                'tps': tps
            }
        else:
            self.logger.error(f"{server_name} API 오류: HTTP {response.status_code}, 응답: {response.text[:200]}")
            print(f"❌ {server_name} API 오류: HTTP {response.status_code}")
            return {'success': False, 'error': f"HTTP {response.status_code}"}
    
    def _test_ollama_api(self, server_name: str, base_url: str, config: Dict, prompt: str) -> Dict:
        """Ollama API 테스트"""
        full_prompt = f"{self.system_prompt}\n\n사용자 질문: {prompt}"
        
        payload = {
            "model": config['model_name'],
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature
            }
        }
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}{config['api_endpoint']}",
            json=payload,
            timeout=self.timeout_seconds
        )
        inference_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', '')
            
            # Ollama는 토큰 속도를 직접 제공
            eval_count = data.get('eval_count', len(response_text.split()))
            tps = eval_count / inference_time if inference_time > 0 else 0
            
            # 로그 기록
            self.logger.info(f"[{server_name}] 추론 시간: {inference_time:.3f}초")
            self.logger.info(f"[{server_name}] 생성 토큰: {eval_count}개")
            self.logger.info(f"[{server_name}] TPS: {tps:.2f}")
            self.logger.info(f"[{server_name}] 응답 길이: {len(response_text.split())}단어, {len(response_text)}자")
            self.logger.info(f"[{server_name}] 응답 내용: {response_text}")
            self.logger.info(f"[{server_name}] " + "-" * 60)
            
            return {
                'success': True,
                'response': response_text,
                'inference_time': inference_time,
                'tokens_generated': eval_count,
                'tps': tps
            }
        else:
            self.logger.error(f"{server_name} API 오류: HTTP {response.status_code}, 응답: {response.text[:200]}")
            print(f"❌ {server_name} API 오류: HTTP {response.status_code}")
            return {'success': False, 'error': f"HTTP {response.status_code}"}
    
    def _test_llamacpp_api(self, server_name: str, base_url: str, config: Dict, prompt: str) -> Dict:
        """llama.cpp 서버 API 테스트"""
        full_prompt = f"{self.system_prompt}\n\n사용자 질문: {prompt}"
        
        payload = {
            "prompt": full_prompt,
            "n_predict": self.max_tokens,
            "temperature": self.temperature,
            "stream": False
        }
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}{config['api_endpoint']}",
            json=payload,
            timeout=self.timeout_seconds
        )
        inference_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('content', '')
            
            # 토큰 정보 (사용 가능한 경우)
            tokens_predicted = data.get('tokens_predicted', len(response_text.split()))
            tps = tokens_predicted / inference_time if inference_time > 0 else 0
            
            # 로그 기록
            self.logger.info(f"[{server_name}] 추론 시간: {inference_time:.3f}초")
            self.logger.info(f"[{server_name}] 생성 토큰: {tokens_predicted}개")
            self.logger.info(f"[{server_name}] TPS: {tps:.2f}")
            self.logger.info(f"[{server_name}] 응답 길이: {len(response_text.split())}단어, {len(response_text)}자")
            self.logger.info(f"[{server_name}] 응답 내용: {response_text}")
            self.logger.info(f"[{server_name}] " + "-" * 60)
            
            return {
                'success': True,
                'response': response_text,
                'inference_time': inference_time,
                'tokens_generated': tokens_predicted,
                'tps': tps
            }
        else:
            self.logger.error(f"{server_name} API 오류: HTTP {response.status_code}, 응답: {response.text[:200]}")
            print(f"❌ {server_name} API 오류: HTTP {response.status_code}")
            return {'success': False, 'error': f"HTTP {response.status_code}"}
    
    def run_benchmark(self):
        """전체 벤치마크 실행"""
        print("🚀 API 기반 벤치마크 시작!")
        print(f"📊 설정: {self.num_runs}회 반복, {len(self.test_prompts)}개 프롬프트")
        print(f"⚡ 모드: {'빠른 테스트' if self.quick_test else '정식 벤치마크'}")
        print(f"📝 상세 로그: {self.log_filename}")
        
        # 로그에 벤치마크 정보 기록
        self.logger.info("🚀 API 기반 벤치마크 시작!")
        self.logger.info(f"실행 횟수: {self.num_runs}회")
        self.logger.info(f"테스트 프롬프트 수: {len(self.test_prompts)}개")
        self.logger.info(f"최대 토큰 수: {self.max_tokens}")
        self.logger.info(f"온도 설정: {self.temperature}")
        self.logger.info(f"빠른 테스트 모드: {self.quick_test}")
        
        # 서버 시작
        started_servers = self.server_manager.start_all_servers()
        if not started_servers:
            self.logger.error("시작된 서버가 없습니다.")
            print("❌ 시작된 서버가 없습니다.")
            return
        
        try:
            # 각 서버별 벤치마크 실행
            for server_name in started_servers:
                print(f"\n🔥 {server_name} 서버 테스트 시작...")
                self.logger.info(f"=== {server_name} 서버 테스트 시작 ===")
                server_results = []
                
                for run_idx in range(self.num_runs):
                    print(f"  실행 {run_idx + 1}/{self.num_runs}...")
                    self.logger.info(f"[{server_name}] 실행 {run_idx + 1}/{self.num_runs} 시작")
                    run_data = []
                    
                    for prompt_idx, prompt in enumerate(self.test_prompts):
                        result = self.test_server_api(server_name, prompt)
                        if result and result.get('success'):
                            run_data.append({
                                'prompt_idx': prompt_idx,
                                'prompt': prompt,
                                'response': result['response'],
                                'inference_time': result['inference_time'],
                                'tokens_generated': result['tokens_generated'],
                                'tps': result['tps']
                            })
                            print(f"    프롬프트 {prompt_idx + 1}: {result['tps']:.2f} TPS")
                        else:
                            print(f"    프롬프트 {prompt_idx + 1}: 실패")
                    
                    if run_data:
                        avg_tps = sum(r['tps'] for r in run_data) / len(run_data)
                        server_results.append({
                            'run_index': run_idx,
                            'avg_tps': avg_tps,
                            'tests': run_data
                        })
                        print(f"    실행 {run_idx + 1} 평균 TPS: {avg_tps:.2f}")
                        self.logger.info(f"[{server_name}] 실행 {run_idx + 1} 평균 TPS: {avg_tps:.2f}")
                    else:
                        self.logger.warning(f"[{server_name}] 실행 {run_idx + 1}: 모든 프롬프트 실패")
                
                # 서버별 통계 계산
                if server_results:
                    all_tps = [test['tps'] for run in server_results for test in run['tests']]
                    all_times = [test['inference_time'] for run in server_results for test in run['tests']]
                    
                    self.results[server_name] = {
                        'engine': f"{server_name.title()} Server",
                        'num_runs': self.num_runs,
                        'statistics': {
                            'tps': self._calculate_statistics(all_tps),
                            'inference_time': self._calculate_statistics(all_times),
                        },
                        'all_runs': server_results
                    }
                    
                    avg_tps = self.results[server_name]['statistics']['tps']['mean']
                    print(f"✅ {server_name} 완료 - 전체 평균 TPS: {avg_tps:.2f}")
                    self.logger.info(f"[{server_name}] 테스트 완료 - 전체 평균 TPS: {avg_tps:.2f}")
                else:
                    print(f"❌ {server_name} 테스트 실패")
                    self.logger.error(f"[{server_name}] 테스트 실패")
        
        finally:
            # 서버 종료
            self.server_manager.stop_all_servers()
        
        # 결과 생성
        self._generate_report()
    
    def _calculate_statistics(self, values: List[float]) -> Dict:
        """통계 계산"""
        if not values:
            return {'mean': 0, 'median': 0, 'min': 0, 'max': 0, 'std': 0, 'count': 0}
        
        return {
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'min': min(values),
            'max': max(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'count': len(values)
        }
    
    def _generate_report(self):
        """결과 리포트 생성"""
        if not self.results:
            print("❌ 생성할 결과가 없습니다.")
            return
        
        timestamp = datetime.now().isoformat()
        
        # 콘솔 출력
        print("\n" + "="*80)
        print("🎯 API 기반 벤치마크 최종 결과")
        print(f"실행 시간: {timestamp}")
        print("="*80)
        
        # 성능 테이블
        baseline_tps = list(self.results.values())[0]['statistics']['tps']['mean']
        
        print(f"{'서버':<15} {'평균TPS':<10} {'TPS범위':<15} {'표준편차':<10} {'상대성능':<10}")
        print("-" * 75)
        
        for server_name, data in self.results.items():
            tps_stats = data['statistics']['tps']
            avg_tps = tps_stats['mean']
            min_tps = tps_stats['min']
            max_tps = tps_stats['max']
            std_tps = tps_stats['std']
            relative = avg_tps / baseline_tps if baseline_tps > 0 else 0
            
            tps_range = f"{min_tps:.1f}-{max_tps:.1f}"
            print(f"{data['engine']:<15} {avg_tps:<10.2f} {tps_range:<15} {std_tps:<10.2f} {relative:<10.1f}x")
        
        # 파일 저장
        quick_suffix = "_quick" if self.quick_test else ""
        
        # JSON 데이터 저장
        json_file = f'output/api_benchmark_results_{self.num_runs}runs{quick_suffix}_{self.timestamp}.json'
        detailed_results = {
            'benchmark_info': {
                'timestamp': timestamp,
                'mode': 'api_based',
                'num_runs': self.num_runs,
                'num_prompts': len(self.test_prompts),
                'max_tokens': self.max_tokens,
                'temperature': self.temperature,
                'quick_test': self.quick_test
            },
            'results': self.results
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Markdown 리포트 저장
        md_file = f'report/api_benchmark_report_{self.num_runs}runs{quick_suffix}_{self.timestamp}.md'
        md_content = self._generate_markdown_report(timestamp, baseline_tps)
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"\n📊 상세 결과: {json_file}")
        print(f"📋 벤치마크 리포트: {md_file}")
        print(f"📝 상세 로그: {self.log_filename}")
        
        self.logger.info("벤치마크 완료!")
        self.logger.info(f"JSON 결과 파일: {json_file}")
        self.logger.info(f"Markdown 리포트: {md_file}")
    
    def _generate_markdown_report(self, timestamp: str, baseline_tps: float) -> str:
        """TPS와 Latency를 모두 고려한 종합적인 Markdown 리포트 생성"""
        md_content = f"""# API 기반 추론 엔진 종합 성능 분석 리포트

## 📊 벤치마크 구성
- **실행 시간**: {timestamp}
- **모드**: {'빠른 테스트' if self.quick_test else '정식 벤치마크'}
- **반복 횟수**: {self.num_runs}회
- **프롬프트 수**: {len(self.test_prompts)}개
- **최대 토큰**: {self.max_tokens}개
- **온도 설정**: {self.temperature} (일관된 응답)
- **총 실행 횟수**: {len(self.test_prompts) * self.num_runs * len(self.results)}회

## 🚀 성능 요약

### 처리량 성능 (TPS)

| 순위 | 서버 | 평균 TPS | TPS 범위 | 표준편차 | 상대 성능 |
|------|------|----------|----------|----------|----------|
"""
        
        # TPS 기준 정렬
        sorted_servers_tps = sorted(
            [(server, data) for server, data in self.results.items()], 
            key=lambda x: x[1]['statistics']['tps']['mean'], 
            reverse=True
        )
        
        for rank, (server_name, data) in enumerate(sorted_servers_tps, 1):
            tps_stats = data['statistics']['tps']
            avg_tps = tps_stats['mean']
            min_tps = tps_stats['min']
            max_tps = tps_stats['max']
            std_tps = tps_stats['std']
            relative = avg_tps / baseline_tps if baseline_tps > 0 else 0
            
            tps_range = f"{min_tps:.1f}-{max_tps:.1f}"
            md_content += f"| {rank}위 | **{data['engine']}** | {avg_tps:.2f} | {tps_range} | {std_tps:.2f} | **{relative:.1f}x** |\n"
        
        # Latency 기준 성능 표
        md_content += "\n### 응답 속도 성능 (Latency)\n\n"
        md_content += "| 순위 | 서버 | 평균 Latency (초) | Latency 범위 | 표준편차 | 일관성 |\n"
        md_content += "|------|------|-------------------|--------------|----------|--------|\n"
        
        # Latency 기준 정렬 (낮은 순)
        sorted_servers_latency = sorted(
            [(server, data) for server, data in self.results.items()], 
            key=lambda x: x[1]['statistics']['inference_time']['mean']
        )
        
        for rank, (server_name, data) in enumerate(sorted_servers_latency, 1):
            latency_stats = data['statistics']['inference_time']
            avg_latency = latency_stats['mean']
            min_latency = latency_stats['min']
            max_latency = latency_stats['max']
            std_latency = latency_stats['std']
            
            latency_range = f"{min_latency:.2f}-{max_latency:.2f}"
            consistency = "높음" if std_latency < avg_latency * 0.2 else "보통" if std_latency < avg_latency * 0.5 else "낮음"
            
            md_content += f"| {rank}위 | **{data['engine']}** | {avg_latency:.3f} | {latency_range} | {std_latency:.3f} | {consistency} |\n"
        
        # 상세 통계 추가
        md_content += "\n## 📈 상세 성능 통계\n\n"
        
        for server_name, data in self.results.items():
            engine_name = data['engine']
            tps_stats = data['statistics']['tps']
            latency_stats = data['statistics']['inference_time']
            
            md_content += f"### {engine_name}\n\n"
            
            # 양쪽 지표를 테이블로 비교
            md_content += "| 지표 | 평균 | 중간값 | 최소값 | 최대값 | 표준편차 |\n"
            md_content += "|------|------|--------|--------|--------|---------|\n"
            md_content += f"| **TPS** | {tps_stats['mean']:.2f} | {tps_stats['median']:.2f} | {tps_stats['min']:.2f} | {tps_stats['max']:.2f} | {tps_stats['std']:.2f} |\n"
            md_content += f"| **Latency (초)** | {latency_stats['mean']:.3f} | {latency_stats['median']:.3f} | {latency_stats['min']:.3f} | {latency_stats['max']:.3f} | {latency_stats['std']:.3f} |\n\n"
        
        # 성능 분석 섹션 추가
        md_content += "## 🔍 성능 분석\n\n"
        
        # 최고 성능 서버 식별
        best_tps_server = sorted_servers_tps[0][0]
        best_latency_server = sorted_servers_latency[0][0]
        
        best_tps_name = self.results[best_tps_server]['engine']
        best_latency_name = self.results[best_latency_server]['engine']
        
        md_content += f"### 성능 우수 서버\n\n"
        md_content += f"- **최고 처리량**: {best_tps_name} ({self.results[best_tps_server]['statistics']['tps']['mean']:.2f} TPS)\n"
        md_content += f"- **최단 응답시간**: {best_latency_name} ({self.results[best_latency_server]['statistics']['inference_time']['mean']:.3f}초)\n\n"
        
        # 성능 일관성 분석
        md_content += "### 성능 안정성 분석\n\n"
        stability_analysis = []
        for server_name, data in self.results.items():
            engine_name = data['engine']
            tps_cv = data['statistics']['tps']['std'] / data['statistics']['tps']['mean'] if data['statistics']['tps']['mean'] > 0 else 0
            latency_cv = data['statistics']['inference_time']['std'] / data['statistics']['inference_time']['mean'] if data['statistics']['inference_time']['mean'] > 0 else 0
            stability_analysis.append((engine_name, tps_cv, latency_cv))
        
        # CV(변동계수) 기준 정렬
        stability_analysis.sort(key=lambda x: (x[1] + x[2]) / 2)
        
        for rank, (engine_name, tps_cv, latency_cv) in enumerate(stability_analysis, 1):
            stability_rating = "높음" if (tps_cv + latency_cv) / 2 < 0.1 else "보통" if (tps_cv + latency_cv) / 2 < 0.3 else "낮음"
            md_content += f"{rank}. **{engine_name}**: TPS 변동성 {tps_cv:.1%}, Latency 변동성 {latency_cv:.1%} (안정성: {stability_rating})\n"
        
        # 시나리오별 권장사항
        md_content += "\n## 🎯 시나리오별 권장 서버\n\n"
        
        md_content += "### 처리량 우선 (API 서비스)\n"
        for rank, (server, data) in enumerate(sorted_servers_tps[:3], 1):
            engine_name = data['engine']
            avg_tps = data['statistics']['tps']['mean']
            avg_latency = data['statistics']['inference_time']['mean']
            md_content += f"{rank}. **{engine_name}**: {avg_tps:.2f} TPS, {avg_latency:.1f}초 지연\n"
        
        md_content += "\n### 응답속도 우선 (실시간 API)\n"
        for rank, (server, data) in enumerate(sorted_servers_latency[:3], 1):
            engine_name = data['engine']
            avg_tps = data['statistics']['tps']['mean']
            avg_latency = data['statistics']['inference_time']['mean']
            md_content += f"{rank}. **{engine_name}**: {avg_latency:.3f}초 지연, {avg_tps:.1f} TPS\n"
        
        md_content += "\n### 안정성 우선 (프로덕션 API)\n"
        for rank, (engine_name, tps_cv, latency_cv) in enumerate(stability_analysis[:3], 1):
            overall_stability = (tps_cv + latency_cv) / 2
            md_content += f"{rank}. **{engine_name}**: 전체 변동성 {overall_stability:.1%}\n"
        
        # 결론 및 권장사항
        md_content += "\n## 📋 종합 결론\n\n"
        
        # 종합 우수 서버 선정 (TPS와 Latency 가중 평균)
        overall_scores = []
        for server_name, data in self.results.items():
            engine_name = data['engine']
            tps_score = data['statistics']['tps']['mean'] / max(d['statistics']['tps']['mean'] for d in self.results.values())
            latency_score = min(d['statistics']['inference_time']['mean'] for d in self.results.values()) / data['statistics']['inference_time']['mean']
            overall_score = (tps_score + latency_score) / 2  # 균등 가중치
            overall_scores.append((engine_name, overall_score, tps_score, latency_score))
        
        overall_scores.sort(key=lambda x: x[1], reverse=True)
        
        md_content += "**종합 성능 순위** (TPS와 Latency 균등 가중치)\n\n"
        for rank, (engine_name, overall_score, tps_score, latency_score) in enumerate(overall_scores, 1):
            md_content += f"{rank}. **{engine_name}**: 종합 점수 {overall_score:.3f} (처리량: {tps_score:.3f}, 응답속도: {latency_score:.3f})\n"
        
        quick_suffix = "_quick" if self.quick_test else ""
        md_content += f"\n---\n\n"
        md_content += f"**데이터 파일**\n"
        md_content += f"- JSON 데이터: `output/api_benchmark_results_{self.num_runs}runs{quick_suffix}_{self.timestamp}.json`\n"
        md_content += f"- 상세 로그: `{self.log_filename}`\n\n"
        md_content += f"*리포트 생성 시간: {timestamp}*\n"
        
        return md_content

if __name__ == "__main__":
    import sys
    
    # 명령행 인자 처리
    num_runs = None
    quick_test = False
    
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
                print("사용법: python api_benchmark.py [실행횟수] [quick]")
                sys.exit(1)
    
    runner = APIBenchmarkRunner(num_runs=num_runs, quick_test=quick_test)
    runner.run_benchmark() 