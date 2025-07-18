#!/usr/bin/env python3
"""
Uzu 단독 테스트 스크립트
"""

import subprocess
import time
import requests
import json

def test_uzu_server():
    """Uzu 서버 모드 테스트"""
    print("🚀 Uzu 서버 모드 테스트 시작...")
    
    # Uzu 서버 시작 (포트 51839 사용)
    print("  서버 시작 중... (포트 51839)")
    import os
    env = os.environ.copy()
    env['ROCKET_PORT'] = '51839'
    
    server_process = subprocess.Popen(
        ['./uzu/target/release/uzu_cli', 'serve', './models/gemma-3-1b-it-uzu'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    # 서버 출력 확인을 위한 스레딩
    import threading
    
    def print_server_output():
        for line in server_process.stdout:
            print(f"  [서버 stdout] {line.strip()}")
    
    def print_server_error():
        for line in server_process.stderr:
            print(f"  [서버 stderr] {line.strip()}")
    
    stdout_thread = threading.Thread(target=print_server_output, daemon=True)
    stderr_thread = threading.Thread(target=print_server_error, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    
    # 서버가 시작될 때까지 대기
    server_ready = False
    print("  서버 시작 대기 중...")
    
    for i in range(60):  # 60초 대기
        try:
            # 포트 8000에서 확인 (서버가 실제로 사용하는 포트)
            response = requests.get('http://localhost:8000/', timeout=2)
            print(f"  서버 응답: {response.status_code}")
            server_ready = True
            break
        except requests.exceptions.ConnectionError:
            print(f"    연결 대기 중... ({i+1}/60)")
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
        return False
    
    print("  ✅ 서버 시작 성공!")
    
    # API 테스트
    try:
        print("  API 테스트 중...")
        payload = {
            "model": "gemma-3-1b-it-uzu",
            "messages": [
                {"role": "user", "content": "안녕하세요! 반갑습니다."}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        print(f"  요청 데이터: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        start_time = time.time()
        response = requests.post(
            'http://localhost:8000/chat/completions',
            json=payload,
            timeout=30
        )
        inference_time = time.time() - start_time
        
        print(f"  응답 시간: {inference_time:.3f}초")
        print(f"  HTTP 상태: {response.status_code}")
        print(f"  응답 헤더: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  응답 데이터: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
            response_text = data['choices'][0]['message']['content']
            response_tokens = len(response_text.split())
            tps = response_tokens / inference_time if inference_time > 0 else 0
            
            print(f"\n  📊 결과:")
            print(f"    - 응답: {response_text}")
            print(f"    - 토큰 수: {response_tokens}")
            print(f"    - TPS: {tps:.2f}")
            print(f"    - 추론 시간: {inference_time:.3f}초")
            
        else:
            print(f"  ❌ API 호출 실패: {response.text}")
            
    except Exception as e:
        print(f"  ❌ API 테스트 실패: {e}")
    
    finally:
        print("  서버 종료 중...")
        server_process.terminate()
        server_process.wait()
        print("  ✅ 서버 종료 완료")

def test_uzu_cli():
    """Uzu CLI 직접 모드 테스트"""
    print("\n🎯 Uzu CLI 직접 모드 테스트...")
    
    # CLI 버전 확인
    try:
        result = subprocess.run(
            ['./uzu/target/release/uzu_cli', '--help'],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(f"  CLI 도움말: {result.stdout[:200]}...")
    except Exception as e:
        print(f"  CLI 도움말 실패: {e}")

if __name__ == "__main__":
    print("🧪 Uzu 단독 테스트 시작!")
    print("=" * 50)
    
    # CLI 기본 확인
    test_uzu_cli()
    
    # 서버 모드 테스트
    test_uzu_server()
    
    print("\n✅ 테스트 완료!") 