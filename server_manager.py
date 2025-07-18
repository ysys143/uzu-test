#!/usr/bin/env python3
"""
벤치마크용 서버 관리자
모든 AI 엔진 서버를 시작/종료하고 상태를 관리
"""

import subprocess
import time
import requests
import os
import signal
from typing import Dict, List, Optional
import json

class ServerManager:
    def __init__(self, config: Dict):
        self.config = config
        self.servers = {}  # {server_name: process}
        self.server_urls = {}  # {server_name: base_url}
        
    def start_server(self, server_name: str) -> bool:
        """개별 서버 시작"""
        if server_name in self.servers:
            print(f"⚠️  {server_name} 서버가 이미 실행 중입니다.")
            return True
            
        server_config = self.config['servers'][server_name]
        if not server_config.get('enabled', True):
            print(f"⏭️  {server_name} 서버가 비활성화되어 있습니다.")
            return False
        
        port = server_config['port']
        self.server_urls[server_name] = f"http://localhost:{port}"
        
        print(f"🚀 {server_name} 서버 시작 중... (포트: {port})")
        
        try:
            if server_name == "pytorch":
                process = self._start_pytorch_server(server_config)
            elif server_name == "ollama":
                process = self._start_ollama_server(server_config)
            elif server_name == "llamacpp":
                process = self._start_llamacpp_server(server_config)
            elif server_name == "uzu":
                process = self._start_uzu_server(server_config)
            else:
                print(f"❌ 알 수 없는 서버: {server_name}")
                return False
                
            if process:
                self.servers[server_name] = process
                
                # 서버 상태 확인
                if self._wait_for_server(server_name, server_config['startup_timeout']):
                    print(f"✅ {server_name} 서버 시작 완료")
                    return True
                else:
                    print(f"❌ {server_name} 서버 시작 실패 (타임아웃)")
                    self._stop_server(server_name)
                    return False
            else:
                print(f"❌ {server_name} 서버 프로세스 시작 실패")
                return False
                
        except Exception as e:
            print(f"❌ {server_name} 서버 시작 중 오류: {e}")
            return False
    
    def _start_pytorch_server(self, config: Dict) -> subprocess.Popen:
        """PyTorch 서버 시작"""
        cmd = [
            "python3", "pytorch_server.py",
            "--host", "127.0.0.1",
            "--port", str(config['port']),
            "--model-path", config['model_path']
        ]
        
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid  # 새로운 프로세스 그룹 생성
        )
    
    def _start_ollama_server(self, config: Dict) -> subprocess.Popen:
        """Ollama 서버 시작"""
        # Ollama는 데몬 형태로 실행
        cmd = ["ollama", "serve"]
        
        # 포트 설정
        env = os.environ.copy()
        env['OLLAMA_HOST'] = f"127.0.0.1:{config['port']}"
        
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            preexec_fn=os.setsid
        )
    
    def _start_llamacpp_server(self, config: Dict) -> subprocess.Popen:
        """llama.cpp 서버 시작"""
        cmd = [
            "llama-server",
            "--model", config['model_path'],
            "--host", "127.0.0.1",
            "--port", str(config['port']),
            "--ctx-size", "4096",
            "--n-gpu-layers", str(config.get('ngl', 99)),
            "--chat-template", config.get('chat_template', 'gemma')
        ]
        
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
    
    def _start_uzu_server(self, config: Dict) -> subprocess.Popen:
        """Uzu 서버 시작"""
        env = os.environ.copy()
        env['ROCKET_PORT'] = str(config['port'])
        
        cmd = ["./uzu/target/release/uzu_cli", "serve", config['model_path']]
        
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            preexec_fn=os.setsid
        )
    
    def _wait_for_server(self, server_name: str, timeout: int) -> bool:
        """서버가 준비될 때까지 대기"""
        base_url = self.server_urls[server_name]
        
        for i in range(timeout):
            try:
                # 헬스체크 시도
                response = requests.get(f"{base_url}/", timeout=2)
                if response.status_code in [200, 404]:  # 404도 서버가 응답하는 것으로 간주
                    return True
            except requests.exceptions.RequestException:
                pass
            
            print(f"  {server_name} 대기 중... ({i+1}/{timeout})")
            time.sleep(1)
        
        return False
    
    def _stop_server(self, server_name: str):
        """개별 서버 종료"""
        if server_name not in self.servers:
            return
        
        process = self.servers[server_name]
        try:
            # 프로세스 그룹 전체 종료
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            # 강제 종료
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        
        del self.servers[server_name]
        if server_name in self.server_urls:
            del self.server_urls[server_name]
    
    def start_all_servers(self) -> List[str]:
        """모든 서버 시작"""
        print("🚀 모든 서버 시작 중...")
        
        started_servers = []
        for server_name in self.config['servers'].keys():
            if self.start_server(server_name):
                started_servers.append(server_name)
        
        print(f"✅ {len(started_servers)}개 서버 시작 완료: {', '.join(started_servers)}")
        return started_servers
    
    def stop_all_servers(self):
        """모든 서버 종료"""
        print("🛑 모든 서버 종료 중...")
        
        for server_name in list(self.servers.keys()):
            print(f"  {server_name} 종료 중...")
            self._stop_server(server_name)
        
        print("✅ 모든 서버 종료 완료")
    
    def get_server_url(self, server_name: str) -> Optional[str]:
        """서버 URL 반환"""
        return self.server_urls.get(server_name)
    
    def is_server_running(self, server_name: str) -> bool:
        """서버 실행 상태 확인"""
        if server_name not in self.servers:
            return False
        
        process = self.servers[server_name]
        return process.poll() is None
    
    def check_all_servers(self) -> Dict[str, bool]:
        """모든 서버 상태 확인"""
        status = {}
        for server_name in self.server_urls.keys():
            try:
                base_url = self.server_urls[server_name]
                response = requests.get(f"{base_url}/", timeout=2)
                status[server_name] = response.status_code in [200, 404]
            except:
                status[server_name] = False
        
        return status

if __name__ == "__main__":
    # 테스트용 코드
    import json
    
    with open("benchmark_config.json", "r") as f:
        config = json.load(f)
    
    manager = ServerManager(config)
    
    try:
        # 모든 서버 시작
        started = manager.start_all_servers()
        
        if started:
            print("\n📊 서버 상태 확인:")
            status = manager.check_all_servers()
            for server, is_running in status.items():
                print(f"  {server}: {'✅ 실행 중' if is_running else '❌ 중지'}")
            
            input("\nEnter를 누르면 모든 서버를 종료합니다...")
    
    finally:
        manager.stop_all_servers() 