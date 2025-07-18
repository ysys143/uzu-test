# Uzu 벤치마크 설정 가이드

## 설정 파일 사용법

벤치마크는 `benchmark_config.json` 파일을 통해 설정됩니다.

### 주요 설정 항목

```json
{
  "benchmark": {
    "max_tokens": 500,         // 최대 생성 토큰 수 (300자 제한을 위한 여유분)
    "temperature": 0.3,        // 생성 온도 (0.0-1.0)
    "num_runs": 10,           // 반복 실행 횟수
    "timeout_seconds": 120,    // 타임아웃 (초)
    "system_prompt_override": null  // 시스템 프롬프트 오버라이드
  },
  "engines": {
    "pytorch": {
      "enabled": true,         // 엔진 활성화/비활성화
      "device": "mps",         // PyTorch 디바이스
      "torch_dtype": "float16" // 데이터 타입
    },
    "ollama": {
      "enabled": true,
      "model_name": "gemma-3-1b-it-bench",  // Ollama 모델명
      "verbose": true          // 상세 출력
    },
    "llamacpp": {
      "enabled": true,
      "model_path": "./models/gemma-3-1b-it-gguf-llama/model.gguf",
      "ngl": 99,              // GPU 레이어 수
      "chat_template": "gemma" // 채팅 템플릿
    },
    "uzu": {
      "enabled": true,
      "model_path": "./models/gemma-3-1b-it-uzu",
      "port": 51839,          // 서버 포트
      "server_timeout": 60     // 서버 시작 대기 시간
    }
  }
}
```

## 환경변수로 설정 오버라이드

설정 파일 값을 환경변수로 오버라이드할 수 있습니다:

```bash
# 기본 벤치마크 설정
export BENCHMARK_MAX_TOKENS=150
export BENCHMARK_TEMPERATURE=0.5
export BENCHMARK_NUM_RUNS=5
export BENCHMARK_TIMEOUT=90

# 시스템 프롬프트 오버라이드
export BENCHMARK_SYSTEM_PROMPT="당신은 도움이 되는 AI 어시스턴트입니다."

# 벤치마크 실행
python3 bench_run.py
```

## 사용 예시

### 1. 빠른 테스트 (낮은 토큰, 적은 반복)
```bash
export BENCHMARK_MAX_TOKENS=50
export BENCHMARK_NUM_RUNS=3
python3 bench_run.py
```

### 2. 정확한 성능 측정 (높은 토큰, 많은 반복)
```bash
export BENCHMARK_MAX_TOKENS=300
export BENCHMARK_NUM_RUNS=20
export BENCHMARK_TEMPERATURE=0.1
python3 bench_run.py
```

### 3. 특정 엔진만 테스트
설정 파일에서 `"enabled": false`로 설정하거나:

```json
{
  "engines": {
    "pytorch": {"enabled": false},
    "ollama": {"enabled": true},
    "llamacpp": {"enabled": false},
    "uzu": {"enabled": true}
  }
}
```

### 4. 다양한 온도 설정 비교
```bash
# 낮은 온도 (더 일관된 응답)
export BENCHMARK_TEMPERATURE=0.1
python3 bench_run.py

# 높은 온도 (더 창의적인 응답)
export BENCHMARK_TEMPERATURE=0.8
python3 bench_run.py
```

## 로그 설정

```json
{
  "logging": {
    "directory": "logging",      // 로그 저장 디렉토리
    "level": "INFO",            // 로그 레벨 (DEBUG, INFO, WARNING, ERROR)
    "console_output": true       // 콘솔 출력 여부
  }
}
```

## 설정 우선순위

1. **명령행 인자** (가장 높음)
2. **환경변수**
3. **설정 파일**
4. **기본값** (가장 낮음)

## 주의사항

- 시스템 프롬프트에서 **300자 이내 응답** 제한을 명시하여 일관된 응답 길이를 유도합니다
- `max_tokens`는 500으로 설정하여 300자 제한 내에서도 충분한 토큰 여유를 제공합니다
- `temperature`는 0.0-1.0 범위에서 설정하세요
- 높은 `num_runs` 값은 벤치마크 시간을 크게 늘립니다
- `timeout_seconds`는 모델 크기와 하드웨어 성능에 맞게 조정하세요 