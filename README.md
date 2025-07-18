# Uzu AI 추론 엔진 성능 벤치마크

Apple Silicon (M3 Max)에서 다양한 AI 추론 엔진들의 성능을 비교하는 벤치마크 시스템입니다.

## 📊 프로젝트 개요

**목표**: Uzu, PyTorch, Ollama, llama.cpp의 Apple Silicon 최적화 성능 비교  
**테스트 모델**: Google Gemma-3-1B-IT  
**테스트 환경**: macOS (Apple Silicon M3 Max)  
**측정 지표**: TPS (Tokens Per Second), 추론 시간, 메모리 사용량

## 🏗️ 서빙 방식 개요

### 1. CLI 기반 서빙 (Subprocess 벤치마크)

순수한 CLI 성능을 측정하는 방식으로, 각 엔진의 최적화된 명령행 도구를 직접 실행합니다.

1. **PyTorch + MPS**: HuggingFace Transformers + Apple Metal Performance Shaders
2. **Ollama CLI**: GGUF 모델을 통한 대화형 추론
3. **llama.cpp CLI**: Metal 가속 활용한 직접 추론

### 2. API 기반 서빙 (API 벤치마크)

HTTP API를 통한 서버 모드 성능을 측정하는 방식입니다.

4. **PyTorch Server**: FastAPI 기반 OpenAI 호환 서버
5. **Ollama Server**: 내장 HTTP 서버 모드
6. **llama.cpp Server**: llama-server를 통한 HTTP API
7. **Uzu Server**: Rust 기반 네이티브 Metal 서버

## 🛠️ 시스템 요구사항

- **운영체제**: macOS 12.0+ (Apple Silicon 권장)
- **메모리**: 최소 16GB (32GB+ 권장)
- **Python**: 3.9+
- **Rust**: 1.86.0+
- **Xcode Command Line Tools**: Uzu Metal 컴파일을 위해 필수
- **Xcode**: 15.0+ (또는 Command Line Tools for Xcode)

## 📦 의존성 설치

### 1. 기본 환경 설정

```bash
# benchmark 디렉토리로 이동
cd benchmark

# Python 가상환경 생성 및 활성화
uv venv
source .venv/bin/activate

# 벤치마크 의존성 설치
uv pip install -e .

# 프로젝트 루트로 복귀
cd ..

# 이후 모든 Python 명령은 benchmark/.venv 환경에서 실행됩니다
```

**중요**: 가상환경은 `benchmark/.venv`에 생성되며, 터미널을 새로 열 때마다 다음 명령으로 활성화해야 합니다:
```bash
cd benchmark && source .venv/bin/activate && cd ..
```

### 2. AI 추론 엔진 설치

#### PyTorch (MPS 백엔드)
```bash
# PyTorch with Metal Performance Shaders (benchmark/pyproject.toml에 포함됨)
# 별도 설치 불필요 - 벤치마크 의존성에서 자동 설치됨

# 수동 설치가 필요한 경우:
# uv pip install torch torchvision torchaudio transformers accelerate
```

#### Ollama
```bash
# Ollama 설치
curl -fsSL https://ollama.com/install.sh | sh

# 서비스 확인
ollama --version  # v0.9.3+
```

#### llama.cpp (Metal 지원)
```bash
# llama.cpp 클론 및 빌드
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Metal 지원으로 빌드
make GGML_METAL=1

# 실행 파일 PATH 추가
export PATH="$PWD:$PATH"
```

#### Uzu 엔진
```bash
# 1. Xcode Command Line Tools 설치 (Metal 컴파일을 위해 필수)
xcode-select --install

# Xcode 라이센스 동의 (빌드 오류 방지)
sudo xcodebuild -license accept

# 2. Uzu 프로젝트 클론
git clone https://github.com/your-org/uzu.git
cd uzu/uzu

# 3. Rust 빌드 (Release 모드)
# Metal 셰이더 컴파일을 위해 Xcode 툴체인 사용
cargo build --release

# 4. CLI 도구 확인
./target/release/uzu_cli --help

# 5. Metal 지원 확인
./target/release/uzu_cli --version
```

**Xcode 의존성 이유:**
- **Metal 셰이더 컴파일**: Uzu의 Metal 커널(.metal 파일) 컴파일에 필요
- **Apple 시스템 프레임워크**: Metal Performance Shaders 등 Apple 네이티브 API 사용
- **빌드 툴체인**: Rust에서 Metal 코드를 빌드하기 위한 Apple 컴파일러 필요

#### Lalamo (모델 변환 도구)
```bash
# Lalamo 설치 (모델 변환용)
git clone https://github.com/your-org/lalamo.git
cd lalamo
uv pip install -e .
```

## 🎯 벤치마크 공정성을 위한 프롬프트 형식 통일

### 구조화된 프롬프트 시스템

벤치마크의 공정성과 일관성을 위해 모든 엔진에서 동일한 프롬프트 형식을 사용합니다:

#### 1. 통일된 시스템 프롬프트

```python
SYSTEM_PROMPT = """당신은 도움이 되는 AI 어시스턴트입니다. 질문에 대해 정확하고 유용한 답변을 제공해주세요. 
답변은 다음 형식을 따라주세요:
1. 핵심 답변 (1-2문장)
2. 상세 설명 (2-3문장)  
3. 추가 정보나 팁 (1-2문장)

**중요: 답변은 반드시 300자 이내로 작성해주세요.**
항상 한국어로 답변해주세요."""
```

#### 2. 계층화된 테스트 프롬프트

응답 길이와 복잡도에 따라 3단계로 구분된 프롬프트를 사용합니다:

- **짧은 프롬프트** (30-80 토큰): "파이썬에서 리스트와 튜플의 차이점은?"
- **중간 프롬프트** (80-150 토큰): "데이터베이스 설계 시 정규화의 개념과 중요성을 설명해주세요."
- **긴 프롬프트** (150+ 토큰): "분산 시스템에서 일관성, 가용성, 분할 허용성 간의 트레이드오프를 설명하고..."

#### 3. 채팅 템플릿 통일화 과정

각 엔진이 사용하는 서로 다른 채팅 템플릿을 **Gemma 표준 형식**으로 통일했습니다:

##### Gemma 표준 채팅 템플릿
```xml
<start_of_turn>system
{시스템 프롬프트}
<end_of_turn>
<start_of_turn>user
{사용자 질문}
<end_of_turn>
<start_of_turn>model
{AI 응답}
<end_of_turn>
```

##### 엔진별 템플릿 통일 과정

**1. PyTorch (HuggingFace Transformers)**
```python
# 자동 템플릿 적용 (내장 Gemma 템플릿 사용)
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prompt}
]
formatted_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
# 결과: "<start_of_turn>system\n{system}<end_of_turn>\n<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
```

**2. Ollama (Modelfile 템플릿 설정)**
```bash
# models/Modelfile에서 Gemma 템플릿 명시적 정의
TEMPLATE """{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{ end }}<start_of_turn>model
{{ .Response }}<end_of_turn>
"""
PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
```

**3. llama.cpp (명령행 옵션 설정)**
```bash
# --chat-template gemma 옵션으로 Gemma 템플릿 강제 적용
llama-server \
    --model ./models/gemma-3-1b-it-gguf-llama/model.gguf \
    --chat-template gemma \
    --host 127.0.0.1 --port 8002
```

**4. Uzu (OpenAI 호환 API)**
```python
# OpenAI 호환 메시지 형식 (내부적으로 Gemma 템플릿 적용)
payload = {
    "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
}
# Uzu 내부에서 Gemma 형식으로 자동 변환
```

##### 템플릿 통일의 중요성

채팅 템플릿이 다르면 **동일한 프롬프트라도 완전히 다른 결과**가 나올 수 있습니다:

```python
# 잘못된 경우 (템플릿 불일치)
llama.cpp: "User: 안녕하세요\nAssistant:"           # 성능 저하
PyTorch:   "<start_of_turn>user\n안녕하세요<end_of_turn><start_of_turn>model\n"  # 정상

# 올바른 경우 (Gemma 템플릿 통일)
모든 엔진: "<start_of_turn>system\n{system}<end_of_turn><start_of_turn>user\n안녕하세요<end_of_turn><start_of_turn>model\n"
```

이 통일화를 통해 **순수한 추론 엔진 성능**만을 비교할 수 있게 되었습니다.

#### 4. 응답 일관성 보장

- **토큰 제한**: 모든 엔진에서 max_tokens=500으로 통일
- **온도 설정**: temperature=0.3으로 일관된 창의성 수준 유지
- **길이 제한**: 시스템 프롬프트로 300자 이내 응답 유도
- **언어 통일**: 모든 응답을 한국어로 제한

이러한 통일화를 통해 각 엔진의 순수한 추론 성능만을 비교할 수 있습니다.

## 🤖 모델 준비 및 변환

### 1. HuggingFace 원본 모델 다운로드

```bash
# 모델 디렉토리 생성
mkdir -p models

# HuggingFace Hub에서 Gemma-3-1B-IT 다운로드
uv pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('google/gemma-3-1b-it', local_dir='./models/gemma-3-1b-it')
"
```

### 2. 엔진별 모델 형식 변환

#### 모델 형식 지원 현황

| 엔진 | 지원 형식 | 변환 필요 | 설명 |
|------|-----------|-----------|------|
| **PyTorch** | SafeTensors | ❌ | HuggingFace 원본 직접 사용 |
| **Uzu** | SafeTensors | ❌ | HuggingFace 원본 직접 사용 |
| **Ollama** | SafeTensors, GGUF | ❌ | 네이티브 SafeTensors 지원 |
| **llama.cpp** | GGUF | ✅ | GGUF 변환 필요 |

#### PyTorch (변환 불필요)
```bash
# HuggingFace 원본 모델 직접 사용
# ./models/gemma-3-1b-it/ 그대로 사용
```

#### Uzu (변환 불필요)
```bash
# HuggingFace SafeTensors 형식 직접 지원
cp -r ./models/gemma-3-1b-it ./models/gemma-3-1b-it-uzu
```

#### Ollama (SafeTensors 직접 사용)
```bash
# Ollama는 HuggingFace SafeTensors를 직접 지원
# 별도 변환 없이 원본 모델을 바로 사용

# 1. Ollama Modelfile 생성
cat > ./models/Modelfile << 'EOF'
FROM ./gemma-3-1b-it
TEMPLATE """{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{ end }}<start_of_turn>model
{{ .Response }}<end_of_turn>
"""
PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
EOF

# 2. Ollama에 모델 등록
cd models
ollama create gemma-3-1b-it-bench -f Modelfile
ollama list  # 등록 확인

# 3. 모델 정보 확인
ollama show gemma-3-1b-it-bench
```

#### llama.cpp (GGUF 변환)
```bash
# HuggingFace → GGUF 변환 (llama.cpp는 GGUF 형식 필요)
python llama.cpp/convert_hf_to_gguf.py \
    ./models/gemma-3-1b-it/ \
    --outfile ./models/gemma-3-1b-it-gguf-llama/model.gguf \
    --outtype f16

# Metal 지원 확인
llama-cli --help | grep -i metal
```

## ⚙️ 벤치마크 설정

### 설정 파일 구조

벤치마크는 `benchmark_config.json` 파일로 통합 관리됩니다:

```json
{
  "benchmark": {
    "max_tokens": 500,
    "temperature": 0.3,
    "num_runs": 10,
    "timeout_seconds": 120,
    "system_prompt_override": null
  },
  "engines": {
    "pytorch": {
      "enabled": true,
      "device": "mps",
      "torch_dtype": "float16"
    },
    "ollama": {
      "enabled": true,
      "model_name": "gemma-3-1b-it-bench",
      "verbose": true
    },
    "llamacpp": {
      "enabled": true,
      "model_path": "./models/gemma-3-1b-it-gguf-llama/model.gguf",
      "ngl": 99,
      "chat_template": "gemma"
    },
    "uzu": {
      "enabled": false,
      "model_path": "./models/gemma-3-1b-it-uzu",
      "port": 51839,
      "server_timeout": 60,
      "note": "CLI 모드는 대화형 전용으로 스크립트 자동화 불가능"
    }
  },
  "servers": {
    "pytorch": {
      "enabled": true,
      "port": 8001,
      "model_path": "./models/gemma-3-1b-it",
      "startup_timeout": 60,
      "api_endpoint": "/chat/completions"
    },
    "ollama": {
      "enabled": true,
      "port": 11434,
      "model_name": "gemma-3-1b-it-bench",
      "startup_timeout": 30,
      "api_endpoint": "/api/generate"
    },
    "llamacpp": {
      "enabled": true,
      "port": 8002,
      "model_path": "./models/gemma-3-1b-it-gguf-llama/model.gguf",
      "startup_timeout": 30,
      "api_endpoint": "/completion",
      "ngl": 99,
      "chat_template": "gemma"
    },
    "uzu": {
      "enabled": true,
      "port": 8000,
      "model_path": "./models/gemma-3-1b-it-uzu",
      "startup_timeout": 60,
      "api_endpoint": "/chat/completions"
    }
  }
}
```

### 환경변수 설정

```bash
# 벤치마크 매개변수 오버라이드
export BENCHMARK_MAX_TOKENS=300
export BENCHMARK_TEMPERATURE=0.1
export BENCHMARK_NUM_RUNS=5
export BENCHMARK_TIMEOUT=90

# 시스템 프롬프트 커스텀
export BENCHMARK_SYSTEM_PROMPT="당신은 도움이 되는 AI 어시스턴트입니다."
```

## 🚀 벤치마크 실행

### 1. CLI 기반 벤치마크 (Subprocess 방식)

순수한 각 엔진의 CLI 성능을 측정합니다:

```bash
# 정식 벤치마크 (10회 반복, 10개 프롬프트)
python3 subprocess_benchmark.py

# 빠른 테스트 (1회 반복, 1개 프롬프트)
python3 subprocess_benchmark.py 1 quick

# 커스텀 반복 횟수
python3 subprocess_benchmark.py 5
```

**지원 엔진**: PyTorch, Ollama, llama.cpp  
**제외 엔진**: Uzu (대화형 CLI로 스크립트 자동화 불가)

### 2. API 기반 벤치마크 (서버 방식)

모든 엔진을 HTTP 서버로 실행하여 API 성능을 측정합니다:

```bash
# 정식 벤치마크 (모든 서버 자동 시작/종료)
python3 api_benchmark.py

# 빠른 테스트
python3 api_benchmark.py quick

# 커스텀 설정
export BENCHMARK_NUM_RUNS=3
python3 api_benchmark.py
```

**지원 엔진**: PyTorch Server, Ollama Server, llama.cpp Server, Uzu Server

### 3. 개별 서버 관리

```bash
# 서버 관리자를 통한 수동 제어
python3 server_manager.py

# 개별 서버 실행
python3 pytorch_server.py --port 8001 --model-path ./models/gemma-3-1b-it
```

## 📈 결과 분석

### 최신 성능 결과 (2025-07-18)

#### CLI 모드 성능 (Subprocess)
| 엔진 | 평균 TPS | 상대 성능 | 특징 |
|------|----------|----------|------|
| PyTorch + MPS | 7.31 | 1.0x (기준) | 높은 품질, 느린 속도 |
| Ollama (GGUF) | 74.49 | 10.2x | 균형잡힌 성능 |
| llama.cpp (Metal) | 2,337.30 | 319.5x | 압도적 속도 |

#### API 모드 성능 (Server)
| 엔진 | 평균 TPS | 상대 성능 | 특징 |
|------|----------|----------|------|
| PyTorch Server | 7.27 | 1.0x (기준) | OpenAI 호환 API |
| Ollama Server | 46.59 | 6.4x | 멀티클라이언트 지원 |
| llama.cpp Server | 71.52 | 9.8x | HTTP 오버헤드 존재 |
| Uzu Server | 26.78 | 3.7x | 네이티브 Metal 최적화 |

### 결과 파일

벤치마크 실행 후 다음 파일들이 생성됩니다:

```
report/
├── benchmark_report_10runs_20250718_151256.md
├── api_benchmark_report_10runs_20250718_151256.md
output/
├── benchmark_results_10runs_20250718_151256.json
├── api_benchmark_results_10runs_20250718_151256.json
logging/
├── benchmark_detailed_20250718_151256.log
├── api_benchmark_detailed_20250718_151256.log
```

## 🔧 문제 해결

### 일반적인 문제

#### 1. PyTorch MPS 오류
```bash
# MPS 백엔드 확인
python -c "import torch; print(torch.backends.mps.is_available())"

# 메모리 부족 시
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

#### 2. Ollama 모델 등록 실패
```bash
# 기존 모델 삭제 후 재등록
ollama rm gemma-3-1b-it-bench
ollama create gemma-3-1b-it-bench -f ./models/Modelfile
```

#### 3. llama.cpp Metal 미지원
```bash
# Metal 지원 재빌드
cd llama.cpp
make clean
make GGML_METAL=1
```

#### 4. Uzu 빌드 오류
```bash
# Xcode Command Line Tools 설치 확인
xcode-select -p
# 출력: /Applications/Xcode.app/Contents/Developer 또는 /Library/Developer/CommandLineTools

# Xcode 라이센스 동의
sudo xcodebuild -license accept

# Rust 툴체인 업데이트
rustup update stable

# Metal 컴파일 오류 시 Xcode 재설치
sudo xcode-select --reset
xcode-select --install
```

**일반적인 Uzu 빌드 오류:**

1. **Metal 셰이더 컴파일 실패**
   ```
   error: failed to run custom build command for `uzu`
   xcrun: error: unable to find utility "metal"
   ```
   **해결**: `xcode-select --install`로 Command Line Tools 설치

2. **Apple 프레임워크 링크 오류**
   ```
   error: linking with `cc` failed: exit status: 1
   ld: framework not found Metal
   ```
   **해결**: Xcode 라이센스 동의 및 개발자 도구 활성화

3. **Rust + Metal 호환성 문제**
   ```
   error: failed to compile `metal` v0.x.x
   ```
   **해결**: `rustup update`로 최신 Rust 사용

### 서버 포트 충돌

각 서버는 고유한 포트를 사용합니다:
- PyTorch: 8001
- llama.cpp: 8002  
- Uzu: 8000
- Ollama: 11434 (기본값)

## 📚 추가 정보

### 벤치마크 설정 가이드

자세한 설정 방법은 `benchmark_usage.md`를 참조하세요.

### 프로젝트 구조

```
uzu/
├── benchmark_config.json       # 통합 설정 파일
├── subprocess_benchmark.py     # CLI 벤치마크
├── api_benchmark.py           # API 벤치마크
├── server_manager.py          # 서버 관리
├── pytorch_server.py          # PyTorch 서버
├── benchmark_prompts.py       # 구조화된 프롬프트 시스템
├── benchmark_usage.md         # 상세 사용법 가이드
├── models/                    # 모델 저장소
│   ├── gemma-3-1b-it/        # HuggingFace 원본
│   ├── gemma-3-1b-it-uzu/    # Uzu 형식
│   ├── gemma-3-1b-it-gguf-llama/ # llama.cpp GGUF
│   └── Modelfile             # Ollama 모델 설정
├── report/                   # Markdown 리포트
├── output/                   # JSON 결과 데이터
└── logging/                  # 상세 로그
```

### 프롬프트 시스템 설계 철학

벤치마크의 핵심은 **공정한 비교**입니다. 이를 위해 다음과 같은 설계 원칙을 적용했습니다:

1. **동일한 입력**: 모든 엔진이 동일한 시스템 프롬프트와 사용자 질문을 받습니다
2. **일관된 제약**: 토큰 수, 온도, 언어 등 모든 생성 조건을 통일했습니다  
3. **구조화된 응답**: 300자 제한과 3단계 구조로 응답 품질을 표준화했습니다
4. **다양한 복잡도**: 짧은/중간/긴 프롬프트로 다양한 추론 능력을 테스트합니다
5. **재현 가능성**: 모든 프롬프트와 설정이 코드로 관리되어 재현 가능합니다

### 기여하기

이슈 신고나 개선 제안은 GitHub Issues를 통해 제출해주세요.

### 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

---

**마지막 업데이트**: 2025-07-18 15:12:56  
**테스트 환경**: macOS 15.5, Apple M3 Max, 36GB RAM 