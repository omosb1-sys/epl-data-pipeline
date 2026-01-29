---
name: hardware-optimizer
description: 8GB RAM Mac 환경을 위한 하드웨어 최적화 및 고성능 데이터 처리 스킬
user-invocable: true
---

# ⚡ Hardware Optimizer: 8GB RAM Mac 전용 고성능 프로토콜

이 스킬은 8GB RAM MacBook Air 환경에서 대규모 데이터(크몽 차량 데이터 등)를 지연 없이 처리하기 위한 안티그래비티의 하드웨서 가속 최적화 지침입니다.

## 🚀 1. 고성능 데이터 엔진 (Polars & DuckDB)
1.  **Zero-Copy Loading**: `Polars`와 `DuckDB`의 Lazy API를 활용하여 메모리 복사 없이 필요한 시점에만 데이터를 로드합니다.
2.  **Parquet First**: 무거운 Excel이나 CSV 대신 `Parquet` 파일 포맷을 기본 저장소로 사용하여 파일 I/O 속도를 10배 이상 가속합니다.
3.  **Streaming Inference**: 대규모 집계 작업 시 `streaming=True` 옵션을 강제하여 메모리 부족(OOM) 현상을 방지합니다.

## 💻 2. 리소스 관리 (Resource Guarding)
1.  **Garbage Collection**: 대규모 데이터 변환 작업 직후 `gc.collect()`를 호출하여 스왑 메모리 발생을 선제적으로 차단합니다.
2.  **Memory Mapping**: 큰 파일을 읽을 때 `mmap` 기술이 내장된 라이브러리를 우선 채택하여 물리적 램 소모를 최소화합니다.
3.  **Attention Sink**: 에이전틱 작업 시 불필요한 과거 컨텍스트를 주기적으로 플러싱(Flushing)하여 시스템 리소스를 확보합니다. (Rule 9.4 연계)

## 🏎️ 3. Mac GPU 가속 (MPS Integration)
1.  **Metal Performance Shaders (MPS)**: PyTorch나 TensorFlow 작업 시 `device='mps'`를 사용하여 Mac의 내장 GPU 연산 성능을 활용합니다.
2.  **WebGPU Rendering**: 대규모 데이터 시각화 시 CPU 대신 브라우저의 GPU 가속(ChartGPU 등)을 활용하도록 지시합니다. (Rule 8.16 연계)

---
*Refactored by Antigravity (Mac Hardware Efficiency Lab) - 2026.01.26*
