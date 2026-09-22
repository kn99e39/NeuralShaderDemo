# Worklog — macOS Execution Compatibility Audit (2026-09-22)

## 목적

Batch 1/2 및 공용 RNA 실험 도구가 현재 Apple Silicon macOS에서 실행 가능한 구조인지, 코드와 로컬 실행 환경을 기준으로 점검했다.

## 판정

전체 RNA 학습·추론 파이프라인은 현재 macOS 지원 구조가 아니다. 설정과 renderer가 CUDA를 기본 또는 고정 대상으로 삼고 있으며, Apple Silicon의 PyTorch MPS를 선택하거나 검증하는 경로가 없다. CPU로 가능한 일부 후처리/검토 도구와 Blender 장면 생성 가능성은 전체 RNA pipeline 호환 판정을 바꾸지 않는다.

## 근거

- `shared/dynamic_renderer.py`의 `--device` 기본값은 `cuda`다.
- Batch 1의 세 training 설정 3개와 Batch 2의 training 설정 3개가 모두 `device: cuda`, `ops_device: cuda`를 고정한다.
- `tools/capture_environment.py`는 CUDA runtime, `nvidia-smi`, Ubuntu release를 수집하도록 작성되어 있고, README도 official RNA 실행을 Ubuntu/WSL에서 하도록 안내한다.
- `tools/run_with_seed.py`와 `shared/metrics.py`는 CUDA가 없을 때 각각 host-side seed 처리 또는 LPIPS CPU fallback을 제공하지만, 이는 RNA renderer/training의 MPS 이식이 아니다.
- 현재 macOS 26.5.2 / arm64 환경에는 `blender`, `ffmpeg`, `nvidia-smi` 실행 파일이 PATH에 없고, 시스템 Python에는 numpy, torch, bpy, imageio, matplotlib, Pillow, pyexr, scipy, tensorboard, omegaconf가 설치되어 있지 않았다.
- official RNA 외부 저장소 `external/relightable-neural-assets`도 현재 작업 사본에는 존재하지 않아, upstream이 macOS/MPS를 별도로 지원하는지는 이 세션에서 실행 검증할 수 없었다.

## 실행 범위와 다음 과제

공용 correspondence 로직 및 이미지 검토 도구는 OS 고정 경로를 사용하지 않아, 적절한 Python 의존성과 FFmpeg를 설치하면 macOS CPU에서 실행할 수 있을 가능성이 높다. Blender 기반 asset 생성도 macOS Blender의 Python 환경과 official RNA 의존성이 호환될 경우 별도로 확인할 수 있다.

Mac에서 전체 pipeline을 지원하려면 official RNA의 CUDA 전용 연산 및 renderer 의존성을 먼저 조사하고, MPS 지원 가능 여부를 결정해야 한다. 가능하다면 device resolver, MPS-compatible config, macOS 환경 설정 및 작은 smoke render를 추가로 구현·검증해야 한다. 현 단계에서는 이를 수행하지 않았으며, 성능·gate·case 판정도 없다.
