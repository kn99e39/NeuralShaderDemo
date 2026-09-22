# Worklog — Batched Experiment Layout (2026-09-22)

## 목적

동적 transport validity 실험의 입력, 자산, 스크립트가 한 디렉터리에 혼재되어 있던 구조를 배치 중심으로 정리해 실험 이력과 실행 대상을 빠르게 식별할 수 있게 했다.

## 수행 내용

`experiments/dynamic_transport_validity/`를 다음 책임으로 분리했다.

- `batch1/`: Lego와 synthetic folding-sheet 제어군, G4 refit 비교, Batch 1 자산 및 설정.
- `batch2/`: A/B/C authored deformation 군, Batch 2 자산·설정·생성·ROI 검토 코드.
- `shared/`: 두 배치가 함께 쓰는 renderer, correspondence, metrics, compatibility, review 도구.
- `tools/`: 환경 캡처, checkpoint 검사, artifact 관리, seed launcher, training 요약.

모든 YAML의 자산 및 evaluation JSON 경로와 이동한 Python import/workspace 계산을 새 위치로 갱신했다. 최상위 README에도 새 구조와 대표 실행 경로를 기록했다.

## 검증과 한계

캐시를 생성하지 않는 Python 구문 검사에서 24개 Python 파일이 통과했고, 설정에 기록된 workspace 경로 24개가 모두 존재함을 확인했다. 이전 `assets/`, `configs/` 및 Batch 2 스크립트의 옛 경로를 가리키는 참조도 남지 않았다.

단위 테스트는 현재 시스템 Python에 `numpy`가 없어 import 단계에서 실행되지 않았다. Blender/RNA와 연구용 Python 의존성이 준비된 Ubuntu/WSL 환경에서 실제 renderer 및 metrics 실행을 별도로 확인해야 하며, 이번 세션은 실험 성능이나 gate 판정을 수행하지 않았다.
