# PREP-AI
PREP 카테고리 분류 AI 모델

## 📁 폴더 구조

```
scripts/    실행 스크립트 (수집/라벨링/병합/학습/평가/ONNX 변환) — 전부 저장소 루트에서 실행
  scrape_google_play.py       구글 플레이 앱 수집
  predict_labels.py           학습된 모델로 라벨 사전예측(사람 검토 보조용)
  merge_datasets.py           기존 데이터셋 + 신규 라벨링 데이터 병합
  train.py                    멀티헤드 분류 모델 학습
  train_large_advanced.py     대형/고급 버전 학습 실험용
  evaluate_model.py           체크포인트 클래스별(per-class) F1 진단
  export_onnx.py              PyTorch 체크포인트 -> ONNX 변환
  quantize_onnx.py            ONNX int8 동적 양자화

data/
  raw/        스크래핑 원본(라벨 없음) + 모델 사전예측만 붙은 prelabeled
  labeled/    사람 검토 완료된 라벨 데이터, pilot_all_labeled_completed.csv(현재 학습셋)
  backups/    이전 버전 학습셋 스냅샷 (재현/비교용)

models/     학습된 체크포인트 (.gitignore 대상, train.py로 재생성)
logs/       학습 실행 로그
docs/       라벨링 가이드, 학습 기록, ONNX 경량화 문서
```

사용 예: `python scripts/train.py`, `python scripts/merge_datasets.py --existing data/labeled/pilot_all_labeled_completed.csv ...`

## 🚀 Git 컨벤션 규칙

### Commit 규칙

| Gitmoji | Tag | Description |
|:-------:|:---:| --- |
| ✨ | `feat` | 새로운 기능 추가 |
| 🔧 | `fix` | 버그 수정 |
| 🐛 | `bug` | 버그 이슈 |
| 📋 | `docs` | 문서 추가, 수정, 삭제 |
| ✅ | `test` | 테스트 코드 추가, 수정, 삭제 |
| ♻️ | `refactor` | 코드 리팩토링 |
| ⚙️ | `chore` | 설정 및 기타 변경사항 |
| 🔄 | `ci-cd` | CI/CD 관련 설정 수정 |

#### Commit Message Format
- **헤더(Header)**: `<타입>(스코프): <주제>`
- **본문(Body)**: 커밋의 상세 내용 (선택적)
- **바닥글(Footer)**: 관련 이슈 번호
