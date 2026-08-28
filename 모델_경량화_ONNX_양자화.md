# 카테고리 분류 모델 경량화 (ONNX + int8 동적 양자화)

## 배경

PREP-BE는 EC2 t3.small(vCPU 2, RAM 2GB)에 배포될 예정이다. `train.py`가 만드는
멀티헤드 체크포인트(`klue/roberta-base` 인코더 + category_head(8종) +
function_head(4종))를 raw PyTorch(`transformers.AutoModel`)로 그대로 서빙하면
메모리 여유가 빠듯하다 — 아래 실측 수치 참고. ONNX로 변환 후 int8 동적
양자화하면 파일 크기·런타임 메모리 둘 다 크게 줄어든다.

`.gitignore` 정책상 실제 체크포인트(`best_healthcare_model_2line/` 등)와
변환된 `.onnx` 파일은 이 저장소에 커밋하지 않는다(수백MB 단위 바이너리 —
GitHub 100MB/파일 제한도 걸림). 이 문서 + `export_onnx.py`/`quantize_onnx.py`
스크립트로 누구나 로컬에서 재생성한다.

## 사용법

```bash
# 1. train.py로 학습된 체크포인트에서 ONNX(fp32) 추출
python export_onnx.py --model-dir best_healthcare_model_2line --out model_fp32.onnx

# 2. int8 동적 양자화
python quantize_onnx.py --in model_fp32.onnx --out model_int8.onnx
```

추론 시 입력은 `input_ids`/`attention_mask` (둘 다 `int64`, shape `[batch, seq_len]`)이고
출력은 `category_logits`(8종)/`function_logits`(4종)이다 — `torch.softmax` +
`argmax`로 최종 라벨을 뽑는 건 기존 PyTorch 추론과 동일하다.

## 실측 결과 (2026-08-23, `best_healthcare_model_2line` 체크포인트 기준)

### 파일 크기

| 형태 | 크기 | 비고 |
|---|---|---|
| PyTorch 체크포인트(`model.pt`) | 422MB | fp32 원본 |
| ONNX fp32 | 420.0MB | 변환만, 양자화 전 |
| ONNX int8 | 105.7MB | 동적 양자화, **-74.8%** |

### 추론 1회 기준 프로세스 피크 메모리(RSS, macOS 로컬 측정)

| 방식 | 피크 RSS |
|---|---|
| PyTorch + transformers(`AutoModel`) 로드 후 추론 | 1117.4MB |
| ONNX Runtime(fp32) 로드 후 추론 | 639.0MB |
| ONNX Runtime(int8) 로드 후 추론 | **230.2MB** |

PyTorch 대비 ONNX int8은 **약 79%** 메모리 절감 — t3.small(RAM 2GB, 앱 서버 자체·
DB 커넥션 등 다른 프로세스와 공유)에 훨씬 안전한 여유를 만들어준다.

### 정확도(수치 동등성) 검증

무작위 입력(고정 시드) 1건으로 PyTorch 출력과 ONNX 출력을 직접 비교했다:

- **fp32**: PyTorch와 완전히 동일(`max|diff| = 0.0000`), argmax 라벨도 동일.
- **int8**: 로짓 값 자체는 소폭 달라짐(`max|diff|` category 0.46 / function 0.69)
  — 동적 양자화가 가중치를 int8로 반올림하는 데서 오는 정상적인 오차. 다만
  이 샘플에서 **최종 예측 라벨(argmax)은 fp32/int8 모두 PyTorch와 동일**했다.
  실서비스 적용 전에는 실제 검증셋(오프라인 校정 데이터)으로 Macro F1이
  유의미하게 떨어지지 않는지 한 번 더 확인을 권장한다 — 이 문서의 수치는
  "변환 파이프라인 자체가 정상 동작한다"는 것만 보증한다.

## 알려진 함정

- **`torch.onnx.export`에 `dynamo=False` 필수** — PyTorch 2.x부터 기본값이
  `torch.export` 기반 dynamo 익스포터로 바뀌었는데, 이게 `onnxscript` 패키지를
  추가로 요구한다. `dynamo=False`로 기존 TorchScript 기반 익스포터를 쓰면
  별도 설치 없이 바로 된다(단, PyTorch 2.9+에서는 legacy 경로라는
  `DeprecationWarning`이 뜬다 — 아직은 정상 동작).
- **더미 입력에 AutoTokenizer를 쓰지 않는다** — 이 체크포인트의
  `tokenizer_config.json`은 `tokenizer_class`가 실제 vocab(BERT WordPiece)과
  다르게 기재돼 있어(PREP-BE `app/domain/category_classifier.py` 참고),
  `AutoTokenizer`로 로드하면 잘못된 토크나이저가 선택될 수 있다. ONNX 변환은
  텐서 shape/dtype만 맞으면 되므로, `export_onnx.py`는 아예 토크나이저를
  거치지 않고 `vocab_size` 범위의 무작위 정수로 더미 입력을 만든다 — 실제
  추론 서빙 코드(`category_classifier.py`)에서는 여전히 `BertTokenizerFast`를
  명시적으로 써야 한다.
- **`quantize_dynamic`이 "pre-processing 먼저 하라"는 경고를 띄운다** —
  `onnxruntime.quantization.shape_inference` 전처리를 생략해도 위 검증에서
  보듯 정상 동작하지만, 그래프가 더 복잡해지면(분기 많은 모델 등) 전처리를
  거치는 게 안전하다.

## 향후 계획

- 다른 데스크탑에서 학습 중인 개선 버전(category_2 정확도 향상)이 완성되면,
  그 최종 체크포인트 기준으로 위 스크립트를 다시 돌려 ONNX(int8) 하나만
  생성한다.
- 실제 `.onnx` 파일 자체는 이 저장소의 기존 방침(체크포인트 git 미포함)과
  GitHub 100MB/파일 제한 때문에 여기 커밋하지 않는다 — Git LFS 또는 GitHub
  Release 자산으로 올릴지는 팀 논의 후 결정 예정.
