"""train.py가 저장한 멀티헤드 체크포인트(model.pt + label_config.pt)를 ONNX로 변환한다.

체크포인트 구조(train.py 참고):
    <model-dir>/model.pt          — MultiHeadHealthcareModel.state_dict()
    <model-dir>/label_config.pt   — {"model_name", "num_category_labels", "num_function_labels", ...}

사용법:
    python export_onnx.py --model-dir best_healthcare_model_2line --out model_fp32.onnx

주의: 더미 입력은 토크나이저 대신 encoder.config.vocab_size 범위의 무작위 정수로
직접 만든다 — tokenizer_config.json의 tokenizer_class 오기재로 AutoTokenizer가
잘못된 토크나이저를 고르는 문제(PREP-BE app/domain/category_classifier.py 주석
참고)가 있어, 트레이싱 단계에서는 실제 토큰 값이 중요하지 않으므로 아예
토크나이저 로딩 자체를 건너뛴다.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from transformers import AutoModel


class MultiHeadHealthcareModel(nn.Module):
    """train.py와 동일한 구조 — 공유 인코더 + 축1(category) 헤드 + 축2(function) 헤드."""

    def __init__(self, model_name: str, num_category_labels: int, num_function_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.category_head = nn.Linear(hidden_size, num_category_labels)
        self.function_head = nn.Linear(hidden_size, num_function_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # pooler_output 아님 — 학습 시 그대로 사용한 방식
        pooled = self.dropout(cls_output)
        return self.category_head(pooled), self.function_head(pooled)


def load_checkpoint(model_dir: str) -> MultiHeadHealthcareModel:
    label_config = torch.load(os.path.join(model_dir, "label_config.pt"), map_location="cpu", weights_only=False)
    model = MultiHeadHealthcareModel(
        label_config["model_name"],
        label_config["num_category_labels"],
        label_config["num_function_labels"],
    )
    state_dict = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="best_healthcare_model_2line")
    parser.add_argument("--out", default="model_fp32.onnx")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model = load_checkpoint(args.model_dir)

    vocab_size = model.encoder.config.vocab_size
    input_ids = torch.randint(low=0, high=vocab_size, size=(1, args.max_len), dtype=torch.long)
    attention_mask = torch.ones((1, args.max_len), dtype=torch.long)

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        args.out,
        input_names=["input_ids", "attention_mask"],
        output_names=["category_logits", "function_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "category_logits": {0: "batch"},
            "function_logits": {0: "batch"},
        },
        opset_version=args.opset,
        dynamo=False,  # torch 2.x 기본 dynamo 익스포터는 onnxscript 의존성을 추가로 요구한다.
    )
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"ONNX 저장 완료: {args.out} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
