"""
구글 플레이 원본 수집 데이터(google_play_apps_raw.csv)에 대해
학습된 멀티헤드 모델로 축1(category_id)/축2(function_type)를 사전 예측합니다.

사람이 수백 건을 처음부터 라벨링하는 대신, 모델 예측 + 확신도(confidence)를
참고해서 확인/수정만 하도록 돕는 것이 목적입니다. 확신도가 낮은 순으로
정렬해 저장하므로, 위쪽부터 검토하면 애매한 케이스를 먼저 챙길 수 있습니다.

설치:
    pip install torch transformers

사용법:
    python predict_labels.py \
        --input google_play_apps_raw.csv \
        --checkpoint best_healthcare_model \
        --output google_play_apps_prelabeled.csv

주의:
- 이 스크립트가 채우는 predicted_category_id / predicted_function_type /
  category_confidence / function_confidence / minority_candidate 컬럼은
  참고용 "제안"입니다. 최종 category_id / function_type 컬럼은 사람이
  검토 후 직접 채워야 합니다.
- 카테고리 정의는 LABELING_GUIDE.md 참고.
"""
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizerFast

from train import MultiHeadHealthcareModel

CATEGORY_NAME_MAP = {
    0: "수면", 1: "정신건강", 2: "운동", 3: "식단",
    4: "만성질환", 5: "여성건강", 6: "유전자", 7: "미용",
}
FUNCTION_NAME_MAP = {"A": "정보제공", "B": "데이터기록관리", "C": "매칭연결", "D": "개입치료"}

# 표본이 극소수인 카테고리(유전자=6, 미용=7)를 우선 검토할 수 있도록 표시하는 키워드
MINORITY_KEYWORDS = {
    "유전자": ["유전자", "dna", "유전", "genetic"],
    "미용": ["피부", "시술", "미용", "성형", "뷰티", "skin", "beauty"],
}


def load_model(checkpoint_dir, device):
    label_config = torch.load(f"{checkpoint_dir}/label_config.pt", weights_only=False)
    tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint_dir)
    model = MultiHeadHealthcareModel(
        label_config["model_name"],
        label_config["num_category_labels"],
        label_config["num_function_labels"],
    )
    state_dict = torch.load(f"{checkpoint_dir}/model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    category_classes = label_config["category_classes"]  # index -> category_id
    function_type_map = label_config["function_type_map"]  # 'A' -> 0 등
    idx_to_function = {v: k for k, v in function_type_map.items()}
    return model, tokenizer, category_classes, idx_to_function


def flag_minority_candidate(text):
    text = str(text).lower()
    hits = [name for name, kws in MINORITY_KEYWORDS.items() if any(kw in text for kw in kws)]
    return ",".join(hits)


def main():
    parser = argparse.ArgumentParser(description="멀티헤드 모델로 축1/축2 사전 예측")
    parser.add_argument("--input", default="google_play_apps_raw.csv")
    parser.add_argument("--checkpoint", required=True, help="학습된 모델 체크포인트 폴더 (model.pt, label_config.pt, tokenizer 포함)")
    parser.add_argument("--output", default="google_play_apps_prelabeled.csv")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="테스트용: 앞에서부터 N개만 처리")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer, category_classes, idx_to_function = load_model(args.checkpoint, device)

    df = pd.read_csv(args.input)
    if args.limit:
        df = df.head(args.limit).copy()
    df["collected_data"] = df["collected_data"].fillna("")
    df["description"] = df["description"].fillna("")
    combined = df["description"] + " [SEP] 수집 데이터: " + df["collected_data"]

    pred_category_ids, pred_category_names = [], []
    pred_function_types, pred_function_names = [], []
    category_confidences, function_confidences = [], []

    total = len(combined)
    with torch.no_grad():
        for i, text in enumerate(combined.tolist(), start=1):
            enc = tokenizer(
                str(text), add_special_tokens=True, max_length=args.max_len,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            category_logits, function_logits = model(input_ids, attention_mask)

            cat_probs = F.softmax(category_logits, dim=1)[0]
            func_probs = F.softmax(function_logits, dim=1)[0]

            cat_idx = int(torch.argmax(cat_probs))
            func_idx = int(torch.argmax(func_probs))

            category_id = category_classes[cat_idx]
            function_type = idx_to_function[func_idx]

            pred_category_ids.append(category_id)
            pred_category_names.append(CATEGORY_NAME_MAP.get(category_id, ""))
            pred_function_types.append(function_type)
            pred_function_names.append(FUNCTION_NAME_MAP.get(function_type, ""))
            category_confidences.append(float(cat_probs[cat_idx]))
            function_confidences.append(float(func_probs[func_idx]))

            if i % 50 == 0 or i == total:
                print(f"[예측] {i}/{total} 완료")

    df["predicted_category_id"] = pred_category_ids
    df["predicted_category_name"] = pred_category_names
    df["predicted_function_type"] = pred_function_types
    df["predicted_function_name"] = pred_function_names
    df["category_confidence"] = category_confidences
    df["function_confidence"] = function_confidences
    df["min_confidence"] = df[["category_confidence", "function_confidence"]].min(axis=1)
    df["minority_candidate"] = (
        df["description"].astype(str) + " " + df["app_name"].astype(str)
    ).apply(flag_minority_candidate)

    # 확신도가 낮은 순(=검토 우선순위 높은 순)으로 정렬
    df = df.sort_values("min_confidence", ascending=True).reset_index(drop=True)

    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"완료: {len(df)}건에 대해 예측 완료 -> '{args.output}'")
    print(
        f"평균 category 확신도: {df['category_confidence'].mean():.3f}, "
        f"평균 function 확신도: {df['function_confidence'].mean():.3f}"
    )
    print(f"소수 클래스(유전자/미용) 키워드 후보: {(df['minority_candidate'] != '').sum()}건")


if __name__ == "__main__":
    main()
