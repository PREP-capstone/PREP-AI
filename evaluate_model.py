"""train.py가 저장한 체크포인트를 train.py와 동일한 val split으로 재평가해
축1/축2 클래스별(macro가 아닌 per-class) F1을 뽑는다.

macro F1만으로는 어떤 클래스가 평균을 깎아먹는지 안 보이기 때문에,
sklearn classification_report로 클래스별 precision/recall/F1/support를
확인하고, 축2는 category_id(EXC 포함)별로도 쪼개서 "EXC에서 끌어온
애매한 라벨이 축2를 흐리는지" 아니면 "특정 클래스 표본 부족이 원인인지"
구분하는 데 쓴다.

사용법:
    python evaluate_model.py --model-dir best_healthcare_model_2line
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast

from train import (
    CATEGORY_IGNORE_INDEX,
    FUNCTION_TYPE_MAP,
    HealthcareDataset,
    MultiHeadHealthcareModel,
)

CATEGORY_NAME_MAP = {0: "수면", 1: "정신건강", 2: "운동", 3: "식단", 4: "만성질환", 5: "여성건강", 6: "유전자", 7: "미용"}
FUNCTION_NAME_MAP = {0: "A(정보제공)", 1: "B(기록관리)", 2: "C(매칭연결)", 3: "D(개입치료)"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="best_healthcare_model_2line")
    parser.add_argument("--data", default="pilot_all_labeled_completed.csv")
    parser.add_argument("--max-len", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train.py와 완전히 동일한 전처리 + split (random_state=42로 재현)
    df = pd.read_csv(args.data)
    category_classes = sorted(int(c) for c in df["category_id"].unique() if c != "EXC")
    category_to_label = {c: i for i, c in enumerate(category_classes)}

    def map_category(value):
        return CATEGORY_IGNORE_INDEX if value == "EXC" else category_to_label[int(value)]

    df["category_label"] = df["category_id"].apply(map_category)
    df["function_label"] = df["function_type"].map(FUNCTION_TYPE_MAP)
    df["collected_data"] = df["collected_data"].fillna("")
    df["combined_text"] = df["description"] + " [SEP] 수집 데이터: " + df["collected_data"]

    _, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["category_id"])
    val_df = val_df.reset_index(drop=True)

    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_dir)
    label_config = torch.load(f"{args.model_dir}/label_config.pt", weights_only=False)
    model = MultiHeadHealthcareModel(
        label_config["model_name"], label_config["num_category_labels"], label_config["num_function_labels"],
    )
    model.load_state_dict(torch.load(f"{args.model_dir}/model.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    val_dataset = HealthcareDataset(val_df, tokenizer, max_len=args.max_len)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

    cat_preds, cat_trues, func_preds, func_trues = [], [], [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cat_logits, func_logits = model(input_ids, attention_mask)
            cat_pred = torch.argmax(cat_logits, dim=1).cpu().numpy()
            func_pred = torch.argmax(func_logits, dim=1).cpu().numpy()

            cat_true = batch["category_labels"].numpy()
            valid = cat_true != CATEGORY_IGNORE_INDEX
            cat_preds.extend(cat_pred[valid])
            cat_trues.extend(cat_true[valid])
            func_preds.extend(func_pred)
            func_trues.extend(batch["function_labels"].numpy())

    lines = []
    lines.append("=== 축1(category_id) 클래스별 성능 (EXC 제외) ===")
    cat_names = [CATEGORY_NAME_MAP[category_classes[i]] for i in sorted(set(cat_trues))]
    lines.append(classification_report(cat_trues, cat_preds, target_names=cat_names, zero_division=0))

    lines.append("=== 축2(function_type) 클래스별 성능 (EXC 포함 전체 val) ===")
    func_names = [FUNCTION_NAME_MAP[i] for i in sorted(set(func_trues) | set(func_preds))]
    lines.append(classification_report(func_trues, func_preds, target_names=func_names, zero_division=0))

    # 축2를 "실제 카테고리 행" vs "EXC 행"으로 나눠서 따로 평가 -> EXC 노이즈 여부 확인
    val_df["func_pred"] = np.nan
    val_df.loc[val_df.index[:len(func_preds)], "func_pred"] = func_preds  # index는 val_loader와 동일 순서(shuffle 없음)
    is_exc = (val_df["category_id"] == "EXC").values
    func_trues_arr = np.array(func_trues)
    func_preds_arr = np.array(func_preds)

    lines.append("=== 축2 성능 비교: 실제 카테고리 행 vs EXC 행 ===")
    for label, mask in [("실제 카테고리(EXC 아님)", ~is_exc), ("EXC 행", is_exc)]:
        n = mask.sum()
        if n == 0:
            continue
        report = classification_report(
            func_trues_arr[mask], func_preds_arr[mask],
            labels=sorted(set(func_trues_arr[mask])),
            target_names=[FUNCTION_NAME_MAP[i] for i in sorted(set(func_trues_arr[mask]))],
            zero_division=0, output_dict=True,
        )
        lines.append(f"[{label}] n={n}, macro F1={report['macro avg']['f1-score']:.4f}")

    with open("evaluate_result.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("done -> evaluate_result.txt")


if __name__ == "__main__":
    main()
