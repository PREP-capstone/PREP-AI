"""
기존 라벨 데이터셋(pilot_all_labeled_completed.csv)과 신규 라벨링된
구글 플레이 데이터(google_play_apps_claude_labeled.csv)를 병합합니다.

병합 전 신규 데이터의 감사(audit)용 컬럼(predicted_*, *_confidence,
minority_candidate, labeled_by)을 제거하고, 기존 데이터셋과 동일한
13개 컬럼만 남긴 뒤 concat합니다.

축1/축2는 별도 head로 각각 학습되므로(train.py), 두 축 중 하나라도
쓸 수 있는 값이 있으면 병합에 포함합니다.
- category_id=='EXC' 이지만 function_type이 실제 의미있는 값인 행
  (예: 범용 병원예약/복약알림처럼 질환 도메인은 특정되지 않아도 축2
  기능 자체는 실재하는 경우) -> 포함. train.py가 category 손실에서
  ignore_index로 자동 제외하므로 축1 학습은 오염되지 않고, 축2 학습에는
  정상적으로 기여합니다.
- category_id=='EXC' 이고 function_type도 비어있는 행(게임/카메라필터/
  구인구직·노동매칭/전문가용 B2B 툴/쇼핑 등 완전히 무관하거나 축2 라벨도
  형식적으로만 채운 경우) -> 두 축 모두 정보가 없으므로 완전히 제외.
--strict-exclude-exc 옵션을 주면 이전처럼 category_id=='EXC'인 행을
축2 값 유무와 무관하게 전부 제외하는 이전 동작으로 되돌릴 수 있습니다.

사용법:
    python scripts/merge_datasets.py \
        --existing data/labeled/pilot_all_labeled_completed.csv \
        --new data/labeled/google_play_apps_claude_labeled.csv \
        --output pilot_all_labeled_merged.csv
"""
import argparse

import pandas as pd

CSV_COLUMNS = [
    "platform", "country", "app_id", "app_name", "description",
    "genre", "store_url", "idea_desc", "collected_data",
    "category_id", "function_type", "is_boundary_case", "note",
]


def main():
    parser = argparse.ArgumentParser(description="기존 데이터셋과 신규 라벨링 데이터 병합")
    parser.add_argument("--existing", default="data/labeled/pilot_all_labeled_completed.csv")
    parser.add_argument("--new", default="data/labeled/google_play_apps_claude_labeled.csv")
    parser.add_argument("--output", default="data/labeled/pilot_all_labeled_merged.csv")
    parser.add_argument("--strict-exclude-exc", action="store_true",
                         help="EXC 행을 축2 값 유무와 무관하게 전부 제외하는 이전 동작으로 되돌림")
    args = parser.parse_args()

    existing = pd.read_csv(args.existing, dtype={"category_id": str})
    new_full = pd.read_csv(args.new, dtype={"category_id": str})

    is_exc = new_full["category_id"] == "EXC"
    has_function = new_full["function_type"].notna() & (new_full["function_type"].astype(str).str.strip() != "")
    if args.strict_exclude_exc:
        drop_mask = is_exc
    else:
        drop_mask = is_exc & ~has_function
    dropped = int(drop_mask.sum())
    new_full = new_full[~drop_mask]
    new = new_full[CSV_COLUMNS]

    merged = pd.concat([existing, new], ignore_index=True)
    print(f"신규 데이터 중 병합 제외: {dropped}건"
          + ("(--strict-exclude-exc: EXC 전부 제외)" if args.strict_exclude_exc
             else "(EXC이면서 function_type도 없는 완전 무관 행만 제외)"))
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"기존: {len(existing)}건 + 신규: {len(new)}건 = 병합: {len(merged)}건 -> '{args.output}'")
    print("\n병합 후 category_id 분포:")
    print(merged["category_id"].value_counts())
    print("\n병합 후 function_type 분포:")
    print(merged["function_type"].value_counts())


if __name__ == "__main__":
    main()
