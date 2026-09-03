"""
기존 라벨 데이터셋(pilot_all_labeled_completed.csv)과 신규 라벨링된
구글 플레이 데이터(google_play_apps_claude_labeled.csv)를 병합합니다.

병합 전 신규 데이터의 감사(audit)용 컬럼(predicted_*, *_confidence,
minority_candidate, labeled_by)을 제거하고, 기존 데이터셋과 동일한
13개 컬럼만 남긴 뒤 concat합니다.

기본적으로 신규 데이터 중 category_id=='EXC'인 행은 제외합니다
(--include-exc로 포함 가능). EXC 행은 사람이 검토하지 않은 AI 초안
판단이라, 안전하게 축1(category_id) 8종에 해당하는 행만 우선 반영합니다.
(참고: train.py는 기존 EXC 행을 category 손실에서 ignore_index로
자동 제외하므로, 나중에 검토 후 포함시켜도 코드 수정은 필요 없습니다.)

사용법:
    python merge_datasets.py \
        --existing pilot_all_labeled_completed.csv \
        --new google_play_apps_claude_labeled.csv \
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
    parser.add_argument("--existing", default="pilot_all_labeled_completed.csv")
    parser.add_argument("--new", default="google_play_apps_claude_labeled.csv")
    parser.add_argument("--output", default="pilot_all_labeled_merged.csv")
    parser.add_argument("--include-exc", action="store_true", help="신규 데이터의 EXC 행도 포함 (기본: 제외)")
    args = parser.parse_args()

    existing = pd.read_csv(args.existing, dtype={"category_id": str})
    new_full = pd.read_csv(args.new, dtype={"category_id": str})
    excluded_exc = (new_full["category_id"] == "EXC").sum()
    if not args.include_exc:
        new_full = new_full[new_full["category_id"] != "EXC"]
    new = new_full[CSV_COLUMNS]

    merged = pd.concat([existing, new], ignore_index=True)
    if not args.include_exc:
        print(f"신규 데이터의 EXC {excluded_exc}건은 제외했습니다 (--include-exc로 포함 가능)")
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"기존: {len(existing)}건 + 신규: {len(new)}건 = 병합: {len(merged)}건 -> '{args.output}'")
    print("\n병합 후 category_id 분포:")
    print(merged["category_id"].value_counts())
    print("\n병합 후 function_type 분포:")
    print(merged["function_type"].value_counts())


if __name__ == "__main__":
    main()
