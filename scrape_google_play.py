"""
구글 플레이 스토어 헬스케어 앱 수집 스크립트

pilot_all_labeled_completed.csv 는 iOS(App Store) 데이터만 포함하고 있어
안드로이드 전용 앱이 데이터셋에서 완전히 빠져 있습니다. 이 스크립트는
google-play-scraper 패키지로 플레이 스토어의 건강/의료 관련 앱을 수집해
동일한 컬럼 스키마의 CSV로 저장합니다.

라벨 컬럼(category_id, function_type, is_boundary_case, note)은 비워둔
채로 저장되며, 수동 라벨링을 거친 뒤 pilot_all_labeled_completed.csv와
병합해서 학습 데이터로 사용하는 것을 전제로 합니다.

설치:
    pip install google-play-scraper

실행 예시:
    python scrape_google_play.py
    python scrape_google_play.py --output google_play_apps_raw.csv --n-hits 30 --sleep 0.5
    python scrape_google_play.py --keywords-file my_keywords.txt

주의:
- Google 비공식 스크래핑이라 과도한 요청 시 일시적으로 차단될 수 있습니다.
  기본 --sleep 값(0.4초)을 유지하거나 필요하면 늘려서 사용하세요.
- 검색 키워드는 기존 데이터셋의 축1(category_id 0~7)이 다루는 디지털 헬스케어
  영역을 폭넓게 커버하도록 구성했지만, 정확한 카테고리 정의가 코드에 문서화되어
  있지 않아 실제 라벨 분포는 수집 후 검수가 필요합니다. 특히 표본이 극히 적은
  category_id=6, 7에 해당하는 앱 유형을 발견하면 KEYWORDS에 키워드를 추가해주세요.
"""
import argparse
import csv
import sys
import time

from google_play_scraper import app, search
from google_play_scraper.exceptions import NotFoundError

# pilot_all_labeled_completed.csv 와 동일한 컬럼 스키마
CSV_COLUMNS = [
    "platform", "country", "app_id", "app_name", "description",
    "genre", "store_url", "idea_desc", "collected_data",
    "category_id", "function_type", "is_boundary_case", "note",
]

# 디지털 헬스케어 전 영역을 폭넓게 커버하기 위한 기본 검색 키워드
KEYWORDS = [
    # 만성질환/질병 관리
    "당뇨 관리", "혈압 관리", "만성질환 관리", "암 환자 관리",
    # 복약/처방
    "복약관리", "복약알림", "처방전 관리",
    # 정신건강
    "정신건강", "우울증 관리", "명상 앱", "스트레스 관리",
    # 원격의료/상담
    "원격의료", "비대면 진료", "의료 상담",
    # 운동/피트니스
    "홈트레이닝", "운동 기록", "다이어트 관리",
    # 임신/육아
    "임신 관리", "산모 건강", "육아 건강",
    # 수면
    "수면 관리", "수면 기록",
    # 재활/물리치료
    "재활 운동", "물리치료",
    # 영양/식단
    "식단 관리", "영양제 관리", "칼로리 계산",
    # 웨어러블 연동/생체 데이터
    "혈당 측정", "심박수 측정", "건강검진 기록",
    # 유전자/정밀의료
    "유전자 검사", "유전자 분석",
    # 응급/돌봄
    "응급 대처", "노인 돌봄", "간병",
    # 만성 통증/희귀질환
    "통증 관리", "희귀질환",
]


def collect_app_ids(keywords, lang, country, n_hits, sleep_sec):
    """검색 키워드로 후보 app_id를 모아 중복 제거한 뒤 반환합니다."""
    seen = {}
    for kw in keywords:
        try:
            results = search(kw, lang=lang, country=country, n_hits=n_hits)
        except Exception as e:
            print(f"[검색 실패] '{kw}': {e}", file=sys.stderr)
            continue
        for r in results:
            seen.setdefault(r["appId"], kw)
        print(f"[검색] '{kw}' -> {len(results)}건 (누적 고유 앱 {len(seen)}개)")
        time.sleep(sleep_sec)
    return seen


def fetch_app_details(app_ids, lang, country, sleep_sec):
    """app_id 목록에 대해 상세 정보를 조회해 CSV 스키마에 맞는 dict 리스트로 반환합니다."""
    rows = []
    total = len(app_ids)
    for i, app_id in enumerate(app_ids, start=1):
        try:
            detail = app(app_id, lang=lang, country=country)
        except NotFoundError:
            print(f"[{i}/{total}] {app_id}: 앱을 찾을 수 없음 (건너뜀)", file=sys.stderr)
            continue
        except Exception as e:
            print(f"[{i}/{total}] {app_id}: 조회 실패 ({e})", file=sys.stderr)
            continue

        rows.append({
            "platform": "android",
            "country": country,
            "app_id": app_id,
            "app_name": detail.get("title", ""),
            "description": detail.get("description") or "",
            "genre": detail.get("genre") or "",
            "store_url": detail.get("url")
                or f"https://play.google.com/store/apps/details?id={app_id}&hl={lang}&gl={country.upper()}",
            "idea_desc": "",
            "collected_data": "",
            "category_id": "",
            "function_type": "",
            "is_boundary_case": "",
            "note": "",
        })
        if i % 10 == 0 or i == total:
            print(f"[상세 조회] {i}/{total} 완료")
        time.sleep(sleep_sec)
    return rows


def main():
    parser = argparse.ArgumentParser(description="구글 플레이 스토어 헬스케어 앱 수집")
    parser.add_argument("--output", default="google_play_apps_raw.csv", help="저장할 CSV 경로")
    parser.add_argument("--lang", default="ko", help="검색/조회 언어 (기본: ko)")
    parser.add_argument("--country", default="kr", help="검색/조회 국가 코드 (기본: kr)")
    parser.add_argument("--n-hits", type=int, default=30, help="키워드당 최대 검색 결과 수 (기본: 30, Play 검색 API 상한도 30)")
    parser.add_argument("--sleep", type=float, default=0.4, help="요청 사이 대기 시간(초), 차단 방지용 (기본: 0.4)")
    parser.add_argument("--keywords-file", default=None, help="키워드를 한 줄에 하나씩 담은 텍스트 파일 (지정 시 기본 KEYWORDS 대신 사용)")
    args = parser.parse_args()

    keywords = KEYWORDS
    if args.keywords_file:
        with open(args.keywords_file, encoding="utf-8") as f:
            keywords = [line.strip() for line in f if line.strip()]

    print(f"총 {len(keywords)}개 키워드로 검색을 시작합니다...")
    app_ids = collect_app_ids(keywords, args.lang, args.country, args.n_hits, args.sleep)
    print(f"검색 완료: 고유 앱 {len(app_ids)}개. 상세 정보를 조회합니다...")

    rows = fetch_app_details(list(app_ids.keys()), args.lang, args.country, args.sleep)

    with open(args.output, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"완료: {len(rows)}개 앱을 '{args.output}'에 저장했습니다.")
    print("라벨(category_id, function_type 등)은 비어 있습니다. 수동 라벨링 후 "
          "pilot_all_labeled_completed.csv와 병합해서 사용하세요.")


if __name__ == "__main__":
    main()
