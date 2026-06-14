import os
import argparse
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

def crawl_images(keyword, max_num, output_dir, engine='google'):
    """
    지정된 키워드로 이미지를 크롤링하여 output_dir에 저장합니다.
    """
    # 저장할 디렉토리 생성 (키워드별로 폴더 생성)
    safe_keyword = keyword.replace(' ', '_').replace('/', '_')
    keyword_dir = os.path.join(output_dir, safe_keyword)
    os.makedirs(keyword_dir, exist_ok=True)
    
    print(f"\n[{engine.upper()}] '{keyword}' 검색 및 다운로드 시작 (목표: {max_num}장) ...")
    
    if engine.lower() == 'google':
        crawler = GoogleImageCrawler(storage={'root_dir': keyword_dir})
    else:
        crawler = BingImageCrawler(storage={'root_dir': keyword_dir})
        
    # 크롤러 실행
    crawler.crawl(keyword=keyword, max_num=max_num)
    print(f"다운로드 완료: {keyword_dir}")

def main():
    # 우리의 진짜 타겟: 숏폼 및 고보정 사진 (Hard Negative)
    keywords = [
        "틱톡 뷰티 필터 셀카 캡처",
        "인스타그램 릴스 보정 필터",
        "유튜브 쇼츠 얼굴 보정 캡처",
        "스튜디오 증명사진 포토샵 고화질",
        "바디프로필 스튜디오 얼굴 근접"
    ]
    
    # 키워드당 80장씩 총 400장
    MAX_NUM_PER_KEYWORD = 80 
    
    # 서버 환경 고려하여 경로 설정
    OUTPUT_BASE_DIR = "crawled_sns_images"
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    print("=== [숏폼 & 스튜디오 도메인] Hard Negative 크롤링 시작 ===")
    
    for kw in keywords:
        # Google은 차단이 심하므로 Bing 엔진을 사용합니다.
        crawl_images(keyword=kw, max_num=MAX_NUM_PER_KEYWORD, output_dir=OUTPUT_BASE_DIR, engine='bing')
        
    print("\n=== 모든 크롤링 작업이 완료되었습니다! ===")
    print(f"저장 경로: {os.path.abspath(OUTPUT_BASE_DIR)}")

if __name__ == '__main__':
    main()
