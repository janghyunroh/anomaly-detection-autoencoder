import pandas as pd
import os
import argparse
from utils.config import CONFIG, REQUIRED_FEATURES

def check_outliers(df, feature_name, lower_bound, upper_bound):
    """
    특정 컬럼의 값이 주어진 범위를 벗어나는 행을 찾아 출력
    
    Parameters:
        df: DataFrame
        feature_name: 검사할 컬럼명
        lower_bound: 하한값
        upper_bound: 상한값
    """
    outliers = df[(df[feature_name] < lower_bound) | (df[feature_name] > upper_bound)]
    if not outliers.empty:
        print(f"\n{feature_name} 이상치 발견:")
        print(f"총 {len(outliers)}개의 이상치")
        print("-" * 50)
        for idx, row in outliers.iterrows():
            print(f"인덱스: {idx}")
            print(f"값: {row[feature_name]}")
            print(f"타임스탬프: {row['timestamp']}")
            print("-" * 30)

def main():
    parser = argparse.ArgumentParser(description='CSV 파일의 이상치 검사')
    parser.add_argument('file_path', help='검사할 CSV 파일 경로')
    parser.add_argument('--feature', help='검사할 특정 feature (미지정시 모든 feature 검사)')
    parser.add_argument('--lower', type=float, help='하한값')
    parser.add_argument('--upper', type=float, help='상한값')
    args = parser.parse_args()

    # CSV 파일 로드
    try:
        df = pd.read_csv(args.file_path)
        print(f"파일 로드 완료: {args.file_path}")
        print(f"총 {len(df)}개의 행")
    except Exception as e:
        print(f"파일 로드 실패: {str(e)}")
        return

    # 검사할 features 결정
    if args.feature:
        features = [args.feature]
    else:
        features = CONFIG['features']  # config.py에 정의된 모든 features

    # 각 feature별로 이상치 검사
    for feature in features:
        if feature not in df.columns:
            print(f"\n경고: {feature} 컬럼이 데이터에 없습니다.")
            continue

        # feature별 기본 범위 설정
        if args.lower is not None and args.upper is not None:
            lower_bound = args.lower
            upper_bound = args.upper
        else:
            # feature별 기본 범위 설정
            if 'accel' in feature:
                lower_bound = 0.3  # g
                upper_bound = 0.7   # g
            elif feature == 'temperature':
                lower_bound = 0    # °C
                upper_bound = 100  # °C
            elif feature == 'voltage':
                lower_bound = 0    # V
                upper_bound = 250  # V
            elif feature == 'rpm':
                lower_bound = 0    # RPM
                upper_bound = 2000 # RPM
            else:
                print(f"\n경고: {feature}의 기본 범위가 설정되지 않았습니다.")
                continue

        check_outliers(df, feature, lower_bound, upper_bound)

if __name__ == "__main__":
    main()