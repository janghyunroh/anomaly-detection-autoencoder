import pandas as pd
import os
import argparse
from utils.config import CONFIG

def downsample_csv(input_file, output_file, sample_rate=20):
    """
    CSV 파일을 다운샘플링하여 새로운 파일로 저장
    
    Parameters:
        input_file: 입력 CSV 파일 경로
        output_file: 출력 CSV 파일 경로
        sample_rate: 샘플링 간격 (기본값: 20, 50ms -> 1s)
    """
    try:
        # CSV 파일 로드
        df = pd.read_csv(input_file)
        print(f"원본 파일 로드 완료: {input_file}")
        print(f"원본 데이터 행 수: {len(df)}")
        
        # 20번째 행마다 선택
        downsampled_df = df.iloc[::sample_rate]
        print(f"다운샘플링 후 행 수: {len(downsampled_df)}")
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 새로운 CSV 파일로 저장
        downsampled_df.to_csv(output_file, index=False)
        print(f"다운샘플링된 파일 저장 완료: {output_file}")
        
        return True
    
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        return False

def process_directory(input_dir, output_dir, sample_rate=20):
    """
    디렉토리 내의 모든 CSV 파일을 다운샘플링
    
    Parameters:
        input_dir: 입력 디렉토리 경로
        output_dir: 출력 디렉토리 경로
        sample_rate: 샘플링 간격
    """
    success_count = 0
    fail_count = 0
    
    # 입력 디렉토리의 모든 CSV 파일 처리
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                # 입력 파일의 전체 경로
                input_path = os.path.join(root, file)
                
                # 출력 파일의 상대 경로 계산
                rel_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                # 출력 파일명 생성 (downsampled_ 접두어 추가)
                output_file = os.path.join(output_path, f"downsampled_{file}")
                
                print(f"\n처리 중: {file}")
                if downsample_csv(input_path, output_file, sample_rate):
                    success_count += 1
                else:
                    fail_count += 1
    
    return success_count, fail_count

def main():
    parser = argparse.ArgumentParser(description='CSV 파일 다운샘플링 (50ms -> 1s)')
    parser.add_argument('--input', required=True, help='입력 파일 또는 디렉토리 경로')
    parser.add_argument('--output', required=True, help='출력 파일 또는 디렉토리 경로')
    parser.add_argument('--rate', type=int, default=20, help='샘플링 간격 (기본값: 20)')
    
    args = parser.parse_args()
    
    # 입력이 디렉토리인 경우
    if os.path.isdir(args.input):
        print(f"디렉토리 처리 시작: {args.input}")
        success_count, fail_count = process_directory(args.input, args.output, args.rate)
        print("\n처리 완료")
        print(f"성공: {success_count}개 파일")
        print(f"실패: {fail_count}개 파일")
    
    # 입력이 단일 파일인 경우
    else:
        if downsample_csv(args.input, args.output, args.rate):
            print("\n파일 처리 완료")
        else:
            print("\n파일 처리 실패")

if __name__ == "__main__":
    main()