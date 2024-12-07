import pandas as pd
import numpy as np
import os

# 전처리 모듈 1 - 다운 샘플링
class TimeSeriesDownSampler:
    def __init__(self, sampling_rate, method="mean"):
        """
        Time Series 데이터 다운 샘플링 클래스
        Args:
            sampling_rate (int): 샘플링 간격 (밀리초 기준).
            method (str): 'mean', 'max', 'min' 중 선택.
        """
        if method not in ["mean", "max", "min"]:
            raise ValueError("Method must be 'mean', 'max', or 'min'")
        self.sampling_rate = sampling_rate
        self.method = method

    def down_sample(self, data):
        """
        주어진 시계열 데이터를 다운 샘플링.
        Args:
            data (pd.DataFrame): 타임스탬프가 포함된 시계열 데이터.
                예: ["timestamp", "temperature", "x_acc", "y_acc", "z_acc", "voltage", "rpm"]
        Returns:
            pd.DataFrame: 다운 샘플링된 데이터.
        """
        if "timestamp" not in data.columns:
            raise ValueError("Data must include a 'timestamp' column")

        # 타임스탬프를 datetime 형식으로 변환
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        # 타임스탬프를 기준으로 리샘플링
        data.set_index("timestamp", inplace=True)
        if self.method == "mean":
            downsampled = data.resample(f"{self.sampling_rate}ms").mean()
        elif self.method == "max":
            downsampled = data.resample(f"{self.sampling_rate}ms").max()
        elif self.method == "min":
            downsampled = data.resample(f"{self.sampling_rate}ms").min()

        # NaN 값 제거 (필요에 따라 처리 방식을 변경 가능)
        downsampled = downsampled.dropna().reset_index()
        return downsampled
    
    def down_sample_numeric_timestamp(self, data):
        """
        ms 단위의 정수형 timestamp 데이터를 다운 샘플링.
        Args:
            data (pd.DataFrame): 정수형 timestamp 데이터를 포함한 데이터프레임.
        Returns:
            pd.DataFrame: 다운 샘플링된 데이터.
        """
        if "timestamp" not in data.columns:
            raise ValueError("Data must include a 'timestamp' column")

        # 그룹화 키 생성: timestamp를 샘플링 간격으로 나눠서 그룹화
        data["group"] = (data["timestamp"] // self.sampling_rate)

        # 그룹화 및 다운 샘플링
        if self.method == "mean":
            downsampled = data.groupby("group").mean()
        elif self.method == "max":
            downsampled = data.groupby("group").max()
        elif self.method == "min":
            downsampled = data.groupby("group").min()

        # timestamp를 각 그룹의 첫 번째 timestamp로 복원
        downsampled["timestamp"] = data.groupby("group")["timestamp"].first().values

        downsampled = downsampled.reset_index(drop=True)

        return downsampled
    
    
    # 디렉토리에서 파일 읽고 다운 샘플링 결과 저장
    def process_directory(input_dir, output_dir, down_sampler):
        """
        특정 디렉토리의 CSV 파일을 다운 샘플링하여 저장.
        Args:
            input_dir (str): 입력 디렉토리 경로.
            output_dir (str): 출력 디렉토리 경로.
            down_sampler (TimeSeriesDownSampler): 다운 샘플링 클래스 객체.
        """
        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 입력 디렉토리 내의 모든 파일 처리
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".csv"):
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, file_name)

                # CSV 파일 읽기
                data = pd.read_csv(input_path)
                print(f"Processing file: {file_name}")

                # 다운 샘플링 적용
                downsampled_data = down_sampler.down_sample(data)

                # 결과를 새로운 CSV로 저장
                downsampled_data.to_csv(output_path, index=False)
                print(f"Saved downsampled file: {output_path}")

    # 디렉토리에서 파일 읽고 다운 샘플링 결과를 DataFrame 리스트로 반환
    def process_directory_to_dfs(self, input_dir, numeric_timestamp=False):
        """
        특정 디렉토리의 CSV 파일을 다운 샘플링하여 DataFrame 리스트로 반환.
        Args:
            input_dir (str): 입력 디렉토리 경로.
            down_sampler (TimeSeriesDownSampler): 다운 샘플링 클래스 객체.
        Returns:
            list[pd.DataFrame]: 다운 샘플링된 데이터프레임들의 리스트.
        """
        df_list = []
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".csv"):
                input_path = os.path.join(input_dir, file_name)

                # CSV 파일 읽기
                data = pd.read_csv(input_path)
                print(f"Processing file: {file_name}")

                # 다운 샘플링 적용
                if numeric_timestamp:
                    downsampled_data = self.down_sample_numeric_timestamp(data)
                else:
                    downsampled_data = self.down_sample(data)

                # 리스트에 추가
                df_list.append(downsampled_data)
        
        return df_list