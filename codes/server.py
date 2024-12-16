import socket
import struct
import numpy as np
import torch 
import joblib
from collections import deque
import json
from datetime import datetime
import logging
from utils.logger import setup_logger
from utils.config import CONFIG
from lstmae import LSTMAutoencoder
import threading
from threading import Thread, Event
import time

# 서버 정보
SERVER_IP = '165.246.43.216'  # 서버 IP 주소
SERVER_PORT = 13000      # 서버 포트 번호
RESULT_IP = '165.246.43.59'
RESULT_PORT = 6000

# 수신할 구조체 형식 정의 (C struct와 동일한 형식)
STRUCT_FORMAT = 'iiQ6f'  # fac_id, dev_id, timestamp, 6개의 float: x_accel, y_accel, z_accel, rpm, temp, voltage
STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)  # 구조체 크기 계산

class MultiModelAnomalyDetector:
    def __init__(self, sensor_host, sensor_port, result_host, result_port, features=None):
        """
        Args:
            features: 사용할 feature 리스트. None이면 모든 feature 사용
                     example: ['vibration', 'voltage']
        """
        self.features = features if features else ['vibration', 'voltage', 'temperature']
        self.detectors = {}
        
        # 선택된 feature에 대해서만 detector 생성
        if 'vibration' in self.features:
            self.detectors['vibration'] = RealTimeAnomalyDetector(
                model_path='../models/vib_5.pth',
                scaler_path='../models/scaler_vib_5.joblib',
                rpm_type='rpm_600',
                sequence_length=5,
                feature_type='vibration'
            )
            
        if 'voltage' in self.features:
            self.detectors['voltage'] = RealTimeAnomalyDetector(
                model_path='../models/volt_5.pth',
                scaler_path='../models/scaler_volt_5.joblib',
                rpm_type='rpm_600',
                sequence_length=5,
                feature_type='voltage'
            )
            
        if 'temperature' in self.features:
            self.detectors['temperature'] = RealTimeAnomalyDetector(
                model_path='models/temp_5.pth',
                scaler_path='models/scaler_temp_5.joblib',
                rpm_type='rpm_600',
                sequence_length=5,
                feature_type='temperature'
            )
        
        # 결과 동기화를 위한 이벤트
        self.results_ready = threading.Event()
        self.results = {}
        
        # 서버 정보 저장
        self.sensor_host = sensor_host
        self.sensor_port = sensor_port
        self.result_host = result_host
        self.result_port = result_port
        
        # 소켓 초기화
        self.sensor_socket = None
        self.result_socket = None
    
    def process_data(self, data):
        """각 모델의 이상치 탐지를 별도 스레드로 실행"""
        threads = []
        
        def run_detector(detector, data, feature_type):
            result = detector.process_data(data)
            if result:
                self.results[feature_type] = result
            
            # 모든 결과가 준비되었는지 확인
            if len(self.results) == len(self.features):  # 선택된 모든 모델의 결과가 있으면
                print("All results ready")  # 추가
                self.results_ready.set()
        
        # 선택된 feature에 대해서만 스레드 생성
        for feature_type, detector in self.detectors.items():
            threads.append(threading.Thread(
                target=run_detector,
                args=(detector, data, feature_type)
            ))
        
        # 스레드 시작
        for thread in threads:
            thread.start()
        
        # 모든 결과가 준비될 때까지 대기 (타임아웃 설정)
        if self.results_ready.wait(timeout=0.8):  # 0.8초 타임아웃
            # 결과 전송
            combined_result = {
                'timestamp': data['timestamp'],
                'anomalies': self.results
            }
            print("Sending combined result:", combined_result)  # 추가
            self.send_result(combined_result)
        
        # 결과 초기화
        self.results = {}
        self.results_ready.clear()
        
    def connect_to_servers(self):
       """서버 연결"""
       try:
           # 센서 데이터 서버 연결
           self.sensor_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
           self.sensor_socket.connect((self.sensor_host, self.sensor_port))
           print(f"Connected to sensor server at {self.sensor_host}:{self.sensor_port}")
           
           # 결과 전송 서버 연결
           self.result_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
           self.result_socket.connect((self.result_host, self.result_port))
           print(f"Connected to result server at {self.result_host}:{self.result_port}")
           
       except Exception as e:
           print(f"Failed to connect to servers: {str(e)}")
           raise
    
    def receive_sensor_data(self):
        """구조체 형식의 센서 데이터 수신"""
        data = self.sensor_socket.recv(STRUCT_SIZE)
        print(f"Received data size: {len(data)}")  # 추가
        if len(data) != STRUCT_SIZE:
            return None
            
        # 구조체 언패킹
        unpacked_data = struct.unpack(STRUCT_FORMAT, data)
        print("Unpacked data:", unpacked_data)  # 추가
        return {
            'fac_id': unpacked_data[0],
            'dev_id': unpacked_data[1],
            'timestamp': unpacked_data[2],
            'accel_x': unpacked_data[3],
            'accel_y': unpacked_data[4],
            'accel_z': unpacked_data[5],
            'voltage': unpacked_data[6],
            'rpm': unpacked_data[7],
            'temperature': unpacked_data[8]
        }
    
    def send_result(self, result):
        """통합된 결과 전송"""
        try:
            result_json = json.dumps(result)
            self.result_socket.send(result_json.encode() + b'\n')
        except Exception as e:
            print(f"Failed to send result: {str(e)}")
    
    def run(self):
        """실시간 이상치 탐지 실행"""
        print("Starting multi-model anomaly detection...")
        
        try:
            while True:
                # 센서 데이터 수신
                sensor_data = self.receive_sensor_data()
                if not sensor_data:
                    print("No data received or invalid data, exiting...")  # 추가
                    break
                
                # 다중 모델 처리
                self.process_data(sensor_data)
                
                # 1초 동기화
                elapsed = time.time() - sensor_data['timestamp']
                if elapsed < 1.0:
                    time.sleep(1.0 - elapsed)
                
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            print(f"Error details:", e.__class__.__name__)  # 추가
    
    def cleanup(self):
        """연결 종료 및 리소스 정리"""
        if self.sensor_socket:
            self.sensor_socket.close()
        if self.result_socket:
            self.result_socket.close()
        print("Connections closed")

class DataBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.last_processed_time = None
        
    def add(self, data):
        self.buffer.append(data)
        
    def is_ready(self):
        return len(self.buffer) == self.buffer.maxlen

# 각 모델 클래스
class RealTimeAnomalyDetector:
    def __init__(self, model_path, scaler_path, rpm_type='rpm_600', feature_type='vibration', sequence_length=30):
        """
        실시간 이상치 탐지 클라이언트 초기화
        
        Args:
            model_path: 학습된 모델 파일 경로
            scaler_path: 저장된 스케일러 파일 경로
            rpm_type: RPM 설정 ('rpm_600' 또는 'rpm_1200')
            sequence_length: 시퀀스 길이
        """
        # 모델과 스케일러 로드
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_type = feature_type
        
        # 입력 차원 설정
        input_dim = 3 if feature_type == 'vibration' else 1
        
        # 모델 구조 초기화
        self.model = LSTMAutoencoder(
            input_dim=input_dim,          # X, Y, Z 축 데이터
            hidden_dim=64,        # 히든 레이어 차원
            seq_length=30,        # 시퀀스 길이
            n_layers=2,           # LSTM 레이어 수
            dropout=0.2
        ).to(self.device)
    
        # 저장된 가중치 로드
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # 평가 모드로 설정
        self.scaler = joblib.load(scaler_path)
        
        # 설정 로드
        self.config = CONFIG['models'][feature_type]
        self.sequence_length = self.config['lookback']
        self.threshold = self.config['threshold']
        
        # 데이터 버퍼 초기화 (3축 가속도)
        self.data_buffer = deque(maxlen=sequence_length)
        
        # 로거 설정
        self.logger = setup_logger(
            f'{feature_type}_anomaly_detector',
            f'../logs/{rpm_type}/{feature_type}_detection.log'
        )
    
    def process_data(self, sensor_data):
        """수신한 데이터 처리 및 이상치 탐지"""
        # 데이터 타입에 따른 특징 추출
        if self.feature_type == 'vibration':
            features = [
                sensor_data['accel_x'],
                sensor_data['accel_y'],
                sensor_data['accel_z']
            ]
        elif self.feature_type == 'voltage':
            features = [sensor_data['voltage']]
        elif self.feature_type == 'temperature':
            features = [sensor_data['temperature']]
            
        self.data_buffer.append(features)
        
        # 버퍼가 가득 찼을 때만 이상치 탐지 수행
        if len(self.data_buffer) == self.sequence_length:
            sequence = np.array(self.data_buffer)
            
            # 데이터 정규화
            sequence_normalized = self.scaler.transform(sequence)
            
            # PyTorch 텐서로 변환하고 배치 차원 추가
            sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)
            
            # 시퀀스 복원
            with torch.no_grad():
                reconstructed_sequence = self.model(sequence_tensor)
            
            # 재구성 오차 계산
            mse = torch.mean((sequence_tensor - reconstructed_sequence) ** 2).item()
            
            # 이상치 판단
            is_anomaly = mse > self.threshold
            
            return {
                'timestamp': sensor_data['timestamp'],
                'feature_type': self.feature_type,
                'is_anomaly': bool(is_anomaly),
                'reconstruction_error': float(mse),
                'threshold': float(self.threshold)
            }
        
        return None

def main():
    
    # 사용할 feature 선택
    selected_features = ['vibration']  # 진동 데이터만 사용
    
    try:
        # 다중 모델 탐지기 초기화
        detector = MultiModelAnomalyDetector(
            sensor_host=SERVER_IP,
            sensor_port=SERVER_PORT,
            result_host=RESULT_IP,
            result_port=RESULT_PORT,
            features=selected_features  # 선택한 feature 전달
        )
        
        print(f"Initializing anomaly detector for features: {selected_features}")
        
        # 서버 연결
        print("Connecting to servers...")
        detector.connect_to_servers()
        
        # 실행
        print("Starting detection...")
        detector.run()
        
    except KeyboardInterrupt:
        print("\nShutting down detector...")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        print("\nCleaning up...")
        detector.cleanup()

if __name__ == "__main__":
    main()
