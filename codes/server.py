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

# 서버 정보
SERVER_IP = '165.246.43.59'  # 서버 IP 주소
SERVER_PORT = 3000      # 서버 포트 번호
RESULT_IP = '165.246.43.59'
RESULT_PORT = 6000

# 수신할 구조체 형식 정의 (C struct와 동일한 형식)
STRUCT_FORMAT = 'iiQ6f'  # fac_id, dev_id, timestamp, 6개의 float: x_accel, y_accel, z_accel, rpm, temp, voltage
STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)  # 구조체 크기 계산


# def receive_motor_data():
#     try:
#         # TCP 소켓 생성
#         client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         print("Socket created successfully.")

#         # 서버 연결
#         client_socket.connect((SERVER_IP, SERVER_PORT))
#         print(f"Connected to server at {SERVER_IP}:{SERVER_PORT}")

#         while True:
#             # 서버로부터 데이터 수신
#             data = client_socket.recv(STRUCT_SIZE)
#             if not data:
#                 print("Connection closed by server.")
#                 break

#             # 데이터가 구조체 크기보다 작은 경우 무시
#             if len(data) != STRUCT_SIZE:
#                 print(f"Incomplete data received: {len(data)} bytes.")
#                 continue

#             # 구조체 데이터를 파싱
#             motor_data = struct.unpack(STRUCT_FORMAT, data)
#             f_id, d_id, timestamp, acc_x, acc_y, acc_z, voltage, rpm, temperature = motor_data
#             print(f"Received Data - Fac_id: {f_id}, D_id: {d_id}, Time: {timestamp}, X: {acc_x:.2f}, Y: {acc_y:.2f}, Z: {acc_z:.2f}, RPM: {rpm:.2f}, Temp: {temperature:.2f}")

#     except Exception as e:
#         print(f"An error occurred: {e}")

#     finally:
#         # 소켓 닫기
#         client_socket.close()
#         print("Socket closed.")

class RealTimeAnomalyDetector:
    def __init__(self, model_path, scaler_path, rpm_type='rpm_600', sequence_length=30):
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
        # 모델 구조 초기화
        self.model = LSTMAutoencoder(
            input_dim=3,          # X, Y, Z 축 데이터
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
        self.config = CONFIG[rpm_type]
        self.sequence_length = sequence_length
        self.threshold = self.config['anomaly_threshold']
        
        # 데이터 버퍼 초기화 (3축 가속도)
        self.data_buffer = deque(maxlen=sequence_length)
        
        # 로거 설정
        self.logger = setup_logger(
            'anomaly_detector',
            f'../logs/{rpm_type}/real_time_detection.log'
        )
        
        # 소켓 초기화
        self.sensor_socket = None  # 센서 데이터 수신용 소켓
        self.result_socket = None  # 결과 전송용 소켓
        
    def connect_to_servers(self, sensor_host, sensor_port, result_host, result_port):
        """센서 서버와 결과 서버에 연결"""
        # 센서 데이터 서버 연결
        self.sensor_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sensor_socket.connect((sensor_host, sensor_port))
            self.logger.info(f"Connected to sensor server at {sensor_host}:{sensor_port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to sensor server: {str(e)}")
            raise
        
        # 결과 전송 서버 연결
        self.result_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.result_socket.connect((result_host, result_port))
            self.logger.info(f"Connected to result server at {result_host}:{result_port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to result server: {str(e)}")
            raise
    
    def receive_sensor_data(self):
        """
        구조체 형식의 센서 데이터 수신
        
        Expected struct format:
        struct SensorData {
            double timestamp;
            double accel_x;
            double accel_y;
            double accel_z;
            double voltage;
            double temperature;
            double rpm;
        }
        """
        struct_format = STRUCT_FORMAT  # 7개의 double 값
        struct_size = STRUCT_SIZE
        
        data = self.sensor_socket.recv(struct_size)
        if len(data) != struct_size:
            return None
            
        unpacked_data = struct.unpack(struct_format, data)
        return {
            'timestamp': unpacked_data[0],
            'accel_x': unpacked_data[1],
            'accel_y': unpacked_data[2],
            'accel_z': unpacked_data[3],
            'voltage': unpacked_data[4],
            'temperature': unpacked_data[5],
            'rpm': unpacked_data[6]
        }
    
    def process_data(self, sensor_data):
        """수신한 데이터 처리 및 이상치 탐지"""
        # 3축 가속도 데이터 추출 및 버퍼에 추가
        accel_data = [
            sensor_data['accel_x'],
            sensor_data['accel_y'],
            sensor_data['accel_z']
        ]
        self.data_buffer.append(accel_data)
        
        # 버퍼가 가득 찼을 때만 이상치 탐지 수행
        if len(self.data_buffer) == self.sequence_length:
            # 버퍼 데이터를 numpy 배열로 변환
            sequence = np.array(self.data_buffer)
            
            # 데이터 정규화
            sequence_normalized = self.scaler.transform(sequence)
            
            # PyTorch 텐서로 변환하고 배치 차원 추가
            sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)
            
            # 시퀀스 복원 (PyTorch 모델 사용)
            with torch.no_grad():
                reconstructed_sequence = self.model(sequence_tensor)
            
            # 재구성 오차 계산
            mse = torch.mean((sequence_tensor - reconstructed_sequence) ** 2).item()
            
            # 이상치 판단
            is_anomaly = mse > self.threshold
            
            return {
                'timestamp': sensor_data['timestamp'],
                'is_anomaly': bool(is_anomaly),
                'reconstruction_error': float(mse),
                'threshold': float(self.threshold)
            }
        
        return None
    
    def send_result(self, result):
        """결과를 결과 서버로 전송"""
        if result:
            try:
                # 결과를 JSON 형식으로 변환하여 전송
                result_json = json.dumps(result)
                self.result_socket.send(result_json.encode() + b'\n')
                
                # 이상치 발견 시 로그 기록
                if result['is_anomaly']:
                    self.logger.warning(
                        f"Anomaly detected at {datetime.fromtimestamp(result['timestamp'])} "
                        f"with reconstruction error: {result['reconstruction_error']:.6f}"
                    )
            except Exception as e:
                self.logger.error(f"Failed to send result: {str(e)}")
    
    def run(self):
        """메인 실행 루프"""
        print("Starting anomaly detection...")
        
        try:
            while True:
                # 센서 데이터 수신
                sensor_data = self.receive_sensor_data()
                if not sensor_data:
                    print('no data')
                    break
                
                # 데이터 처리 및 이상치 탐지
                result = self.process_data(sensor_data)
                
                # 결과 전송
                if result:
                    self.send_result(result)
                    
        except Exception as e:
            self.logger.error(f"Error in main loop: {str(e)}")
    
    def cleanup(self):
        """연결 종료 및 리소스 정리"""
        if self.sensor_socket:
            self.sensor_socket.close()
        if self.result_socket:
            self.result_socket.close()
        self.logger.info("Connections closed")

def main():
    # 모델 및 스케일러 경로 설정
    rpm_type = 'rpm_600'  # 또는 'rpm_1200'
    model_path = f"../models/autoencoder.pth"
    scaler_path = f"../models/scaler.joblib"
    
    # 서버 주소 설정
    SENSOR_SERVER = (SERVER_IP, SERVER_PORT)  # 센서 데이터 서버 주소
    RESULT_SERVER = (RESULT_IP, RESULT_PORT)  # 결과 전송 서버 주소
    
    # 탐지기 인스턴스 생성
    detector = RealTimeAnomalyDetector(
        model_path=model_path,
        scaler_path=scaler_path,
        rpm_type=rpm_type
    )
    
    try:
        # 서버들에 연결
        detector.connect_to_servers(
            SENSOR_SERVER[0], SENSOR_SERVER[1],
            RESULT_SERVER[0], RESULT_SERVER[1]
        )
        
        # 실행
        detector.run()
        
    except KeyboardInterrupt:
        print("\nShutting down detector...")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
    finally:
        detector.cleanup()

if __name__ == "__main__":
    main()