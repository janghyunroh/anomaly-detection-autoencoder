import pika
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
import mysql.connector  # DB 연결을 위한 예제 (MySQL)

from pytz import timezone  # pytz 라이브러리 임포트
KST = timezone('Asia/Seoul')  # Asia/Seoul 타임존 설정

# RabbitMQ 접속 정보
RABBITMQ_HOST = '165.246.43.215'     # RabbitMQ 호스트명/아이피
RABBITMQ_QUEUE = 'raw_data_queue'    # 구독할 큐 이름
RABBITMQ_ALARM_QUEUE = 'alarm_data_queue'
RABBITMQ_USER = 'openstack'
RABBITMQ_PASS = 'cnlab1110'

# 서버 정보
SERVER_IP = '165.246.43.216'  # 서버 IP 주소
SERVER_PORT = 13000           # 서버 포트 번호
RESULT_IP = '165.246.43.59'
RESULT_PORT = 6000

# 수신할 구조체 형식 정의 (C struct와 동일한 형식)
STRUCT_FORMAT = 'iiQ6f'  # fac_id, dev_id, timestamp, x_accel, y_accel, z_accel, rpm, temp, voltage
STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)

class MultiModelAnomalyDetector:
    def __init__(self, result_host, result_port, features=None):
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
        
        # 결과 서버 정보
        self.result_host = result_host
        self.result_port = result_port
        self.result_socket = None

        # RabbitMQ 연결 관련
        self.connection = None
        self.channel = None

        # DB 연결 관련
        self.db_conn = None
        self.db_cursor = None

    def connect_to_result_server(self):
        """결과 전송 서버 연결"""
        try:
            self.result_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.result_socket.connect((self.result_host, self.result_port))
            print(f"Connected to result server at {self.result_host}:{self.result_port}")
        except Exception as e:
            print(f"Failed to connect to result server: {str(e)}")
            raise

    def connect_to_db(self):
        """DB 연결"""
        try:
            self.db_conn = mysql.connector.connect(
                host='165.246.43.215',
                port=3306,
                user='remote_admin',
                password='cnlab1110',
                database='sys'
            )
            self.db_cursor = self.db_conn.cursor()
            print("Connected to DB successfully.")
        except Exception as e:
            print(f"Failed to connect to DB: {str(e)}")
            raise

    def insert_alarm(self, sensor_type, value, created_at):
        """이상치 알람 정보를 DB에 삽입"""
        try:
            sql = "INSERT INTO alarm (sensor_type, value, create_at) VALUES (%s, %s, %s)"
            self.db_cursor.execute(sql, (sensor_type, value, created_at))
            self.db_conn.commit()
            print("Alarm inserted into DB:", sensor_type, value, created_at)
        except Exception as e:
            print(f"Failed to insert alarm: {str(e)}")

    def connect_to_rabbitmq(self):
        """RabbitMQ 연결 및 채널, 큐 선언"""
        try:
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
            self.connection = pika.BlockingConnection(pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                credentials=credentials
            ))
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
            self.channel.queue_declare(queue=RABBITMQ_ALARM_QUEUE, durable=True)
            print(f"Connected to RabbitMQ at {RABBITMQ_HOST}, subscribed to queue: {RABBITMQ_QUEUE}, and alarm queue: {RABBITMQ_ALARM_QUEUE}")
        except Exception as e:
            print(f"Failed to connect to RabbitMQ: {str(e)}")
            raise

    def send_alarm_to_queue(self, alarm_message):
        """이상치 발생 시 alarm_data_queue 에 알람 정보 전송"""
        try:
            self.channel.basic_publish(
                exchange='',
                routing_key=RABBITMQ_ALARM_QUEUE,
                body=json.dumps(alarm_message)
            )
            print("Alarm message sent to alarm_data_queue:", alarm_message)
        except Exception as e:
            print(f"Failed to send alarm to queue: {str(e)}")

    def on_message(self, ch, method, properties, body):
        """RabbitMQ로부터 메시지를 수신하는 콜백 함수"""
        if len(body) != STRUCT_SIZE:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        unpacked_data = struct.unpack(STRUCT_FORMAT, body)
        sensor_data = {
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

        self.process_data(sensor_data)
        
        ch.basic_ack(delivery_tag=method.delivery_tag)

    def process_data(self, data):
        threads = []
        
        def run_detector(detector, data, feature_type):
            result = detector.process_data(data)
            if result:
                self.results[feature_type] = result
            
            if len(self.results) == len(self.features):
                self.results_ready.set()
        
        for feature_type, detector in self.detectors.items():
            threads.append(threading.Thread(
                target=run_detector,
                args=(detector, data, feature_type)
            ))
        
        for thread in threads:
            thread.start()
        
        if self.results_ready.wait(timeout=0.8):
            combined_result = {
                'timestamp': data['timestamp'],
                'anomalies': self.results
            }
            print("Sending combined result:", combined_result)
            self.send_result(combined_result)

            # 이상치 발생한 경우 DB 및 alarm_data_queue 에 알람 정보 삽입
            for feature_type, anomaly_result in self.results.items():
                if anomaly_result['is_anomaly']:
                    # anomaly_result['timestamp']를 datetime으로 변환
                    event_time = datetime.utcfromtimestamp(anomaly_result['timestamp'])
                    
                    # 실제 센서 값 사용
                    actual_value = anomaly_result['actual_value']
                    
                    self.insert_alarm(
                        sensor_type=anomaly_result['feature_type'],
                        value=actual_value,
                        created_at=event_time
                    )
                    alarm_message = {
                        'sensorType': anomaly_result['feature_type'],
                        'value': actual_value, 
                        'createAt': event_time.strftime("%Y-%m-%d %H:%M:%S"),
                        'timestamp': anomaly_result['timestamp']
                    }
                    self.send_alarm_to_queue(alarm_message)

        self.results = {}
        self.results_ready.clear()
        
    def send_result(self, result):
        try:
            result_json = json.dumps(result)
            self.result_socket.send(result_json.encode() + b'\n')
        except Exception as e:
            print(f"Failed to send result: {str(e)}")

    def run(self):
        print("Starting multi-model anomaly detection with RabbitMQ subscription...")
        #self.connect_to_result_server()
        self.connect_to_db()
        self.connect_to_rabbitmq()

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=RABBITMQ_QUEUE, on_message_callback=self.on_message)
        
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("Interrupted by user.")
        except Exception as e:
            print(f"Error while consuming messages: {str(e)}")
        finally:
            self.cleanup()

    def cleanup(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
        if self.result_socket:
            self.result_socket.close()
        if self.db_cursor:
            self.db_cursor.close()
        if self.db_conn:
            self.db_conn.close()
        print("Connections closed")

class DataBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.last_processed_time = None
        
    def add(self, data):
        self.buffer.append(data)
        
    def is_ready(self):
        return len(self.buffer) == self.buffer.maxlen

class RealTimeAnomalyDetector:
    def __init__(self, model_path, scaler_path, rpm_type='rpm_600', feature_type='vibration', sequence_length=5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.feature_type = feature_type
        
        input_dim = 3 if feature_type == 'vibration' else 1
        
        self.model = LSTMAutoencoder(
            input_dim=input_dim,
            hidden_dim=64,
            seq_length=sequence_length,
            n_layers=2,
            dropout=0.2
        ).to(self.device)
    
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        
        self.config = CONFIG['models'][feature_type]
        self.sequence_length = sequence_length
        self.threshold = self.config['threshold']
        
        self.data_buffer = deque(maxlen=sequence_length)
        
        self.logger = setup_logger(
            f'{feature_type}_anomaly_detector',
            f'../logs/{rpm_type}/{feature_type}_detection.log'
        )
        
    def process_data(self, sensor_data):
        # 실제 센서 값 추출
        if self.feature_type == 'vibration':
            features = [
                sensor_data['accel_x'],
                sensor_data['accel_y'],
                sensor_data['accel_z']
            ]
            # 예: vibration은 accel_x 값을 actual_value로 사용
            actual_value = sensor_data['accel_x']
        elif self.feature_type == 'voltage':
            features = [sensor_data['voltage']]
            actual_value = sensor_data['voltage']
        elif self.feature_type == 'temperature':
            features = [sensor_data['temperature']]
            actual_value = sensor_data['temperature']
            
        self.data_buffer.append(features)
        
        if len(self.data_buffer) == self.sequence_length:
            sequence = np.array(self.data_buffer)
            sequence_normalized = self.scaler.transform(sequence)
            sequence_tensor = torch.FloatTensor(sequence_normalized).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                reconstructed_sequence = self.model(sequence_tensor)
            
            mse = torch.mean((sequence_tensor - reconstructed_sequence) ** 2).item()
            is_anomaly = mse > self.threshold
            
            return {
                'timestamp': sensor_data['timestamp'],
                'feature_type': self.feature_type,
                'is_anomaly': bool(is_anomaly),
                'reconstruction_error': float(mse),
                'threshold': float(self.threshold),
                'actual_value': float(actual_value)  # 실제 센서 값 반환
            }
        
        return None

def main():
    selected_features = ['vibration']  # 진동 데이터만 사용
    
    try:
        detector = MultiModelAnomalyDetector(
            result_host=RESULT_IP,
            result_port=RESULT_PORT,
            features=selected_features
        )
        
        print(f"Initializing anomaly detector for features: {selected_features}")
        print("Starting detection with RabbitMQ...")
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
