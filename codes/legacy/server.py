import socket
import struct

# 서버 정보
SERVER_IP = '165.246.43.59'  # 서버 IP 주소
SERVER_PORT = 3000      # 서버 포트 번호

# 수신할 구조체 형식 정의 (C struct와 동일한 형식)
STRUCT_FORMAT = 'iiQ6f'  # fac_id, dev_id, timestamp, 6개의 float: x_accel, y_accel, z_accel, rpm, temp, voltage
STRUCT_SIZE = struct.calcsize(STRUCT_FORMAT)  # 구조체 크기 계산

def receive_motor_data():
    try:
        # TCP 소켓 생성
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket created successfully.")

        # 서버 연결
        client_socket.connect((SERVER_IP, SERVER_PORT))
        print(f"Connected to server at {SERVER_IP}:{SERVER_PORT}")

        while True:
            # 서버로부터 데이터 수신
            data = client_socket.recv(STRUCT_SIZE)
            if not data:
                print("Connection closed by server.")
                break

            # 데이터가 구조체 크기보다 작은 경우 무시
            if len(data) != STRUCT_SIZE:
                print(f"Incomplete data received: {len(data)} bytes.")
                continue

            # 구조체 데이터를 파싱
            motor_data = struct.unpack(STRUCT_FORMAT, data)
            f_id, d_id, timestamp, acc_x, acc_y, acc_z, voltage, rpm, temperature = motor_data
            print(f"Received Data - Fac_id: {f_id}, D_id: {d_id}, Time: {timestamp}, X: {acc_x:.2f}, Y: {acc_y:.2f}, Z: {acc_z:.2f}, RPM: {rpm:.2f}, Temp: {temperature:.2f}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # 소켓 닫기
        client_socket.close()
        print("Socket closed.")

# 실행
if __name__ == "__main__":
    receive_motor_data()
