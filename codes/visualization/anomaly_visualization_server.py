# 4. anomaly_visualization_server.py
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('anomaly_alert')
def handle_anomaly_alert(data):
    print(f"Anomaly detected: {data}")
    emit('anomaly_log', data, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, debug=True)