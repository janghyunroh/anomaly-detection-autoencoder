# utils/config.py
CONFIG = {
    'data_dir': '../datas/raw',
    'processed_data_dir': '../datas/processed',
    'models_dir': 'models',
    'sequence_length': 100,
    'train_test_split': 0.9,
    'anomaly_threshold': 0.1,
    'features': ['accel_x','accel_y','accel_z','voltage','rpm','temperature'],
    'model_type' : ['vibration', 'voltage', 'rpm', 'temperature', 'multi-feature']
}

REQUIRED_FEATURES = {
    'vibration' : ['accel_x','accel_y','accel_z'],
    'voltage' : 'voltage',
    'rpm' : 'rpm',
    'multi-feature' : ['accel_x','accel_y','accel_z','voltage','rpm','temperature']
}