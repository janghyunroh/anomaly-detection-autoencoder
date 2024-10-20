# utils/config.py
CONFIG = {
    'data_dir': 'data/raw',
    'processed_data_dir': 'data/processed',
    'models_dir': 'models',
    'sequence_length': 100,
    'train_test_split': 0.9,
    'anomaly_threshold': 0.1,
    'features': ['Voltage', 'acc_x', 'acc_y', 'acc_z', 'temperature', 'RPM']
}