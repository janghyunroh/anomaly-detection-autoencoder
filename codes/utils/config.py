# utils/config.py
CONFIG = {
    'rpm_600' : {
        'raw_data_dir' : '../datas/raw/rpm_600',
        'raw_data_train_dir': '../datas/raw/rpm_600/train',
        'raw_data_test_dir' : '../datas/raw/rpm_600/test',
        'processed_data_dir' : '../datas/processed/rpm_600',
        'processed_train_data_dir' : '../datas/processed/rpm_600/train',
        'processed_test_data_dir' : '../datas/processed/rpm_600/test',
        'anomalous_data_dir' : '../datas/anomalous/rpm_600',
        'test_data_dir' : '../datas/anomalous/rpm_600',
        'train_data_dir': '../datas/processed/rpm_600/train',
        'models_dir': '../models/rpm_600',
        'sequence_length': 100,
        'train_test_split': 0.9,
        'anomaly_threshold': 0.1,
        'results_dir' : '../results/rpm_600',
        'log_dir' : '../logs/rpm_600'
    },
    
    'rpm_1200' : {
        'raw_data_dir' : '../datas/raw/rpm_1200',
        'raw_data_train_dir': '../datas/raw/rpm_1200/train',
        'raw_data_test_dir' : '../datas/raw/rpm_1200/test',
        'processed_data_dir' : '../datas/processed/rpm_1200',
        'processed_train_data_dir' : '../datas/processed/rpm_1200/train',
        'processed_test_data_dir' : '../datas/processed/rpm_1200/test',
        'anomalous_data_dir' : '../datas/anomalous/rpm_1200',
        'test_data_dir' : '../datas/anomalous/rpm_1200',
        'train_data_dir': '../datas/processed/rpm_1200/train',
        'models_dir': '../models/rpm_1200',
        'sequence_length': 250,
        'train_test_split': 0.9,
        'anomaly_threshold': 0.1,
        'results_dir' : '../results/rpm_1200',
        'log_dir' : '../logs/rpm_1200'
    },
    
    
    'features': ['accel_x','accel_y','accel_z','voltage','rpm','temperature'],

    'model_type' : ['vibration', 'voltage', 'rpm', 'temperature', 'multi-feature'],
    
    'training' : {
        'epochs' : 10,
        'batch_size' : 32,
        'validation_split' : 0.1,
    }
}

REQUIRED_FEATURES = {
    'vibration' : ['accel_x','accel_y','accel_z'],
    'voltage' : 'voltage',
    'rpm' : 'rpm',
    'temperature' : 'temperature',
    'multi-feature' : ['accel_x','accel_y','accel_z','voltage','rpm','temperature']
}

PER_MODEL = {
    'vibration' : {
        'type': 'vibration',
        'required_feature_num': 3,
        'required_features' : ['accel_x','accel_y','accel_z'],
        'layer_size' : 32,
    },
    'voltage' : {
        'type': 'voltage',
        'required_feature_num': 1,
        'required_features' : 'voltage',
        'layer_size' : 32
    },
    'rpm' : {
        'type': 'rpm',
        'required_feature_num': 1,
        'required_features' : 'rpm',
        'layer_size' : 32,
    },
    'temperature' : {
        'type': 'temperature',
        'required_feature_num': 1,
        'required_features' : 'temperature',
        'layer_size' : 32,
    }
}