# utils/config.py
CONFIG = {

    'models' : {
        'vibration' : {
            'lookback' : 5,
            'model_path' : '../models/vib_5.pth',
            'scaler_path' : '../models/scaler_vib_5.joblib',
            'threshold' : 0.5,
            # 0.1275
            # 0.8
            # 0.9
            'normal_data_path' : '../datas/normal/vib',
            'abnormal_data_path' : '../datas/abnormal/vib'
        },
        'voltage' : {
            'lookback' : 5,
            'model_path' : '../models/volt_5.pth',
            'scaler_path' : '../models/scaler_volt_5.joblib',
            'threshold' : 0.5,
            'normal_data_path' : '../datas/normal/volt',
            'abnormal_data_path' : '../datas/abnormal/volt'
        },
        'temperature' : {
            'lookback' : 5,
            'model_path' : '../models/temp_5.pth',
            'scaler_path' : '../models/scaler_temp_5.joblib',
            'threshold' : 0.1408,
            'normal_data_path' : '../datas/normal/temp',
            'abnormal_data_path' : '../datas/abnormal/temp'
        }
    },

    'training' : {
        'epochs' : 10,
        'batch_size' : 32,
        'validation_split' : 0.1,
        'learning_rate' : 1e-3,
        'patience' : 10
    },





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
        'data_period_ms': 50,
        'train_test_split': 0.9,
        'anomaly_threshold': 0.1408,
        # 0.0413
        # 0.8
        #0.9209
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
        'data_period_ms': 20,
        'train_test_split': 0.9,
        'anomaly_threshold': 0.9209,
        'results_dir' : '../results/rpm_1200',
        'log_dir' : '../logs/rpm_1200'
    },
    
    
    'features': ['accel_x','accel_y','accel_z','voltage','rpm','temperature'],

    'model_type' : ['vibration', 'voltage', 'rpm', 'temperature', 'multi-feature'],
    
    
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
        'layer_size_1' : 64, 
        'layer_size_2' : 32
    },
    'voltage' : {
        'type': 'voltage',
        'required_feature_num': 1,
        'required_features' : 'voltage',
        'layer_size_1' : 64, 
        'layer_size_2' : 32
    },
    'rpm' : {
        'type': 'rpm',
        'required_feature_num': 1,
        'required_features' : 'rpm',
        'layer_size_1' : 64, 
        'layer_size_2' : 32
    },
    'temperature' : {
        'type': 'temperature',
        'required_feature_num': 1,
        'required_features' : 'temperature',
        'layer_size_1' : 64, 
        'layer_size_2' : 32
    }
}

SERVER_CONFIG = {
    'SERVER_IP' : '165.246.43.59',  # 서버 IP 주소
    'SERVER_PORT' : 3000      # 서버 포트 번호
}