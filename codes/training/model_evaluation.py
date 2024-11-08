# model_evaluation.py
import numpy as np
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, fbeta_score

def evaluate_model(model, test_data, test_labels, threshold):
    predictions = model.predict(test_data)
    mse = np.mean(np.power(test_data - predictions, 2), axis=1)
    
    # Classify as anomaly if MSE is above threshold
    anomalies = (mse > threshold).astype(int)
    
    # Calculate metrics
    precision = precision_score(test_labels, anomalies)
    recall = recall_score(test_labels, anomalies)
    f1 = f1_score(test_labels, anomalies)
    f2 = fbeta_score(test_labels, anomalies, beta=2)
    
    return {
        'mse': mse,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'f2_score' : f2
    }