# main.py
from utils.data_loader import load_data
from utils.config import CONFIG
from training.data_preprocessing import load_and_preprocess_data, create_sequences
from training.model_training import create_lstm_autoencoder, train_model
from training.model_evaluation import evaluate_model
import numpy as np

def main():
    # Load and preprocess data
    data_1200 = load_data(CONFIG['data_dir'], 1200)
    data_600 = load_data(CONFIG['data_dir'], 600)

    for feature in CONFIG['features']:
        print(f"Processing feature: {feature}")
        
        # Process data for RPM 1200
        normalized_data_1200, scaler_1200 = load_and_preprocess_data(data_1200, feature)
        sequences_1200 = create_sequences(normalized_data_1200, CONFIG['sequence_length'])

        # Process data for RPM 600
        normalized_data_600, scaler_600 = load_and_preprocess_data(data_600, feature)
        sequences_600 = create_sequences(normalized_data_600, CONFIG['sequence_length'])

        # Combine data
        all_sequences = np.concatenate([sequences_1200, sequences_600], axis=0)

        # Split into train and test
        train_size = int(len(all_sequences) * CONFIG['train_test_split'])
        train_data = all_sequences[:train_size]
        test_data = all_sequences[train_size:]

        # Create and train model
        model = create_lstm_autoencoder(CONFIG['sequence_length'], 1)
        history = train_model(model, train_data)

        # Evaluate model
        # For this example, we're using the test data as normal data
        # In practice, you'd need to create anomalous data for testing
        test_labels = np.zeros(len(test_data))  # All test data is labeled as normal
        results = evaluate_model(model, test_data, test_labels, CONFIG['anomaly_threshold'])

        print(f"Results for {feature}:")
        print(f"Precision: {results['precision']}")
        print(f"Recall: {results['recall']}")
        print(f"F1 Score: {results['f1_score']}")

        # Save model
        model.save(f"{CONFIG['models_dir']}/{feature}_model.h5")

if __name__ == "__main__":
    main()