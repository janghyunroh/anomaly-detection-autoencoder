# model_loader.py
from tensorflow.keras.models import load_model

class ModelLoader:
    def __init__(self):
        self.models = {}

    def load_model(self, model_name):
        if model_name not in self.models:
            model_path = f"models/{model_name}.h5"
            self.models[model_name] = load_model(model_path)
        return self.models[model_name]