import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from src.data.dataset import load_data
from src.config import DATASET_DIR, CSV_PATH

def evaluate_model(model_path):
    x_test, y_test = load_data(DATASET_DIR, CSV_PATH)
    model = load_model(model_path)

    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)

    print("Classification Report:")
    print(classification_report(y_test, predicted_classes, target_names=['No Damage', 'Damage']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predicted_classes))
