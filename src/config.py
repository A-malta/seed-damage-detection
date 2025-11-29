import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
CSV_PATH = os.path.join(BASE_DIR, 'rotulo.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'damage_classification_model.h5')

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 2
EPOCHS = 100
RANDOM_STATE = 42
N_SPLITS = 5
