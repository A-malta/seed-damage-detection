from sklearn.model_selection import KFold
from src.data.dataset import load_data, get_data_generator
from src.models.model import create_model
from src.config import DATASET_DIR, CSV_PATH, MODEL_PATH, EPOCHS, BATCH_SIZE, N_SPLITS, RANDOM_STATE

def train_single_fold(model, train_generator, x_val, y_val, fold):
    print(f'Training on fold {fold}...')
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=(x_val, y_val)
    )
    print(f'Fold {fold} completed.')

def run_cross_validation(images, labels, data_gen, kf):
    fold = 1
    for train_index, val_index in kf.split(images):
        x_train, x_val = images[train_index], images[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        model = create_model()
        train_generator = data_gen.flow(x_train, y_train, batch_size=BATCH_SIZE)

        train_single_fold(model, train_generator, x_val, y_val, fold)
        fold += 1
    return model

def train_model():
    images, labels = load_data(DATASET_DIR, CSV_PATH)
    data_gen = get_data_generator()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    model = run_cross_validation(images, labels, data_gen, kf)

    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
