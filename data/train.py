from tensorflow.keras.datasets import mnist
from model import initialize_model

def load_and_preprocess_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[..., None] / 255.0
    X_test = X_test[..., None] / 255.0
    return X_train, y_train, X_test, y_test

def compile_model(model):
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
