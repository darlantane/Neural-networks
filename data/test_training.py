import numpy as np
from model import initialize_model
from train import compile_model

def test_model_compiles():
    model = initialize_model()
    model = compile_model(model)
    assert model.loss is not None

def test_model_can_train_one_batch():
    model = compile_model(initialize_model())

    X_dummy = np.random.rand(8, 28, 28, 1)
    y_dummy = np.random.randint(0, 10, size=8)

    history = model.fit(X_dummy, y_dummy, epochs=1, batch_size=4, verbose=0)
    assert len(history.history["loss"]) == 1

def test_prediction_output():
    model = compile_model(initialize_model())
    X_dummy = np.random.rand(1, 28, 28, 1)

    prediction = model.predict(X_dummy, verbose=0)
    assert prediction.shape == (1, 10)
    assert abs(prediction.sum() - 1.0) < 1e-5
