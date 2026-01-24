from train import load_and_preprocess_data

def test_data_shapes():
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    assert X_train.shape[1:] == (28, 28, 1)
    assert X_test.shape[1:] == (28, 28, 1)

def test_data_normalization():
    X_train, _, _, _ = load_and_preprocess_data()
    assert X_train.min() >= 0.0
    assert X_train.max() <= 1.0
