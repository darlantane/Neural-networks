from model import initialize_model

def test_model_structure():
    model = initialize_model()
    assert len(model.layers) == 7

def test_model_output_shape():
    model = initialize_model()
    assert model.output_shape == (None, 10)
