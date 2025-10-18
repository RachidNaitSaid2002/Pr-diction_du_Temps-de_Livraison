import pytest
import pandas as pd
from pipline import load_data, dimensions, fill_missing_values,get_max_score, Split_Data, Split_train_test, encode_with_onehot, Scaler_Standerd, Train_Model

@pytest.fixture
def my_data():
    return load_data('MyRealData.csv')

def test_dimensions(my_data):
    dims = dimensions(my_data)
    assert dims == (1000, 9)

def test_MAE(my_data):
    data_filled = fill_missing_values(my_data)
    X_train, X_test, y_train, y_test = Split_train_test(data_filled)
    X_train_encoded = encode_with_onehot(X_train)
    X_test_encoded = encode_with_onehot(X_test)
    X_train_scaled = Scaler_Standerd(X_train_encoded)
    X_test_scaled = Scaler_Standerd(X_test_encoded)
    results = Train_Model(X_train_scaled, X_test_scaled, y_train, y_test)
    best_model, mae = get_max_score(results)
    assert mae < 10.0

