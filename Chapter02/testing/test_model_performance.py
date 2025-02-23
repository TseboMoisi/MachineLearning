import pytest
import numpy as np
from typing import Union
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report

import joblib

@pytest.fixture
def test_dataset() -> Union[np.array, np.array]:
    # Load the dataset
    X, y = load_wine(return_X_y=True)
    # create an array of True for 2 and False otherwise
    y = y == 2
    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_test, y_test


# Here we are getting a model for the test. The repository for the model is public so we don't need to store any secrets
@pytest.fixture
def model() -> sklearn.ensemble._forest.RandomForestClassifier:
    REPO_ID = "electricweegie/mlewp-sklearn-wine"
    FILENAME = "rfc.joblib"
    model = joblib.load(hf_hub_download(REPO_ID, FILENAME))
    return model

# Here we creating the test, that confirms that the predictions fo the model product the correct object types
def test_model_inference_types(model, test_dataset):
    assert isinstance(model.predict(test_dataset[0]), np.ndarray)
    assert isinstance(test_dataset[0], np.ndarray)
    assert isinstance(test_dataset[1], np.ndarray)

# This is now a test make sure specific requirements in terms of performance are met for the model, while the model is using the test dataset 
def test_model_performance(model, test_dataset):
    metrics = classification_report(y_true=test_dataset[1], y_pred=model.predict(test_dataset[0]), output_dict=True)
    assert metrics['False']['f1-score'] > 0.95
    assert metrics['False']['precision'] > 0.9
    assert metrics['True']['f1-score'] > 0.8
    assert metrics['True']['precision'] > 0.8