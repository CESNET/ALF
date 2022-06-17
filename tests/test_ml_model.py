import pytest

from sklearn.ensemble import RandomForestClassifier
from numpy.random import seed
from sklearn.exceptions import NotFittedError

from alf import ml_model
from alf import context_manager
from alf import d_manager

SupervisedMLModel = ml_model.SupervisedMLModel
ContextProvider = context_manager.ContextProvider
DbProvider = d_manager.DbProvider

seed(13)

d_0_path = "tests/test_files/test.csv"
wd = "/tmp/alf"

features = [
    'bytes_rev',
    'bytes'
]


def test_pickle():
    """Test pickleing ability. Model object should be pickleable. Afer
    pickleing, the model should be able to be loaded from file even if the
    original model is deleted (or lost in case of error of the system).
    """
    ContextProvider.create_context("file")
    ContextProvider.get_context().set_experiment_id("id666")
    ContextProvider.get_context().set_working_dir(wd)
    ContextProvider.get_context().set_features(features)

    DbProvider.create_context("file", d_0_path=d_0_path)
    DbProvider.get_context().fetch(test_size=0.5)

    clf = RandomForestClassifier(n_estimators=10)
    m = SupervisedMLModel(clf)
    m.train()
    del m
    m = SupervisedMLModel(clf)
    X, _ = DbProvider.get_context().get_train_set()  # no training need
    assert m.predict(X).shape == (3, 2)


def test_pickle_no_model():
    """Try to predict not fitted model.
    """
    ContextProvider.create_context("file")
    ContextProvider.get_context().set_experiment_id("id667")
    ContextProvider.get_context().set_working_dir(wd)
    ContextProvider.get_context().set_features(features)

    DbProvider.create_context("file", d_0_path=d_0_path)
    DbProvider.get_context().fetch(test_size=0.5)

    clf = RandomForestClassifier(n_estimators=10)
    m = SupervisedMLModel(clf)
    X, _ = DbProvider.get_context().get_train_set()
    with pytest.raises(NotFittedError):
        m.predict(X)
