# import pytest

from numpy.random import seed
import numpy as np
import sklearn

from alf import ml_model
from alf import context_manager
from alf import d_manager
from alf import anotator
from alf import query_strategy

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.dummy import DummyClassifier

SupervisedMLModel = ml_model.SupervisedMLModel
Committee = ml_model.CommitteeMLModel
ContextProvider = context_manager.ContextProvider
DbProvider = d_manager.DbProvider

seed(13)

d_0_path = "tests/test_files/test.csv"
wd = "/tmp/alf"


class MockAnotator(anotator.Anotator):
    def anotate(self, flows):
        classes = np.zeros(len(flows), dtype=int)
        f = flows.copy()
        f["class"] = classes
        return f


def create_mocks(i):
    ContextProvider.create_context("file")
    ContextProvider.get_context().set_experiment_id("q"+i)
    ContextProvider.get_context().set_working_dir(wd)
    ContextProvider.get_context().set_features(["bytes_rev", "bytes"])
    DbProvider.create_context("file", d_0_path=d_0_path)
    DbProvider.get_context().fetch(test_size=0.1)
    return MockAnotator()


def create_prediction_simple():
    clf = RandomForestClassifier(n_estimators=10)
    m = SupervisedMLModel(clf)
    m.train()
    X, _ = DbProvider.get_context().get_train_set()
    return m.predict(X), X, m


def create_prediction_committee():
    clf1 = ("clf1", DummyClassifier(strategy="prior"))
    clf2 = ("clf2", RandomForestClassifier())
    clf3 = ("clf3", DecisionTreeClassifier())
    m = ml_model.CommitteeMLModel(VotingClassifier(
        [clf1, clf2, clf3],
        voting="soft"))
    m.train()
    X, _ = DbProvider.get_context().get_train_set()
    return m.predict(X), X, m


def test_random_correct_count():
    """Testing ability of random query to get the correct number of
    flows and anotate them.
    """
    anotator = create_mocks("fqef")
    prediction, X, m = create_prediction_simple()

    random_query = query_strategy.RandomQueryStrategy(
        anotator,
        max_samples=2)

    s, mask = random_query.select(prediction, X)

    assert prediction.shape == (5, 2)
    assert s.shape[0] == 5
    assert np.count_nonzero(mask) == 2



def test_uncertainty_score():
    anotator = create_mocks("hgfy")
    prediction, X, m = create_prediction_simple()

    uncert = query_strategy.UncertanityRankedBatch(anotator)

    np.testing.assert_almost_equal(
        list(uncert._score(prediction)),
        [0.3, 0.1, 0.2, 0.1, 0.4],
        decimal=2
    )


def test_margin_score():
    anotator = create_mocks("jhiuh")
    prediction, X, m = create_prediction_simple()

    uncert = query_strategy.MarginRankedBatch(anotator)

    np.testing.assert_almost_equal(
        list(uncert._score(prediction)),
        [0.8, 0.6, 0.4, 0.2, 0.6],
        decimal=2
    )


def test_entropy_score():
    anotator = create_mocks("gutyg")
    prediction, X, m = create_prediction_simple()

    uncert = query_strategy.EntropyScoreRankedBatch(anotator)

    np.testing.assert_almost_equal(
        list(uncert._score(prediction)),
        [0.67, 0.61, 0.33, 0.67, 0.33],
        decimal=2
    )


def test_kl_divergence():
    anotator = create_mocks("ftyu")
    prediction, X, m = create_prediction_committee()

    kl = query_strategy.KLDivergenceRankedBatch(anotator)

    assert prediction.shape == (5, 3, 2)
    np.testing.assert_almost_equal(
        list(kl._score(prediction)),
        [0.0758, 0.0702, 0.1173, 0.0842, 0.1302],
        decimal=4
    )


def test_ral_basic():
    anotator = create_mocks("uyuuyjhh")
    prediction, X, m = create_prediction_committee()

    ral = query_strategy.RAL(
        anotator,
        uncertainty_threshold=0.9,
        threshold_greedy=0.025,
        comittee_len=3,
        budget=0.05,
        reward=1,
        penalty=-1,
        eta=0.01)

    assert prediction.shape == (5, 3, 2)
    assert ral._a == [1.0/3, 1.0/3, 1.0/3]

    s, mask = ral.select(prediction, X, classes=m.classes())

    assert len(s) == 5
    assert np.count_nonzero(mask) == 1


def test_uncertainty_correct_shape():
    anotator = create_mocks("hhhg")
    prediction, X, m = create_prediction_simple()

    uncert = query_strategy.UncertanityRankedBatch(anotator, max_samples=2)
    s, mask = uncert.select(prediction, X)

    assert prediction.shape == (5, 2)
    assert s.shape[0] == 5
    assert np.count_nonzero(mask) == 2



def test_entropy_correct_shape():
    anotator = create_mocks("yygui")
    prediction, X, m = create_prediction_simple()

    uncert = query_strategy.EntropyScoreRankedBatch(anotator, max_samples=2)
    s, mask = uncert.select(prediction, X)

    assert prediction.shape == (5, 2)
    assert s.shape[0] == 5
    assert np.count_nonzero(mask) == 2



def test_entropy_unranked_correct_shape():
    anotator = create_mocks("dresjhnjhu")
    prediction, X, m = create_prediction_simple()

    uncert = query_strategy.EntropyUnrankedBatch(anotator, max_samples=2)
    s, mask = uncert.select(prediction, X)

    assert prediction.shape == (5, 2)
    assert s.shape[0] == 5
    assert np.count_nonzero(mask) == 2



def test_entropy_unranked_correct_shape_dryrun():
    anotator = create_mocks("dresjhnjhu")
    prediction, X, m = create_prediction_simple()
    uncert = query_strategy.EntropyUnrankedBatch(anotator, 
        dry_run=True, max_samples=2)
    s, mask = uncert.select(prediction, X)
    assert prediction.shape == (5, 2)
    assert s.shape[0] == 5
    assert mask.shape[0] == 5
    assert mask[0]== True
    assert mask[1]== True
    assert mask[2]== False
    assert mask[3]== False
    assert mask[4]== False




