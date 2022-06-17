import pytest


from alf import context_manager
from alf import d_manager
from alf import ip_flow

ContextProvider = context_manager.ContextProvider
IPFlowsDataFrame = ip_flow.IPFlowsDataFrame
DManagerFile = d_manager.DManagerFile

d_0_path = "tests/test_files/test.csv"
wd = "/tmp/alf"
features = [
    'bytes_rev',
    'bytes'
]

ContextProvider.create_context("file")
ContextProvider.get_context().set_features(features)


def test_fetch():
    ContextProvider.get_context().set_working_dir(wd)
    ContextProvider.get_context().set_experiment_id("alf_t11")
    dm = DManagerFile(d_0_path)
    dm.fetch(test_size=0.5)
    assert len(dm.get_all()) == 6


def test_initially_empty():
    ContextProvider.get_context().set_experiment_id("alf_t10")
    dm = DManagerFile(d_0_path)
    assert dm.get_train_set() is None
    assert dm.get_test_set() is None
    assert len(dm.get_all()) == 0


def test_fetch_twice():
    ContextProvider.get_context().set_experiment_id("alf_t12")
    dm = DManagerFile(d_0_path)
    dm.fetch(test_size=0.5)
    dm.fetch(test_size=0.5)
    assert len(dm.get_all()) == 6


def test_fetch_then_append():
    ContextProvider.get_context().set_experiment_id("alf_t13")
    dm = DManagerFile(d_0_path)
    dm.fetch(test_size=0.5)
    assert len(dm.get_all()) == 6
    dm.append_to_db(IPFlowsDataFrame([{
        "class": True,
        "bytes_rev": 44,
        "bytes": 44,
        "packets": 44,
        "packets_rev": 44
    }]))
    assert len(dm.get_all()) == 7


def test_append_no_class():
    ContextProvider.get_context().set_experiment_id("alf_t14")
    dm = DManagerFile(d_0_path)
    dm.fetch(test_size=0.5)
    with pytest.raises(ValueError):
        dm.append_to_db(IPFlowsDataFrame([{
            "class": None,
            "bytes_rev": 44,
            "bytes": 44,
            "packets": 44,
            "packets_rev": 44
        }]))


def test_commit():
    ContextProvider.get_context().set_experiment_id("alf_t15")
    ContextProvider.get_context().set_features(features)
    dm = DManagerFile(d_0_path)
    dm.fetch(test_size=0.5)
    dm.append_to_db(IPFlowsDataFrame([{
        "class": True,
        "bytes_rev": 44,
        "bytes": 44,
        "packets": 44,
        "packets_rev": 44
    }]))
    dm.commit()
    dm.fetch(test_size=0.5)
    assert len(dm.get_all()) == 7
