# import pytest

from numpy.random import seed

from alf import ml_model
from alf import context_manager
from alf import d_manager
from alf import input_manager
from alf import anotator
from alf import preprocess

SupervisedMLModel = ml_model.SupervisedMLModel
Committee = ml_model.CommitteeMLModel
ContextProvider = context_manager.ContextProvider
DbProvider = d_manager.DbProvider

d_0_path = "tests/test_files/test.csv"
wd = "/tmp/alf"


def test_anotation():
    ContextProvider.create_context("file")
    ContextProvider.get_context().set_experiment_id("anot1")
    ContextProvider.get_context().set_working_dir(wd)
    ContextProvider.get_context().set_features(["bytes_rev", "bytes"])
    DbProvider.create_context("file", d_0_path=d_0_path)
    DbProvider.get_context().fetch(test_size=0.1)
    trapcap_folder_input_manager = input_manager.TrapcapFolderInputManager(
        "tests/example_trapcaps"
    )
    anot = anotator.AnotatorDoH("tests/test_files/test_blacklist.txt")
    for flows in trapcap_folder_input_manager.get():
        assert flows is not None
        assert flows.shape[0] == 7291
        assert flows.shape[1] == 34

        result = anot.anotate(flows)
        assert result.shape[1] == flows.shape[1] + 1
        break

def test_preprocess():
    ContextProvider.create_context("file")
    ContextProvider.get_context().set_experiment_id("anot1")
    ContextProvider.get_context().set_working_dir(wd)
    ContextProvider.get_context().set_features(["bytes_rev", "bytes"])
    DbProvider.create_context("file", d_0_path=d_0_path)
    DbProvider.get_context().fetch(test_size=0.1)
    trapcap_folder_input_manager = input_manager.TrapcapFolderInputManager(
        "tests/example_trapcaps"
    )
    anot = anotator.AnotatorDoH("tests/test_files/test_blacklist.txt")
    for flows in trapcap_folder_input_manager.get():
        assert flows is not None
        assert flows.shape[0] == 7291
        assert flows.shape[1] == 34
        preprocessed = preprocess.PreprocessorDoH().preprocess(flows)
        assert preprocessed.shape[1] == 60
        result = anot.anotate(preprocessed)
        assert result.shape[1] == preprocessed.shape[1] + 1
        assert result.shape == (3341, 61)

        break

