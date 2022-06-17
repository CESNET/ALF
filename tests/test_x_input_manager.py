# import pytest

from numpy.random import seed

from alf import ml_model
from alf import context_manager
from alf import d_manager
from alf import input_manager

SupervisedMLModel = ml_model.SupervisedMLModel
Committee = ml_model.CommitteeMLModel
ContextProvider = context_manager.ContextProvider
DbProvider = d_manager.DbProvider

d_0_path = "tests/test_files/test.csv"
wd = "/tmp/alf"

def test_input_manager_trapcap_folder():
    ContextProvider.create_context("file")
    ContextProvider.get_context().set_experiment_id("qqqq1")
    ContextProvider.get_context().set_working_dir(wd)
    ContextProvider.get_context().set_features(["bytes_rev", "bytes"])
    DbProvider.create_context("file", d_0_path=d_0_path)
    DbProvider.get_context().fetch(test_size=0.1)
    trapcap_folder_input_manager = input_manager.TrapcapFolderInputManager(
        "tests/example_trapcaps"
    )
    flows = trapcap_folder_input_manager.get()
    assert flows is not None
    assert len(list(flows)) > 0

def test_input_manager_trapcap_folder_load_all():
    all_tables = []
    ContextProvider.create_context("file")
    ContextProvider.get_context().set_experiment_id("qqqq1")
    ContextProvider.get_context().set_working_dir(wd)
    ContextProvider.get_context().set_features(["bytes_rev", "bytes"])
    DbProvider.create_context("file", d_0_path=d_0_path)
    DbProvider.get_context().fetch(test_size=0.1)
    trapcap_folder_input_manager = input_manager.TrapcapFolderInputManager(
        "tests/example_trapcaps"
    )
    for flows in trapcap_folder_input_manager.get():
        all_tables.append(flows)
    assert len(all_tables) == 9
