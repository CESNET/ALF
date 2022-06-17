import pytest

from alf import context_manager

ContextProvider = context_manager.ContextProvider
ContextFileMetrics = context_manager.ContextFileMetrics


def test_factory_functionality_none():
    """Test factory function."""
    with pytest.raises(ValueError):
        ContextProvider.create_context(None)


def test_factory_functionality_invalid():
    """Test factory function."""
    with pytest.raises(ValueError):
        ContextProvider.create_context("invalid_option")


def test_factory_functionality_valid():
    """Test factory function."""
    assert ContextProvider.create_context("file") is None


def test_get_context():
    """Test singleton function"""
    ContextProvider.create_context("file")
    assert ContextProvider.get_context() is not None
    assert isinstance(
        ContextProvider.get_context(),
        ContextFileMetrics
    )


def test_id_setting_none():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    with pytest.raises(TypeError):
        ctx.set_experiment_id(None)


def test_id_setting_not_string():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    with pytest.raises(TypeError):
        ctx.set_experiment_id(1)


def test_id_setting():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    ctx.set_experiment_id("test")
    assert ctx.get_experiment_id() == "test"


def test_wd_setting_none():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    with pytest.raises(TypeError):
        ctx.set_working_dir(None)


def test_wd_setting_not_path():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    with pytest.raises(ValueError):
        ctx.set_working_dir("notdirectorypath")


def test_wd_setting():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    ctx.set_working_dir("/tmp")
    assert ctx.get_working_dir() == "/tmp"


def test_features_setting():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    ctx.set_features(["feature1", "feature2"])
    assert ctx.get_features() == ["feature1", "feature2"]


def test_features_setting_none():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    with pytest.raises(ValueError):
        ctx.set_features(None)


def test_features_setting_not_list():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    with pytest.raises(ValueError):
        ctx.set_features("not_a_list")


def test_features_setting_not_string():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    with pytest.raises(ValueError):
        ctx.set_features([1, 2])


def test_metrics_setting():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    assert ctx.get_metrics() == {}
    ctx.append_metrics({"metric1": 1, "metric2": 2})
    assert ctx.get_metrics() == {"metric1": 1, "metric2": 2}
    ctx._metrics = {}


def test_metrics_merging():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    assert ctx.get_metrics() == {}
    ctx.append_metrics({"metric1": 1, "metric2": 2})
    assert ctx.get_metrics() == {"metric1": 1, "metric2": 2}
    ctx.append_metrics({"m3": 3, "m4": 4})
    assert ctx.get_metrics() == {"metric1": 1, "metric2": 2, "m3": 3, "m4": 4}
    ctx._metrics = {}


def test_metrics_mergin_conflict():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    assert ctx.get_metrics() == {}
    ctx.append_metrics({"metric1": 1, "metric2": 2})
    assert ctx.get_metrics() == {"metric1": 1, "metric2": 2}
    ctx.append_metrics({"metric3": 3, "metric2": 4})
    assert ctx.get_metrics() == {"metric1": 1, "metric2": 4, "metric3": 3}
    ctx._metrics = {}


def test_metrics_setting_none():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    with pytest.raises(TypeError):
        ctx.append_metrics(None)


def test_metrics_setting_not_dict():
    """Test context setting"""
    ContextProvider.create_context("file")
    ctx = ContextProvider.get_context()
    with pytest.raises(TypeError):
        ctx.append_metrics("not_a_dict")
