import zsfm


def test_version():
    assert isinstance(zsfm.__version__, str)
    assert zsfm.__version__.count(".") >= 1


def test_list_models():
    models = zsfm.list_models()
    assert "toto" in models
    assert "chronos" in models
    assert "ttm" in models
    assert "mitra" in models
    assert "tabfm" in models
    assert len(models) == 16


def test_list_forecasters_tabular():
    assert len(zsfm.list_forecasters()) == 11
    assert len(zsfm.list_tabular()) == 5
    assert set(zsfm.list_forecasters()) | set(zsfm.list_tabular()) == set(zsfm.list_models())


def test_convert_delete_unkown_model_errors():
    import pytest

    with pytest.raises(Exception, match="unknown model"):
        zsfm.convert("not_a_model")

    with pytest.raises(Exception, match="unknown model"):
        zsfm.delete("not_a_model")


def test_model_classes_exist():
    # All 16 Model classes should be importable at top level
    for cls in [
        "TotoModel",
        "ChronosModel",
        "TimesFmModel",
        "SundialModel",
        "TtmModel",
        "LagLlamaModel",
        "MomentModel",
        "MoiraiModel",
        "Moirai2Model",
        "FlowStateModel",
        "TirexModel",
        "MitraModel",
        "TabDptModel",
        "TabIclModel",
        "TabPfnModel",
        "TabFmModel",
    ]:
        assert hasattr(zsfm, cls), f"missing {cls}"
        assert callable(getattr(zsfm, cls))


def test_delete_no_cache_is_ok(tmp_path):
    # delete on a non-existent cache should not raise, just print "Nothing to delete."
    # Use a temporary model_dir so we don't touch real models/
    zsfm.delete("toto", model_dir=str(tmp_path))
    zsfm.delete("mitra", model_dir=str(tmp_path))
