import os
import json
import tempfile

from darktag.config import config


def test_config_defaults():
    assert config.get("tagging", "method") == "angular_momentum"
    assert config.get("tagging", "ftag") == 0.01
    assert config.get("darklight", "n") == 500


def test_config_get_path():
    path = config.get_path("tangos_path")
    assert isinstance(path, str)
    assert len(path) > 0


def test_config_get_all_paths():
    paths = config.get_all_paths()
    assert "tangos_path" in paths
    assert "pynbody_path" in paths
    assert "manual_halonum_path" in paths
    assert "manual_mstar_path" in paths


def test_config_reload():
    old_tangos = config.get_path("tangos_path")

    test_config = {
        "paths": {
            "tangos_path": "/tmp/test_custom/",
            "pynbody_path": "/tmp/test_custom/",
            "manual_halonum_path": "",
            "manual_mstar_path": ""
        },
        "tagging": {"method": "spatial", "ftag": 0.05},
        "darklight": {"n": 100, "DMO_OR_HYDRO": "DMO", "poccupied": "all"}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        tmp_path = f.name

    try:
        config.reload(tmp_path)
        assert config.get_path("tangos_path") == "/tmp/test_custom/"
        assert config.get("tagging", "method") == "spatial"
        assert config.get("tagging", "ftag") == 0.05
        assert config.get("darklight", "n") == 100
    finally:
        os.unlink(tmp_path)
        config.reload()


def test_config_unknown_key():
    import pytest
    with pytest.raises(KeyError):
        config.get_path("nonexistent_path")


def test_config_unknown_section():
    import pytest
    with pytest.raises(KeyError):
        config.get("nonexistent", "param")


def test_config_env_var_override():
    test_config = {
        "paths": {
            "tangos_path": "/env/test/",
            "pynbody_path": "/env/test/",
            "manual_halonum_path": "",
            "manual_mstar_path": ""
        },
        "tagging": {"method": "binding_energy", "ftag": 0.1},
        "darklight": {"n": 50, "DMO_OR_HYDRO": "DMO", "poccupied": "nadler20"}
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        tmp_path = f.name

    try:
        old_env = os.environ.get("DARKTAG_CONFIG")
        os.environ["DARKTAG_CONFIG"] = tmp_path
        from darktag.config import Config
        env_config = Config()
        assert env_config.get_path("tangos_path") == "/env/test/"
        assert env_config.get("tagging", "method") == "binding_energy"
    finally:
        if old_env is None:
            del os.environ["DARKTAG_CONFIG"]
        else:
            os.environ["DARKTAG_CONFIG"] = old_env
        os.unlink(tmp_path)
