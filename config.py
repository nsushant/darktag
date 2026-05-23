import json 
import os

_CONFIG_FILE_VAR = "DARKTAG_CONFIG"

class Config:
    def __init__(self, config_file=None):
        if config_file is None:
            config_file = os.environ.get(_CONFIG_FILE_VAR)
        if config_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(current_dir, "config.json")
        
        self._config_file = config_file
        self._default_config_file = config_file
        self._load()
    
    def _load(self):
        with open(self._config_file, 'r') as f:
            self._config = json.load(f)
    
    def reload(self, config_file=None):
        if config_file is not None:
            self._config_file = config_file
        else:
            self._config_file = self._default_config_file
        self._load()
    
    def get_path(self, key):
        return self._config['paths'][key]
    
    def get(self, key, param):
        return self._config[key][param]
    
    def get_all_paths(self):
        return self._config.get('paths', {})
    
    def get_with_default(self, key, param, default=None):
        try:
            return self._config[key][param]
        except (KeyError, TypeError):
            return default


config = Config()

