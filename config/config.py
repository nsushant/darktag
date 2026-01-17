import json 
import os

class Config:
    def __init__(self, config_file=None):
        if config_file is None:
            # Get path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_file = os.path.join(current_dir, "config.json")
        
        with open(config_file, 'r') as f:
            self._config = json.load(f)
    
    def get_path(self, key):
        return self._config['paths'][key]
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    def get_all_paths(self):
        return self._config.get('paths', {})


config = Config()

