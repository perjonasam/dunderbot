import yaml
import pickle

from src.util.yaml_loader import YamlLoader

# store previous config reads
memoize = {}

DEFAULT_PATH = "config/config.yml"


def set_default_path(path):
    global DEFAULT_PATH
    DEFAULT_PATH = path


def get_config(path=None):
    """Retrieve config."""
    if path is None:
        path = DEFAULT_PATH
    if path in memoize.keys():
        return memoize[path]

    file_ending = path.split('.')[-1]
    if file_ending in ['pkl', 'pickle', 'issues']:
        try:
            print(f'loading config from {path}')
            with open(path, 'rb') as file:
                config = pickle.load(file)
            config = AttributeDict(config)
            memoize[path] = config
            return config
        except pickle.UnpicklingError as e:
            print(e)
    else:
        try:
            print(f'loading config from {path}')
            config = load_yaml(path)
            config = AttributeDict(config)
            memoize[path] = config
            return config
        except yaml.YAMLError as e:
            print(e)


def save_config(*, config, path):
    """Dumps config to a file"""
    with open(path, 'wb') as file:
        pickle.dump(config, file)


def load_yaml(path):
    """Loads a yaml file with its items accessible as attributes.
    Args:
        path: path to .yml file
    Returns:
        config (AttributeDict): as specified in .yml file
    Example:
        >>> config = load_yaml(path)
        >>> print(config.strings.hello_world)
        "Hello World."
    """
    with open(path, 'r') as file:
        config = yaml.load(file, YamlLoader)
    return config


class AttributeDict(dict):
    def __init__(self, iterable=None, **kwargs):
        if iterable is not None:
            kv = self._key_value_generator(iterable)
            super().__init__(kv, **kwargs)
        else:
            super().__init__(**kwargs)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __getattr__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return super().__getattribute__(key)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttributeDict(value)
        super().__setitem__(key, value)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __delattr__(self, key):
        super().pop(key)

    @staticmethod
    def _key_value_generator(it):
        if isinstance(it, dict):
            it = it.items()

        for key, value in it:
            if isinstance(value, dict):
                yield (key, AttributeDict(value))
            else:
                yield (key, value)
