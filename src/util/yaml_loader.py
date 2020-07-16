import os
import json
import numbers
from typing import Any, IO

import yaml


def check_truthy(something: Any) -> bool:
    if isinstance(something, str):
        return something.lower() in ['true', 't', '1']

    if isinstance(something, numbers.Number):
        return something == 1

    return False


class YamlLoader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""
    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""
        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)


def construct_include(loader: YamlLoader, node: yaml.Node) -> Any:
    """Include file referenced at node."""
    filename = os.path.abspath(
        os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, YamlLoader)
        elif extension in ('json', ):
            return json.load(f)
        return ''.join(f.readlines())


def construct_env(loader: YamlLoader, node: yaml.Node) -> Any:
    """Include environment variable referenced at node."""
    env_var_name = loader.construct_scalar(node)
    if env_var_name not in os.environ:
        print(
            f'Environment variable {env_var_name} referenced in config but is missing'
        )

    return os.environ.get(env_var_name)


def construct_feature(loader: YamlLoader, node: yaml.Node) -> Any:
    """Try to load feature flag from environment, otherwise assume false."""
    env_var_name = loader.construct_scalar(node)
    if env_var_name not in os.environ:
        return False

    truthy = check_truthy(os.environ.get(env_var_name))
    enabled_msg = 'enabled' if truthy else 'disabled'

    print(f'Feature flag {env_var_name}: {enabled_msg}')

    return truthy


yaml.add_constructor('!include', construct_include, YamlLoader)
yaml.add_constructor('!env', construct_env, YamlLoader)
yaml.add_constructor('!feature', construct_feature, YamlLoader)
