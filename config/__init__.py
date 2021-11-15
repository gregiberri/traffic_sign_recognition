import os

from ray import tune
import yaml

yaml.Dumper.ignore_aliases = lambda *args: True


class ConfigNamespace(object):
    def __init__(self, config):
        """
        Make config namespace from dict or from a yaml file path.

        :param config: a dictionary or a path of a yaml file containing the config.
        """
        if isinstance(config, dict):
            self.update(config)
        elif isinstance(config, str):
            self.load(config)
        else:
            raise ValueError('The constructor parameter of the ConfigNameSpace object should be a dict or a str path.')

    def __len__(self):
        return len(self.__dict__)

    def dict(self):
        """
        Make dict from the namespace
        """
        config_dict = dict()
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigNamespace):
                subdict = v.dict()
                config_dict[k] = subdict
            else:
                config_dict[k] = v
        return config_dict

    def load(self, path):
        """
        Load namespace from yaml file
        """
        path = os.path.join('config', 'config_files', path + '.yaml')
        with open(path) as f:
            config = yaml.safe_load(f)
        self.update(config)

    def save(self, path):
        """
        Save namespace into yaml file.
        """
        assert path[-3:] == 'yml' or path[-4:] == 'yaml', 'The file should be a yml file: *.yml or *.yaml'

        with open(path, 'w') as nf:
            config_dict = self.dict()
            yaml.dump(config_dict, nf)

    def str(self, step='  '):
        """
        Make string from the namespace for easier printing
        """
        object_str = ''
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigNamespace):
                new_step = step + '  '
                object_str += step + k + ': \n' + v.str(new_step)
            else:
                object_str += step + k + ': ' + str(v) + '\n'

        return object_str

    def __repr__(self):
        return 'config: \n' + self.str()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def get_hpo_func(self, item):
        """
        Init hpo objects according to the lists containing hpo specifications.
        """
        if isinstance(item, list):
            if isinstance(item[0], str):
                if hasattr(tune, item[0]):
                    function = getattr(tune, item[0])
                    return function(*item[1])
        return item

    def update(self, config):
        if isinstance(config, ConfigNamespace):
            config = config.__dict__
        elif isinstance(config, dict):
            pass
        else:
            raise ValueError('other object used for update should be Namespace of dict.')

        if config.get('base_config') is not None:
            self.load(config['base_config'])

        for key, value in config.items():
            value = self.get_hpo_func(value)
            # if value is a dict, we want to make a child confignamespace from it
            if isinstance(value, dict):
                self.update({key: ConfigNamespace(value)})
            elif isinstance(self.__dict__.get(key), ConfigNamespace):
                self.__dict__.get(key).update(value)
            else:
                self.__dict__.update({key: value})
