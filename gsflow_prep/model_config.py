import os.path
from configparser import ConfigParser


class HRUParameters:

    def __init__(self, config_path):
        section_names = ['FIELDS']
        parser = ConfigParser()
        parser.optionxform = str
        config_path = os.path.join(os.path.dirname(config_path), 'field_list.ini')
        found = parser.read(config_path)
        if not found:
            raise ValueError('No config file found!')

        for name in section_names:
            self.__dict__.update(parser.items(name))


class PRMSConfig:
    def __init__(self, config_path):
        section_names = ['INPUTS', 'PROCESSING']
        parser = ConfigParser()
        parser.optionxform = str
        found = parser.read(config_path)
        if not found:
            print('{} not found'.format(config_path))
            raise ValueError('No config file found!')

        # self.fields = HRUParameters(config_path)

        for name in section_names:
            self.__dict__.update(parser.items(name))


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
