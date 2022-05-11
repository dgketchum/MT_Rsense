import os
from configparser import ConfigParser


class PRMSConfig:
    def __init__(self, config_path):
        parser = ConfigParser()
        parser.optionxform = str
        found = parser.read(config_path)
        if not found:
            print('{} not found'.format(config_path))
            raise ValueError('No config file found!')

        for name in ['MODEL_INFO', 'MODEL_PATHS']:
            for k, v in parser.items(name):
                if 'units' in k:
                    self.__dict__.update({k: int(v)})
                else:
                    self.__dict__.update({k: v})

        for name in ['INPUT_PATHS']:
            _dir = self.__dict__['project_folder']
            dct = {k: os.path.join(_dir, v) for k, v in parser.items(name)}
            self.__dict__.update(dct)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
