class Singleton(type):
    """
    metaclass singleton pattern from 
    http://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger():
    __metaclass__ = Singleton
    def __init__(self, name=None, stream=None):
        if not name:
            #hack to get calling module's name
            #http://stackoverflow.com/a/602967
            frame = inspect.currentframe()
            while hasattr(frame.f_back, "f_globals"):
                frame = frame.f_back
            name = frame.f_globals['__file__']
        else:
            name = name
        self.setup(name, stream)
        if not hasattr(self, "level"):
            self.set("INFO")
    def set(self, level):
        import logging
        level = level.upper()
        level_dict = {"DEBUG":logging.DEBUG,
                      "INFO":logging.INFO,
                      "WARNING":logging.WARNING,
                      "ERROR":logging.ERROR
        }
        if not level in level_dict:
            raise Exception("ERROR: invalid logging level {level}".format(**vars()))
    
        level_val = level_dict[level]
        self.level = level_val
        self.logr.setLevel(level_val)

        for h in self.logr.handlers:
            h.setLevel(level_val)
        
    def setup(self, name="root", stream=None):
        import sys
        import logging
        if not stream:
            stream = sys.stderr

        logger = logging.getLogger(name)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
       
        if not (logger.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        self.logr = logger
    def debug(self,s):
        self.logr.debug(s)
    def info(self,s):
        self.logr.info(s)
    def warning(self,s):
        self.logr.warning(s)
    def error(self,s):
        self.logr.error(s)




