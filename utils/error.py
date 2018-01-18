class ParamNotFoundError(Exception):
    def __init__(self, param_name):
        self.msg = 'Parameter \'{0}\' not found'.format(param_name)
    def __str__(self):
        return repr(self.msg)
    
class InvalidParamError(Exception):
    def __init__(self, param_name, value):
        self.msg = 'Invalid parameter \'{0}\': \'{1}\''.format(param_name, value)
    def __str__(self):
        return repr(self.msg) 