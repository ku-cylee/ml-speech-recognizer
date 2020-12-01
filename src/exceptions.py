class FileParsingFailureException(BaseException):

    def __init__(self, name, path):
        self.name = name
        self.path = path

    def __str__(self):
        return 'Failed to parse {} file: {}'.format(self.name, self.path)
