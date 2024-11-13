 
[PYTHON]
class Asset(metaclass=ABCMeta):
    """
    Abstract class for all asset types.
    """

    @abstractmethod
    def __init__(self, id=None, type=None, path=None, filename=None, content=None, parent=None, user_id=None):
        self.id = int(id) if id else None
        self.type = type
        self.path = path
        self.filename = filename
        self.content = content
        self.parent = parent
        self.user_id = int(user_id) if user_id else None

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def delete(self):
        pass

    @abstractmethod
    def getContent(self):
        pass

    @abstractmethod
    def setContent(self, content):
        pass

    @abstractmethod
    def getThumbnail(self, width=100, height=100, upscale=False, aspectRatio=True):
        pass

    @abstractmethod
    def getProperty(self, name):
        pass

    @abstractmethod
    def setProperty(self, name, value):
        pass

    @abstractmethod
    def deleteProperty(self, name):
        pass

    @abstractmethod
    def getProperties(self):
        pass

    @abstractmethod
    def setProperties(self, properties):
        pass

    @abstractmethod
    def getChilds(self):
        pass

    @abstractmethod
    def hasChilds(self):
        pass

    @abstractmethod
    def addChild(self, child):
        pass

    @abstractmethod
    def moveTo(self, parent):
        pass

    @abstractmethod
    def deleteAllVariants(self):
        pass

    @abstractmethod
    def getStream(self):
        pass

    @abstractmethod
    def getContentStream(self):
        pass

    @abstractmethod
    def setContentFromString(self, content):
        pass

    @abstractmethod
    def setContentFromFile(self, file):
        pass

    @abstractmethod
    def closeStream(self):
        pass
[/PYTHON]
