from abc import ABC, abstractmethod

class Renderer(ABC):
    @abstractmethod
    def __init__(self, G):
        pass

    @abstractmethod
    def random_params(self):
        pass

    @abstractmethod
    def get_custom_loss(self):
        return 0
    