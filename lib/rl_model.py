from abc import ABCMeta, abstractmethod


class RLModel(metaclass=ABCMeta):
    @abstractmethod
    def get_action(self):
        raise NotImplementedError()

    @abstractmethod
    def update(self):
        raise NotImplementedError()
