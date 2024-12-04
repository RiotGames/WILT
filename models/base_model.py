from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def perform_step(self, conversation_history):
        pass

    @abstractmethod
    def initialize_conversation(self, system_prompt):
        pass
