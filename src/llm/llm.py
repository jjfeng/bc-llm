from abc import ABC, abstractmethod
from src.llm.dataset import TextDataset

class LLM(ABC):
    def __init__(self, seed, model_type: str, logging):
        self.seed = seed
        self.model_type = model_type
        self.logging = logging

    @abstractmethod
    def get_output(
          self, 
          prompt, 
          max_new_tokens,
          is_image:bool = False
          ) -> str:
        return NotImplemented

    @abstractmethod
    def get_outputs(
          self, 
          dataset: TextDataset, 
          batch_size:int = 4, 
          top_k:int = 10,
          is_image:bool = False
          ) -> list[str]:
        return NotImplemented
