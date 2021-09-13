import torch.nn as nn
from abc import abstractmethod

from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor') # Type 지정


class VAE_Frame(nn.Module):
    def __init__(self):
        super().__init__()

    def encoder(self, input : Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decoder(self, input:Tensor):
        raise NotImplementedError

    def generate(self, x: Tensor,**kwargs) -> Tensor:
        """
        :param x: Image
        :return: Image
        """
        raise NotImplementedError

    def sample(self, samples_num: int, **kwargs) -> Tensor :
        """
        :param samples_num:
        :return: Sampled images
        """
        raise NotImplementedError

    @abstractmethod # 이 Method는 꼭 구현해야함을 강제.
    def forward(self):
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs):
        pass