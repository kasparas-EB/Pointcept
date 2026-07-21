from functools import partial
from typing import Any, Callable

import torch
from . import activation_checkpoint as checkpoint


def _apply_no_checkpoint(module: torch.nn.Module, *inputs: tuple[Any]):
    """Default no checkpointing behavior"""
    return module(*inputs)


_torch_activation_checkpoint = partial(
    checkpoint.checkpoint,
    use_reentrant=False,
)


class MaybeCheckpoint:
    def __call__(self, module: torch.nn.Module, *inputs: tuple[Any]):
        return self._checkpoint_func(module, *inputs)

    _inactive_func = staticmethod(_apply_no_checkpoint)
    _active_func = staticmethod(_torch_activation_checkpoint)
    _checkpoint_func = _inactive_func

    @classmethod
    def set_checkpoint_func(cls, func: Callable[[torch.nn.Module, tuple[Any]], Any]):
        cls._active_func = staticmethod(func)

    @classmethod
    def activate(cls):
        cls._checkpoint_func = cls._active_func
    
    @classmethod
    def deactivate(cls):
        cls._checkpoint_func = cls._inactive_func


maybe_checkpoint = MaybeCheckpoint()
set_checkpoint_func = maybe_checkpoint.set_checkpoint_func
activate = maybe_checkpoint.activate
deactivate = maybe_checkpoint.deactivate
