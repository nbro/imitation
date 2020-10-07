import abc
import warnings
from typing import Any, Generic, Iterable, Iterator, List, Mapping, Tuple, TypeVar

import gym
import numpy as np
import torch as th

from imitation.data import types

S = TypeVar("S")
T = TypeVar("T")


class DataLoaderWrapper(Generic[S, T], Iterable[T], abc.ABC):
    def __init__(self, data_loader: Iterable[S]):
        self.data_loader = data_loader

    @abc.abstractmethod
    def __iter__(self) -> Iterator[T]:
        pass


class BidirectionalTransform(abc.ABC):
    @abc.abstractmethod
    def forwards(
        self,
        transitions_data: Mapping,
    ) -> dict:
        """Defines a batched transformation from raw to transformed Transitions.

        TODO(shwang): Statement is not true. Look at diagram.
        This method is intended to be applied on batches of data coming from `Env.step`,
        `VecEnv.step`, or `DataLoader.__iter__`.

        `forwards` should be a pure function. In other words, any instance of
        BidirectionalTransform returns the same outputs given the same inputs.

        Args:
            transitions_data: A Mapping roughly corresponding to an instance of
                Transitions or TransitionsWithRew. It must have the keys "obs", "acts",
                "next_obs", and "infos". Optionally it can include the "rews" key.

                Concrete implementations of this method that transform "rews" must be
                able to gracefully handle the case where "rews" is not provided.

        Returns:
            A dict with the same keys as `transitions_data`, but where some values
            have been replaced.
        """

    @abc.abstractmethod
    def forwards_obs(
        self,
        obs: Any,
    ) -> Any:
        """Defines a batched transformation from raw to transformed observations.

        Used transform observations returned by `gym.Wrapper.reset()`.
        """

    @abc.abstractmethod
    def backwards_act(self, transformed_acts: np.ndarray):
        """Defines a (pseudo-)inverse transformation from transformed act to raw act.

        `backwards_act` should be a pure function. In other words, any instance of
        BidirectionalTransform returns the same outputs given the same inputs.

        Args:
            transformed_acts: Batched actions transformed from raw actions by
                `forwards()`.

        Returns:
            A raw action. Not required to be the same raw action as that passed into
            `forwards()` because `forwards()` might not be one-to-one / injective.
        """


class BidirectionalDataLoaderWrapper(DataLoaderWrapper[Mapping, dict]):
    """DataLoaderWrapper that corresponds to an instance of BidirectionalTransform.

    Primarily used by BidirectionalGymWrapper.
    """

    def __init__(
        self,
        data_loader: types.DataLoaderInterface,
        transform: BidirectionalTransform,
    ):
        self.data_loader = data_loader
        self.transform = transform
        super().__init__(data_loader)

    def __iter__(self) -> dict:
        for trans in self.data_loader:
            result = self.transform.forwards(trans)
            result["acts"] = self.transform.backwards_act(trans["acts"])
            yield result


class DefinesDataLoaderWrapper(abc.ABC, Generic[S, T]):
    """Concrete instances of this abstract mixin define a `DatasetWrapper`.

    Intended for use as an abstract mix-in in a subclass of `gym.Wrapper` so that the
    `gym.Wrapper` can be processed by `wrap_data_loader_with_env_wrappers`.
    """

    @abc.abstractmethod
    def apply_dataset_wrapper(
        self,
        data_loader: types.DataLoaderInterface,
    ) -> DataLoaderWrapper[S, T]:
        """Wraps a dataset using the dataset.

        Args:
            data_loader: A DataLoader to wrap.

        Returns:
            A wrapped data_loader.
        """


class BidirectionalGymWrapper(
    gym.Wrapper,
    DefinesDataLoaderWrapper[Mapping, dict],
):
    """Simple Gym Wrapper defined by an instance of BidirectionalTransform.

    It trivially implements `DefinesDataLoaderWrapper.apply_data_loader_wrapper`
    using the same `BidirectionalTransform`.
    """

    def __init__(self, env: gym.Env, transform: BidirectionalTransform):
        self.transform = transform
        super().__init__(env)

    def reset(self):
        obs = self.env.reset()
        return self.forward_obs(obs)

    def step(self, action: Any) -> Tuple[Any, float, dict]:
        act = self.transform.backwards_act(action)
        return self.env.step(act)

    def apply_data_loader_wrapper(
        self,
        data_loader: types.DataLoaderInterface,
    ) -> DataLoaderWrapper[Mapping, dict]:
        return BidirectionalDataLoaderWrapper(data_loader, self.transform)


def wrap_dataset_with_env_wrappers(
    data_loader: Iterable,
    env: gym.Env,
    warn_on_ignore_wrapper: bool = True,
) -> Iterable:
    """Apply DataLoaderWrappers corresponding to each gym.Wrapper on `env`.

    Args:
        data_loader: Base `DataLoaderInterface` interface to wrap with
            `DataLoaderWrapper`s corresponding to each gym.Wrapper.
        env: For every gym.Wrapper `wrapper` wrapping `env` which also subclasses
            `DefinesDataLoaderWrapper`, `data_loader` is wrapped using
            `wrapper.apply_data_loader_wrapper()`.

            Any `gym.Wrapper` that doesn't subclass `GymWrapperDatasetMixin` is skipped.
        warn_on_ignore_wrapper: If True, then warn with RuntimeWarning for every
            `gym.Wrapper` that doesn't subclass `GymWrapperDatasetMixin`.

    Returns:
        `data_loader` wrapped by a new `DatasetWrapper` for every `gym.Wrapper` around
        `env` also subclasses `DefinesDataLoaderWrapper`. If `env` has no wrappers,
        then the return is simply `data_loader`.
    """
    # Traverse all the gym.Wrapper starting from the outermost wrapper. When re-apply
    # them to the Dataset as MineRLDatasetWrapper, we need to apply in reverse order
    # (starting with innermost wrapper), so we save gym.Wrappers in a list first.
    compatible_wrappers: List[DefinesDataLoaderWrapper] = []
    curr = env
    while isinstance(curr, gym.Wrapper):
        if isinstance(curr, DefinesDataLoaderWrapper):
            compatible_wrappers.append(curr)
        elif warn_on_ignore_wrapper:
            warnings.warn(
                f"{curr} doesn't subclass DefinesDatasetWrapperMixin. Skipping "
                "this Wrapper when creating Dataset processing wrappers.",
                RuntimeWarning,
            )
        curr = curr.env

    for wrapper in reversed(compatible_wrappers):
        data_loader = wrapper.apply_dataset_wrapper(data_loader)
    return data_loader


class TensorToNumpyDataLoaderWrapper(DataLoaderWrapper[Mapping, dict]):
    """For dict batch, make new dict where every formerly Tensor value is a Numpy array.

    All other values are left the same.
    """

    def __init__(self, data_loader: types.DataLoaderInterface):
        super().__init__(data_loader)

    def __iter__(self) -> Iterator[dict]:
        for x in self.data_loader:
            result = dict(x)
            for k, v in result.items():
                if isinstance(v, th.Tensor):
                    result[k] = v.detach().numpy()
            yield result
