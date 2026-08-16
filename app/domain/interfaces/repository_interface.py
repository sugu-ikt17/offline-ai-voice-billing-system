"""Generic repository contract shared across all entity-specific repositories."""

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class RepositoryInterface(ABC, Generic[T]):
    @abstractmethod
    def create(self, item: T) -> T:
        raise NotImplementedError

    @abstractmethod
    def get_all(self) -> list[T]:
        raise NotImplementedError

    @abstractmethod
    def get_by_id(self, item_id: int) -> Optional[T]:
        raise NotImplementedError

    @abstractmethod
    def update(self, item: T) -> T:
        raise NotImplementedError

    @abstractmethod
    def delete(self, item: T) -> None:
        raise NotImplementedError
