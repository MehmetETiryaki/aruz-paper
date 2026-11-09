

from typing import (Dict, Generic, Hashable, Iterable, Iterator, List, Optional, TypeVar, overload, Union)

T = TypeVar('T', bound=Hashable)

class Integerizer(Generic[T]):

    def __init__(self, iterable: List[T] = []):
        self._objects: List[T] = []
        self._indices: Dict[T, int] = {}
        self.update(iterable)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Integerizer):
            return self._objects == other._objects
        else:
            return False

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self) -> Iterator[T]:
        return iter(self._objects)

    def __contains__(self, obj: T) -> bool:
        return self.index(obj) is not None

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> List[T]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T, List[T]]:
        return self._objects[index]

    def index(self, obj: T, add: bool = False) -> Optional[int]:
        try:
            return self._indices[obj]
        except KeyError:
            if not add:
                return None

            i = len(self)
            self._objects.append(obj)
            self._indices[obj] = i
            return i

    def add(self, obj: T) -> None:
        self.index(obj, add=True)

    def update(self, iterable: Iterable[T]) -> None:
        for obj in iterable:
            self.add(obj)

if __name__ == "__main__":
    import doctest

    doctest.testmod()
