from typing import TypeVar, List

T = TypeVar('T')


class CircularBuffer:
    def __init__(self, size: int):
        self.size = size
        self.list: List[T] = [None] * size

    def append(self, item: T):
        self.list.pop(0)
        self.list.append(item)

    def get(self) -> List[T]:
        return self.list

    def get_last(self) -> T:
        return self.list[-1]
