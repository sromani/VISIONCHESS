"""Stub registry — satisfies pickle imports only."""


class Registry:
    def __init__(self) -> None:
        self._items: dict = {}

    def register(self, cls):
        self._items[cls.__name__] = cls
        return cls

    def register_as(self, name: str):
        def decorator(registry):
            return registry
        return decorator
