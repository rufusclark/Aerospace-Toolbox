from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class FrozenDataclassBase:
    ...

    def _set(self, attr: str, value: Any) -> None:
        valid_fields = {f.name for f in fields(self)}

        if attr not in valid_fields:
            raise AttributeError(
                f"Trying to set an attribute that hasn't been declared. {attr!r} is not a field of {type(self).__name__}"
            )

        object.__setattr__(self, attr, value)


if __name__ == "__main__":
    @dataclass(frozen=True)
    class Test(FrozenDataclassBase):
        foo: str = field(init=False, default="")

        def __post_init__(self):
            self._set("foo", "bar")

    print(Test())
