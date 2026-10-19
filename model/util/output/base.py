from __future__ import annotations
from typing import Protocol, Literal, Any, Callable
from dataclasses import dataclass, is_dataclass
import pathlib
import warnings
import re
import numpy as np

__all__ = (
    "InfoBase",
    "PlotBase"
)

#
# counts
#


@dataclass
class Count:
    figures: int = 0
    tables: int = 0


count = Count()

#
# base class definitions
#


class HasCategoryAndName(Protocol):
    category: str

    @property
    def name(self) -> str:
        ...


class OutputType[T: HasCategoryAndName]:
    output_type: Literal["info", "plot"]

    def __init__(self, model: T) -> None:
        self.model = model

    @property
    def category(self) -> str:
        return self.model.category


class InfoBase[T: HasCategoryAndName](OutputType[T]):
    output_type = "info"

    def all(self, *args, show: bool = True, save: bool = True):
        from .info import Table
        for name in dir(self):
            method = getattr(self, name)

            if name in dir(type(self)):
                method = getattr(self, name)

                if isinstance(method, Table):
                    plot = method(*args, save=save, show=show)

            # TODO: Remove old code
            if getattr(method, "_scientific", False):
                method(*args, show=show, save=save)


class PlotBase[T: HasCategoryAndName](OutputType[T]):
    output_type = "plot"

    def all(self, *args, show: bool = False, save: bool = True):
        from .plot import Plots, _show

        for name in dir(type(self)):
            method = getattr(self, name)

            if isinstance(method, Plots):
                plot = method(*args, show=False, save=save)

                if show:
                    plot.fig(*args)

        if show:
            _show()

#
# path handling
#

def common_suffix(strings: list[str]):
    if not strings:
        return ""

    suffix = strings[0]

    for s in strings[1:]:
        i = 1
        while i <= min(len(suffix), len(s)) and suffix[-i] == s[-i]:
            i += 1
        suffix = suffix[len(suffix) - i + 1:]

        if not suffix:
            return ""

    return suffix

def common_prefix(a: str, b: str) -> str:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return a[:i]


@dataclass
class Path:
    instance_name: str
    method_name: str
    category_name: str
    output_type: str
    kwargs: dict[str, Any]
    _title: str = ""

    def __post_init__(self):
        self.instance_name = self._sanitise(self.instance_name)
        self.method_name = self._sanitise(self.method_name)
        self.category_name = self._sanitise(self.category_name)
        self.output_type = self._sanitise(self.output_type)
        self._title = self._sanitise(self._title)

    def __add__(self, other: Path) -> Path:
        if not isinstance(other, Path):
            raise TypeError(
                f"{type(other)} is not supported for this operation"
            )

        if (
            self.method_name != other.method_name
            or self.category_name != other.category_name
        ):
            import warnings
            warnings.warn(
                f"__add__ may not be compatible: self={self}, other={other}"
            )

        prefix = common_prefix(
            self.title if self.title else self.instance_name,
            other.instance_name
        ).rstrip("- ")
        title = "Overview" if len(
            prefix) < 4 else f"{prefix} (overview)"

        return Path(
            instance_name=self.instance_name,
            method_name=self.method_name,
            category_name=self.category_name,
            output_type=self.output_type,
            kwargs=self.kwargs,
            _title=title
        )

    @staticmethod
    def _sanitise(value: str) -> str:
        original = str(value).strip()

        value = re.sub(r"[^\w.-]+", "-", original)

        if value != original:
            warnings.warn(
                f"Value {original!r} was sanitised to {value!r}",
                UserWarning,
                stacklevel=2,
            )

        value = value.replace("_", "-").strip("-")

        return value

    @property
    def fmt_kwargs(self) -> str:
        def fmt(v: Any) -> str:
            if is_dataclass(v):
                return v.__class__.__name__  # type: ignore
            match v:
                case bool():
                    return str(v)
                case int():
                    return str(v)
                case float():
                    return f"{v:.4f}"
                case str():
                    if len(v) > 60:
                        return f"{v[:25]}...{v[-25:]}"
                    else:
                        return v
                case list() | np.ndarray():
                    if len(str(v)) > 60 and len(v) > 4:
                        return f"[{fmt(v[0])}, {fmt(v[1])}, ..., {fmt(v[-2])}, {fmt(v[-1])}]"
                    else:
                        return str(v)
                case _:
                    if len(str(v)) > 60:
                        return f"{str(v)[:25]}...{str(v)[-25:]}"
                    else:
                        return str(v)

        def fmt_pair(k: str, v: Any) -> str:
            return f"{k}={fmt(v)}"

        kwargs = "-".join(fmt_pair(k, v) for k, v in self.kwargs.items())
        return self._sanitise(kwargs)

    @property
    def root(self) -> pathlib.Path:
        return pathlib.Path() / "outputs"

    def filename(self, ext: str = "") -> str:
        if len(ext) > 0 and not ext.startswith("."):
            raise ValueError(f"{ext} must start with '.'")

        fmt_kwargs = self.fmt_kwargs
        kwargs_str = f"-{fmt_kwargs}" if fmt_kwargs else ""
        return self.method_name + kwargs_str + ext

    @property
    def to_stem(self) -> pathlib.Path:
        if self._title:
            instance_name = self.title
        else:
            instance_name = self.instance_name

        return self.root / self.category_name / instance_name / self.output_type / self.filename()

    @property
    def parent(self) -> pathlib.Path:
        return self.to_stem.parent

    def to_path(self, ext: str) -> pathlib.Path:
        return self.parent / self.filename(ext)

    @property
    def title(self) -> str:
        if self._title:
            return self._title
        kwargs = " ".join(f"{k}={v}" for k, v in self.kwargs.items())
        kwargs_str = f" ({kwargs})" if kwargs else ""
        return f"{self.instance_name}.{self.method_name}{kwargs_str}"

    @title.setter
    def title(self, title: str) -> None:
        self._title = self._sanitise(title)
