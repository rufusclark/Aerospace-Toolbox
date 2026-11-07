import pathlib

__all__ = (
    "embed_img",
    "hide_warnings",
    "enable_crash_logging",
    "install_requirements"
)


def embed_img(path: str | pathlib.Path, width: int = 600) -> None:
    from IPython.display import HTML, display  # type: ignore
    import base64

    if not isinstance(path, pathlib.Path):
        path = pathlib.Path(path)

    data = base64.b64encode(path.read_bytes()).decode("utf-8")

    ext = path.suffix.lower()
    match ext:
        case ".png":
            mime = "image/png"
        case ".jpeg" | ".jpg":
            mime = "image/jpeg"
        case _:
            raise ValueError(f"Extension: {ext} not yet supported")

    display(HTML(
        f'<img src="data:{mime};base64,{data}" '
        f'width="{width}">'
    ))


def hide_warnings() -> None:
    import numpy as np
    np.seterr("ignore")
    import warnings
    warnings.simplefilter("ignore")

    print("Warnings hidden")


def enable_crash_logging(path: str = "kernel-fault.log") -> None:
    import faulthandler

    _fault_log = open(path, "a", buffering=1)
    faulthandler.enable(file=_fault_log, all_threads=True)

    print("Crash logging enabled")


def install_requirements() -> None:
    import subprocess
    import sys

    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "./requirements.txt"
    ])

    print("Requirements installed")
