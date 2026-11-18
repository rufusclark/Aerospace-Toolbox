import pathlib

__all__ = (
    "is_notebook",
    "embed_img",
    "hide_warnings",
    "enable_crash_logging",
    "install_requirements",
    "header",
    "table"
)


def is_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except:
        return False


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


def header(title, subtitle=""):
    from IPython.display import display, HTML  # type: ignore
    from html import escape

    display(HTML(f"""
    <div style="
        color: var(--vscode-editor-foreground, var(--jp-ui-font-color1));
        border-bottom: 1px solid var(--vscode-panel-border, var(--jp-border-color1));
        padding-bottom: 8px;
        margin-bottom: 15px;
    ">
        <h2 style="margin:0">{escape(title)}</h2>
        <div style="opacity:0.65; margin-top:4px">{escape(subtitle)}</div>
    </div>
    """))


def table(df):
    import itables

    itables.init_notebook_mode()

    itables.show(
        df,
        scrollX=True,
        paging=False,
        columnDefs=[
            {"className": "dt-left", "targets": [0]},
            {"className": "dt-right", "targets": "_all"}
        ]
    )
