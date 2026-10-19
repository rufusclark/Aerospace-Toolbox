import pathlib
from typing import Literal

PathType = Literal["xFoil", "RASAero"]

class PathManagerError(Exception):
    def __init__(self, msg: str, path_type: PathType) -> None:
        msg_cores = {
            "xFoil": (
                "This is necessary to simulate 2D airfoil behaviour.\n"
                "This can be installed from, https://web.mit.edu/drela/Public/web/xfoil/.\n"
            ),
            "RASAero": (
                "This is necessary to calculate aerodynamic performance for rockets.\n"
                "This can be installed from, https://www.rasaero.com/.\n"
            )
        }

        fmt_msg = (
            f"{msg}\n\n"
            f"Please ensure {path_type} is installed.\n"
            f"{msg_cores.get(path_type, "")}"
            f"If {path_type} is installed update the path,\n"
            f"\tpathManager.{path_type} = 'C/new/exe/path'"
        )
        
        super().__init__(fmt_msg)

class __PathManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.__xFoil = pathlib.Path("C:/Program Files/XFOIL/xfoil.exe")
        self.__RASAero = pathlib.Path("C:/Program Files (x86)/RASAero II/RASAero II.exe")

        self._initialized = True

    def __repr__(self) -> str:
        paths = [f"{key.split("__")[1]}='{val}'" for key, val in vars(self).items() if key != "_initialized"]
        return f"<PathManager {' '.join(paths)}>"

    def __validate(self, path: pathlib.Path, path_type: PathType, allowed_extensions: tuple[str, ...] = (".exe",)):
        if path.suffix not in allowed_extensions:
            raise PathManagerError(f"{path} does not end in one of the allowed extensions ({allowed_extensions})", path_type)
        
        if not path.exists():
            raise PathManagerError(f"{path_type} not found at: {path}", path_type)

    @property
    def xFoil(self) -> pathlib.Path:
        self.__validate(self.__xFoil, "xFoil")
        return self.__xFoil

    @xFoil.setter
    def xFoil(self, path: str | pathlib.Path) -> None:
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        self.__validate(path, "xFoil")
        self.__xFoil = path

    @property
    def RASAero(self) -> pathlib.Path:
        self.__validate(self.__RASAero, "RASAero")
        return self.__RASAero

    @RASAero.setter
    def RASAero(self, path: str | pathlib.Path) -> None:
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        self.__validate(path, "RASAero")
        self.__RASAero = path

pathManager = __PathManager()

if __name__=="__main__":
    print(pathManager)

    pathManager.RASAero = "C:/Program Files/XFOIL/xfoil.exe1"

    print(pathManager)

    print(__PathManager())
