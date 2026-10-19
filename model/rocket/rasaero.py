from dataclasses import dataclass, fields, is_dataclass
from pywinauto import Desktop, Application, findwindows
import xml.etree.ElementTree as ET
from ctypes import wintypes
from typing import Literal
from time import sleep
import warnings
import pathlib
import ctypes
import shutil

from model.util.path import pathManager


def m_to_ft(x): return x * 3.28084
def m_to_in(x): return x * 39.3701
def kg_to_lb(x): return x * 2.205
def C_to_F(x): return (x * 9/5) + 32
def K_to_C(x): return x - 273.15
def K_to_F(x): return C_to_F(K_to_C(x))
def ms_to_mph(x): return x * 2.23694
def Pa_to_inhg(x): return x / 3386


RAIL_BUTTON_DISTANCE = 25.4 / 1000


class XMLDataclass:
    @property
    def root_name(self) -> str:
        return self.__class__.__name__.strip("_")

    def to_xml(self) -> ET.Element:
        if not is_dataclass(self):
            raise TypeError("self must be a dataclass instance")

        root = ET.Element(self.root_name)

        for field in fields(self):
            value = getattr(self, field.name)

            if value is None:
                continue

            name = field.name.strip("_")

            if isinstance(value, XMLDataclass):
                root.append(value.to_xml())

            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, XMLDataclass):
                        root.append(item.to_xml())
                    else:
                        element = ET.SubElement(root, name)
                        element.text = str(item)

            else:
                element = ET.SubElement(root, name)
                element.text = str(value)

        return root

    def save_xml(self, path: pathlib.Path | str) -> None:
        """filenames should use the .CDX1 suffix"""
        if isinstance(path, str):
            path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix != ".CDX1":
            raise ValueError(f"path ({path}) should use the .CDX1 suffix")

        root = self.to_xml()

        # Pretty-print the XML tree
        ET.indent(root, space="  ")

        tree = ET.ElementTree(root)
        tree.write(
            path,
            encoding="utf-8",
            xml_declaration=False,
        )

    def pretty_xml(self) -> str:
        root = self.to_xml()
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")


NoseConeShapes = Literal["Von Karman Ogive", "Tangent Ogive"]


@dataclass
class _NoseCone(XMLDataclass):
    Length: float
    """[in]"""
    Diameter: float
    """[in]"""
    Shape: NoseConeShapes
    BluntRadius: float
    """[in]"""
    Location: float = 0
    """[in]"""
    Color: str = "Black"
    PartType: str = "NoseCone"


FinAirfoilSections = Literal["Subsonic NACA", "Square"]


@dataclass
class _Fin(XMLDataclass):
    Count: int
    Chord: float
    """[in]"""
    Span: float
    """[in]"""
    SweepDistance: float
    """[in]"""
    TipChord: float
    """[in]"""
    Thickness: float
    """[in]"""
    LERadius: float
    """[in]"""
    Location: float
    """[in]"""
    AirfoilSection: FinAirfoilSections
    FX1 = 0
    # TODO: Determine what this is
    FX3 = 0
    # TODO: Determine what this is


@dataclass
class _BodyTube(XMLDataclass):
    Length: float
    """[in]"""
    Diameter: float
    """[in]"""
    Location: float
    """[in]"""
    LaunchLugDiameter: float = 0
    """[in]"""
    LaunchLugLength: float = 0
    """[in]"""
    RailGuideDiameter: float = 0
    """[in]"""
    RailGuideHeight: float = 0
    """[in]"""
    LaunchShoeArea: float = 0
    """[in]"""
    BoattailLength: float = 0
    """[in]"""
    BoattailRearDiameter: float = 0
    """[in]"""
    BoattailOffset: float = 0
    """[in]"""
    Overhang: float = 0
    """[in]"""
    Color: str = "Black"
    PartType: str = "BodyTube"
    Fin: _Fin | None = None
    """Only include if the body tube has a fin"""


@dataclass
class _Transition(XMLDataclass):
    Length: float
    """[in]"""
    Diameter: float
    """[in]"""
    Location: float
    """[in]"""
    RearDiameter: float
    """[in]"""
    Color: str = "Black"
    PartType: str = "Transition"


@dataclass
class _BoatTail(XMLDataclass):
    Length: float
    """[in]"""
    Diameter: float
    """[in]"""
    Location: float
    """[in]"""
    RearDiameter: float
    """[in]"""
    Color: str = "Black"
    PartType: str = "BoatTail"


Surfaces = Literal["Polished", "Smooth (Zero Roughness)"]


@dataclass
class _RocketDesign(XMLDataclass):
    NoseCone: _NoseCone
    BodyTube: list[_BodyTube]
    Transition: _Transition | None
    BoatTail: _BoatTail | None
    Surface: Surfaces
    CP: float
    ModifiedBarrowman: bool
    Turbulence: bool
    SustainerNozzle: float
    FileVersion: int = 2
    """Motor Nozzle Diameter [in]"""
    Booster1Nozzle: float = 0
    """Booster 1 Nozzle Diameter [in]"""
    Booster2Nozzle: float = 0
    """Booster 2 Nozzle Diameter [in]"""
    UseBooster1: bool = False
    UseBooster2: bool = False
    Comments: str = ""


@dataclass
class _LaunchSite(XMLDataclass):
    Altitude: float
    """[ft]"""
    Pressure: float
    """[in-hg]"""
    RodAngle: float
    """angle from vertical [deg]"""
    RodLength: float
    """[ft]"""
    Temperature: float
    """[deg F]"""
    WindSpeed: float
    """[mph]"""

    def to_LSD(self) -> str:
        """Get the launch site data for RAS Aero's SOAP interface. Returned as a string.

        This should be saved as 'LSD.xml' in RAS Aero's working directory"""
        return (
            f'''<?xml version="1.0" encoding="utf-8"?>
<SOAP-ENV:Envelope
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xmlns:xsd="http://www.w3.org/2001/XMLSchema"
    xmlns:SOAP-ENC="http://schemas.xmlsoap.org/soap/encoding/"
    xmlns:SOAP-ENV="http://schemas.xmlsoap.org/soap/envelope/"
    xmlns:clr="http://schemas.microsoft.com/soap/encoding/clr/1.0"
    SOAP-ENV:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">
    <SOAP-ENV:Body>
        <a1:LaunchSiteClass
            id="ref-1"
            xmlns:a1="http://schemas.microsoft.com/clr/nsassem/RASAeroII/RASAero%20II%2C%20Version%3D1.0.2.0%2C%20Culture%3Dneutral%2C%20PublicKeyToken%3Dnull">
            <Altitude>{self.Altitude}</Altitude>
            <Temperature>{self.Temperature}</Temperature>
            <Pressure>{self.Pressure}</Pressure>
            <RodLength>{self.RodLength}</RodLength>
            <RodAngle>{self.RodAngle}</RodAngle>
            <WindSpeed>{self.WindSpeed}</WindSpeed>
        </a1:LaunchSiteClass>
    </SOAP-ENV:Body>
</SOAP-ENV:Envelope>'''
        )

    def save_LSD(self, path: pathlib.Path | str) -> None:
        """file should use the LSD.xml filename"""
        if isinstance(path, str):
            path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(self.to_LSD())


@dataclass
class _Recovery(XMLDataclass):
    Altitude1: float
    """Deployment 1 Altitude [ft]"""
    Size1: float
    """Parachute 1 Diameter [in]"""
    EventType1: Literal["Altitude", "Apogee"]
    CD1: float
    """Parachute 1 Drag Coefficient"""
    Event1: bool = True
    DeviceType1: Literal["Parachute"] = "Parachute"

    Altitude2: float = 0
    """Deployment 2 Altitude [ft]"""
    Size2: float = 0
    """Parachute 2 Diameter [in]"""
    EventType2: Literal["Altitude", "Apogee"] = "Altitude"
    CD2: float = 0
    """Parachute 2 Drag Coefficient"""
    Event2: bool = False
    DeviceType2: Literal["Parachute"] = "Parachute"


@dataclass
class MachAltPair():
    Mach: float
    Altitude: float
    """[ft]"""

    def __str__(self) -> str:
        return f"{self.Mach}, {self.Altitude}"


@dataclass
class _MachAlt(XMLDataclass):
    """Can't duplicate Mach's among any pairs"""
    Item: list[MachAltPair]


@dataclass
class _Simulation(XMLDataclass):
    SustainerEngine: str
    """name of rocket engine with 'rasp.eng' file
    
    The name is the first value above the thrust curve data.
    
    Ensure the rasp file content for this motor is appended in the `rasp.eng` file with `Documents/RAS Aero II`"""
    SustainerLaunchWt: float
    """rocket launch mass including (wet) motor mass [lb]"""
    SustainerNozzleDiameter: float
    """Rocket motor nozzle diameter [in]"""
    # TODO: Where can I find good values for this?
    SustainerCG: float
    """Rocket CG Location from nose cone [in]"""
    # TODO: Is this calculated by RAS Aero II?
    SustainerIgnitionDelay: float = 0
    """Rocket motor ignition delay [s]"""

    # Booster 1
    Booster1LaunchWt: float = 0
    """Booster 1 launch mass including (wet) motor mass [lb]"""
    Booster1SeparationDelay: float = 0
    """Booster 1 separation delay [s]"""
    Booster1IgnitionDelay: float = 0
    """Booster 1 motor ignition delay [s]"""
    Booster1CG: float = 0
    """Booster 1 CG location [in]"""
    Booster1NozzleDiameter: float = 0
    """Booster 1 motor nozzle diameter [in]"""
    IncludeBooster1: bool = False

    # Booster 2
    Booster2LaunchWt: float = 0
    """Booster 2 launch mass including (wet) motor mass [lb]"""
    Booster2Delay: float = 0
    """Booster 2 delay [s]"""
    Booster2CG: float = 0
    """Booster 2 CG location [in]"""
    Booster2NozzleDiameter: float = 0
    """Booster 2 motor nozzle diameter"""
    IncludeBooster2: bool = False

    # Outputs
    FlightTime: float = 0
    """[s]"""
    TimetoApogee: float = 0
    """[s]"""
    MaxAltitude: float = 0
    """[ft]"""
    MaxVelocity: float = 0
    """[ft/s]"""
    OptimumWt: float = 0
    """[lb]"""
    OptimumMaxAlt: float = 0
    """[ft]"""


@dataclass
class _SimulationList(XMLDataclass):
    Simulation: list[_Simulation]


@dataclass
class RASAeroDocument(XMLDataclass):
    RocketDesign: _RocketDesign
    LaunchSite: _LaunchSite
    Recovery: _Recovery | None
    MachAlt: _MachAlt
    SimulationList: _SimulationList


class RasAeroII:

    def __init__(
        self,
        name: str,
        rasaerodoc: RASAeroDocument,
        working_path: str | None = None
    ) -> None:
        self.name = name
        self.rasaerodoc = rasaerodoc
        self._working_path = None if working_path is None else pathlib.Path(
            working_path)

    # @property
    # def exe_path(self) -> str:
    #     return "C:/Program Files (x86)/RASAero II/RASAero II.exe"

    @staticmethod
    def __get_documents_path() -> pathlib.Path:
        """Return the user's Windows Documents folder."""
        CSIDL_PERSONAL = 5  # Documents

        path_buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
        result = ctypes.windll.shell32.SHGetFolderPathW(
            None,
            CSIDL_PERSONAL,
            None,
            0,
            path_buf,
        )

        if result != 0:
            raise OSError(f"SHGetFolderPathW failed: {result}")

        return pathlib.Path(path_buf.value)

    @property
    def working_path(self) -> pathlib.Path:
        if self._working_path is None:
            self._working_path = self.__get_documents_path() / "RASAero II"
        return self._working_path

    @property
    def default_aero_plots_path(self) -> pathlib.Path:
        return self.working_path / "CD Test.csv"

    @property
    def data_path(self) -> pathlib.Path:
        path = pathlib.Path("data/rocket") / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def extract_aero_plots(self):
        filename = "rocket.CDX1"

        # save rasaero file
        rasaero_data_path = self.data_path / filename
        self.rasaerodoc.save_xml(rasaero_data_path)

        # copy rasaero to working directory
        shutil.copy2(
            rasaero_data_path,
            self.working_path / filename
        )

        print(f"Saved RASAero II to {rasaero_data_path}")

        # create LSD.xml environment file in working directory
        self.rasaerodoc.LaunchSite.save_LSD(self.working_path / "LSD.xml")
        print(f"Saved LSD in {self.working_path / "LSD.xml"}")

        # raise an error if RASAero II is already open
        windows = findwindows.find_windows(title="RASAero II")
        if windows:
            raise RuntimeError(
                "Please ensure no instances of RASAero II are running. Sometimes other applications with windows called RASAero II may also cause issues too. Please close these too."
            )

        # Start RASAero
        app = Application(backend="uia").start(str(pathManager.RASAero))

        # Wait for the window to be open
        main = app.window(auto_id="Main", control_type="Window")
        main.wait("visible", timeout=10)

        # shortcut to open files
        main.set_focus()
        main.type_keys("^o")

        # write file to open dialog "File name:" box
        file_name_box = main.child_window(
            auto_id="1148", control_type="Edit")
        file_name_box.wait("visible", timeout=10)
        file_name_box.set_edit_text(filename)

        # press open button
        file_name_box.set_focus()
        file_name_box.type_keys("{ENTER}")

        # save screenshot
        main.wait("ready", timeout=10)
        screenshot = main.capture_as_image()
        screenshot.save(self.data_path / "rocket.png")
        print(f"Saved RASAero screenshot to {self.data_path / "rocket.png"}")

        # open Aero Plots
        aero_button = main.child_window(
            title="Aero Plots", control_type="Button")
        aero_button.wait("enabled", timeout=10)
        aero_button.click_input()

        # wait for aero plots to open
        aero_plots = Desktop(backend="win32").window(title="Aero Plots")
        aero_plots.wait("visible", timeout=10)

        # open export to csv dialog
        aero_plots.set_focus()
        aero_plots.type_keys("%fet")

        if self.default_aero_plots_path.exists():
            warnings.warn(
                f"Deleting existing AeroPlot output: {self.default_aero_plots_path}"
            )
            self.default_aero_plots_path.unlink()

        # save with default name "CD TEST.csv"
        save_dialog = Desktop(backend="win32").window(title="Save As")
        save_dialog.wait("visible", timeout=10)

        save_dialog.set_focus()
        save_dialog.type_keys("{ENTER}")

        # wait until Aero Plots data file has been created and no longer write locked
        aero_data_path = self.working_path / "CD Test.csv"
        while not aero_data_path.exists():
            sleep(0.05)
        while True:
            try:
                with open(aero_data_path, "a"):
                    break
            except PermissionError:
                sleep(0.05)

        # move to data path
        (self.data_path / "aeroData.csv").unlink(missing_ok=True)
        aero_data_path.rename(self.data_path / "aeroData.csv")
        print(f"Saved AeroPlot data to {self.data_path / "aeroData.csv"}")

        # close Aero Plots
        aero_plots.close()

        # close RAS Aero
        main.close()


# TODO: Investigate "Mach-Alt"
# TODO: Convert from my rocket to RASAeroII document
# TODO: Add motor to RAS Aero II
#   TODO: Copy existing RAS Aero II rasp.eng file
#   TODO: Create new rasp.eng file with thrust curve appended to end
#   TODO: Set sustainer name to motor name
# TODO: Do "Run Test" and then process output into csv (via pandas)

# TODO: Create power_on() and power_off() drag curve as a f(M, alpha, Re)

"""

Workflow

1. Generate RAS Aero II File
2. Add motor
3. Set (Mach-Alt) if not already done
4. Set motor if not already done
5. Run Test
6. Process output from RunTest into CSV

"""
