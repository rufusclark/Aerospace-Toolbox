from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.path as mpath

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

import numpy as np

from dotenv import load_dotenv
from os import environ

from typing import Literal

from model.util.output.plot import colour_palette

MARKERS = ["+", "x", "p", "D", "s", "v", "1", "2", "3", "4"]


class MapPlots:
    def __init__(
        self,
        ax: Axes,
        projection: ccrs.Projection,
        width: float
    ):
        self._update_ax(ax, projection)
        self.set_width(width)

    @classmethod
    def from_local_projection(cls, ax: Axes, lat: float, lon: float, width: float) -> MapPlots:
        return MapPlots(
            ax=ax,
            projection=cls.local_crs(lat=lat, lon=lon),
            width=width
        )

    @property
    def fig(self) -> Figure:
        return self.ax.figure  # type: ignore

    def _update_ax(self, ax: Axes, projection: ccrs.Projection):
        """update axes with projection and update ax and projection state"""
        fig = ax.figure
        position = ax.get_position()
        ax.remove()

        self.ax = fig.add_axes(
            position.bounds,
            projection=projection
        )
        self.projection = projection

    @classmethod
    def local_crs(cls, lat: float, lon: float) -> ccrs.Projection:
        return ccrs.TransverseMercator(
            central_latitude=lat,
            central_longitude=lon
        )

    def set_width(self, width):
        """set width and fix aspect ratio"""
        self.width = width
        self.ax.set_extent(  # type: ignore
            [-width/2, width/2, -width/2, width/2],
            crs=self.projection
        )
        self.ax.set_aspect("equal", adjustable="box")

    def add_satellite_imagery(self, zoom: int | None = None):
        env_path = "./keys.env"
        load_dotenv(env_path)
        MAPBOX_ACCESS_TOKEN = environ.get("MAPBOX_ACCESS_TOKEN")
        if not MAPBOX_ACCESS_TOKEN:
            raise ValueError(
                f"MAPBOX_ACCESS_TOKEN does not exist at '{env_path}'.\n"
                f"To use satellite imagery you must supply a MAPBOX access token, available here, https://www.mapbox.com/. Then save the access token in the {env_path} file using the format `MAPBOX_ACCESS_TOKEN=ACCESS TOKEN`.\n"
                f"To not use satellite imagery please set `satellite_images=False`."
            )

        tiler = cimgt.MapboxTiles(
            access_token=MAPBOX_ACCESS_TOKEN,
            map_id="satellite-v9",
            cache=True
        )

        if isinstance(zoom, int):
            zoom = min(zoom, 22)
        else:
            zoom = min(int(15 + np.log2(4000 / self.width)), 22)

        self.ax.add_image(tiler, zoom)  # type: ignore

    def add_point(self, x: float, y: float, label: str, colour: str):
        self.ax.scatter(
            x, y,
            marker="*",
            s=200,
            edgecolors="black",
            linewidths=1.5,
            color=colour,
            transform=self.projection,
            label=label,
            zorder=10
        )

    def add_launch_point(self, x: float = 0, y: float = 0, *, label="launch"):
        self.add_point(x, y, label, colour_palette[1])

    def add_mean_point(self, x: float, y: float, *, label="mean landing"):
        self.add_point(x, y, label, colour_palette[2])

    def add_ideal_point(self, x: float, y: float, *, label="ideal landing"):
        self.add_point(x, y, label, colour_palette[3])

    def add_landing_points(self, x_points, y_points, *, labels: list[str] | None = None, label="landing", marker: str = "x", colour=colour_palette[0]):
        if labels is not None:
            x_points = np.asarray(x_points)
            y_points = np.asarray(y_points)
            np_labels = np.array(labels, dtype=str)

            set_labels = set(labels)
            for set_label, colour in zip(set_labels, colour_palette):
                mask = (np_labels == set_label)
                self.add_landing_points(
                    x_points=x_points[mask],
                    y_points=y_points[mask],
                    label=set_label,
                    colour=colour,
                )

        else:
            self.ax.scatter(
                x_points,
                y_points,
                marker=marker,
                s=50,
                alpha=0.8,
                color=colour,
                transform=self.projection,
                label=label
            )

    def add_landing_points_with_colourmap(self, x_points, y_points, c, cbar_label: str, *, scatter_label: str = "landing", cmap: str = "viridis", log: bool = False):
        c_min = min(c)
        c_max = max(c)

        sc = self.ax.scatter(
            x_points,
            y_points,
            c=c,
            marker="x",
            s=50,
            alpha=0.8,
            cmap=cmap,
            norm=LogNorm(vmin=c_min, vmax=c_max) if log else None,
            transform=self.projection,
            label=scatter_label
        )

        cbar = self.fig.colorbar(sc)
        cbar.set_label(cbar_label)
        cbar.set_ticks([c_min, c_max])
        cbar.set_ticklabels([f"{c_min:.2g}", f"{c_max:.2g}"])

    def add_circle_with_label(self, x_center: float, y_center: float, radius: float, label: str = "", color: str = "white"):
        circle = mpatches.Circle(
            (x_center, y_center),
            radius,
            edgecolor=color,
            facecolor="none",
            linestyle="--",
            linewidth=1,
            transform=self.projection,
            zorder=6
        )
        self.ax.add_patch(circle)

        if label:
            self.ax.text(
                x_center, y_center + radius,
                label,
                color=color,
                fontsize=9,
                ha="center",
                va="bottom",
                transform=self.projection,
                zorder=7
            )

    def add_range_rings(self, radii: list[float], x_center: float = 0, y_center: float = 0, unit: Literal["km", "m", "nm"] | None = None):
        if unit is None:
            if max(radii) > 1e3:
                unit = "km"
            else:
                unit = "m"

        for radius in radii:
            if unit == "km":
                label = f"{radius/1e3:.1f} km"
            elif unit == "m":
                label = f"{radius:.0f} m"
            elif unit == "nm":
                label = f"{radius/1852:.1f} nm"
            else:
                raise ValueError(f"Unexpected unit: {unit}")

            self.add_circle_with_label(
                x_center=x_center,
                y_center=y_center,
                radius=radius,
                label=label,
                color="white"
            )

    def compute_ellipses(self, x_points, y_points, probabilities: list[float]):
        """Compute dispersion ellipses for specified confidence probabilities.

        Args:
            probabilities: List of enclosed probabilities (0-1).

        Returns:
            mean: (x, y) tuple of mean position
            angle: orientation angle in degrees
            width: list of ellipse widths
            height: list of ellipse heights
            area: list of ellipse areas
        """
        if any(not 0 < p < 1 for p in probabilities):
            raise ValueError("Probabilities must be between 0 and 1.")

        data = np.vstack((x_points, y_points))

        # Centre point
        mean = np.mean(data, axis=1)

        # Covariance matrix
        cov = np.cov(x_points, y_points)

        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort largest to smallest
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Orientation angle [deg]
        angle = np.degrees(
            np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        )

        n = len(probabilities)
        width = np.zeros(n)
        height = np.zeros(n)
        area = np.zeros(n)

        for i, probability in enumerate(probabilities):

            # Convert confidence probability to 2D Gaussian
            # standard deviation scaling factor
            n_sigma = np.sqrt(-2 * np.log(1 - probability))

            semi_major = n_sigma * np.sqrt(eigvals[0])
            semi_minor = n_sigma * np.sqrt(eigvals[1])

            width[i] = 2 * semi_major
            height[i] = 2 * semi_minor
            area[i] = np.pi * semi_major * semi_minor

        return mean, angle, width, height, area

    def add_probability_ellipses(self, x_points, y_points, probabilities: list[float]):
        """Plot dispersion ellipses for specified confidence probabilities.

        Args:
            ax: Matplotlib axes
            probabilities: List of enclosed probabilities (0-1).
        """
        mean, angle, widths, heights, _ = self.compute_ellipses(
            x_points, y_points, probabilities)

        self.add_mean_point(*mean)

        for probability, width, height in zip(
            probabilities, widths, heights
        ):
            ellipse = mpatches.Ellipse(
                xy=mean,  # type: ignore
                width=width,
                height=height,
                angle=angle,
                label=rf"${probability * 100:.1f}\%$",
                transform=self.projection,
                alpha=0.5,
                zorder=9,
                edgecolor="red",
                facecolor="none",
                lw=2,
            )

            self.ax.add_patch(ellipse)

    def add_wing_arrow(self, speed: float, direction: float):
        # arrow position (axes coordinates)
        x_pos = 0.93
        y_pos = 0.10

        arrow_length = 0.06

        dx = arrow_length * np.sin(np.radians(direction))
        dy = arrow_length * np.cos(np.radians(direction))

        # draw arrow
        self.ax.annotate(
            "",
            xy=(x_pos + dx/2, y_pos + dy/2),
            xytext=(x_pos - dx/2, y_pos - dy/2),
            xycoords="axes fraction",
            arrowprops=dict(
                facecolor="white",
                edgecolor="black",
                width=3,
                headwidth=8
            ),
            zorder=10,
        )

        # add airspeed below arrow
        self.ax.text(
            x_pos,
            y_pos - 0.04,
            f"{speed:.1f} m/s",
            transform=self.ax.transAxes,
            color="white",
            fontweight="bold",
            ha="center",
            va="top",
            fontsize=10,
            zorder=10,
            path_effects=[
                pe.Stroke(linewidth=2.5, foreground="black"),
                pe.Stroke(linewidth=1, foreground="white"),
                pe.Normal()
            ],
        )

    def post_amble(self) -> Axes:
        """post amble and return value"""
        self.ax.legend()
        return self.ax
