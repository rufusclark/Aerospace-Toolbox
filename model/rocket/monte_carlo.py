from model.rocket.rocket import FlightGenerator
from model.rocket.flight import Flight
from model.util.output import *
from model.util.progress import ProgressBar

from typing import Literal

import numpy as np


class MonteCarlo:
    category = "monte-carlo"

    def __init__(
            self,
            flight_generator: FlightGenerator,
            *,
            name: str | None = None,
            n: int = 100,
            verbose: bool = False,
            max_time_step: float | None = None,
            max_time: int | None = None
    ) -> None:
        self.plots = MonteCarloPlots(self)
        self.info = MonteCarloInfo(self)

        self.flight_generator = flight_generator

        self.name = name or flight_generator.name
        self.n = n
        self.verbose = verbose

        self.flights: list[Flight] = [None for i in range(n)]  # type: ignore

        progress = ProgressBar(n, name="monte-carlo")

        for i in range(n):
            last_flight = Flight(
                flight_state=flight_generator.sample(),
                name=name,
                verbose=verbose,
                terminate_on_apogee=False,
                max_time_step=max_time_step,
                max_time=max_time
            )
            self.flights[i] = last_flight
            progress.update(
                i+1,
                msg=" - ".join(
                    [
                        f"apogee: {last_flight.apogee:.0f} m",
                        f"flight time: {last_flight.flight_duration:.1f} s",
                        f"max speed: Mach {last_flight.max_mach:.2f}"
                    ],
                )
            )

    @property
    def launch_point(self) -> tuple[float, float]:
        """launch point [latitude, longitude] in decimal degrees"""
        return (
            self.flight_generator.env.latitude.ideal(),
            self.flight_generator.env.longitude.ideal()
        )


class MonteCarloPlots(PlotBase[MonteCarlo]):

    @scientific_plot(auto_limits=None, outside_legend="lower", outside_legend_cols=4)
    def landing_dispersion(
        self,
        ax: Axes,
        satellite_images: bool = True,
        impact_energy: bool = True,
        width: float | None = None,
        range_rings_radii: list[float] | None = None,
        range_rings_unit: Literal["m", "km", "nm"] | None = None,
        probability_ellipses: list[float] | None = None,
    ) -> Axes:
        from model.util.output.map_plot import MapPlots

        model = self.model
        y_points = np.array(
            [flight.y_landing_position for flight in model.flights])
        x_points = np.array(
            [flight.x_landing_position for flight in model.flights])
        impact_energies = np.array(
            [flight.impact_energy for flight in model.flights])
        labels = []
        for flight in model.flights:
            if flight.flight_state.options.ballistic is True:
                label = "ballistic"
            elif flight.flight_state.options.all_chutes_at_apogee is True:
                label = "main at apogee"
            else:
                label = "nominal"
            labels.append(label)

        d_max = float(np.max(np.abs([*y_points, *x_points])))

        if width is None:
            width = 2 * max(d_max * 1.1, 110)
        elif width < d_max * 2:
            import warnings
            warnings.warn(
                f"manually set width ({width}m) clips landing points (max={d_max}m)"
            )

        map_plots = MapPlots.from_local_projection(
            ax, *model.launch_point, width=width)

        if satellite_images:
            map_plots.add_satellite_imagery()

        map_plots.add_launch_point()
        if impact_energy:
            map_plots.add_landing_points_with_colourmap(
                x_points, y_points, impact_energies,
                cbar_label="Impact Energy [J]",
                cmap="cividis",
            )
        else:
            map_plots.add_landing_points(x_points, y_points, labels=labels)

        if range_rings_radii is not None:
            map_plots.add_range_rings(
                radii=range_rings_radii,
                unit=range_rings_unit
            )

        ideal_flight = Flight(model.flight_generator.ideal())
        map_plots.add_ideal_point(
            x=ideal_flight.x_landing_position,
            y=ideal_flight.y_landing_position
        )

        if probability_ellipses is not None:
            map_plots.add_probability_ellipses(
                x_points=x_points,
                y_points=y_points,
                probabilities=probability_ellipses
            )

        assert model.flight_generator.env.atmospheric_model is not None
        wind_direction, wind_speed = (
            model.flight_generator.env.atmospheric_model
            .ideal().wind_direction_speed
        )
        map_plots.add_wing_arrow(
            direction=wind_direction,
            speed=wind_speed
        )

        return map_plots.post_amble()

    @scientific_plot(auto_limits=None)
    def apogee_distribution(self, ax: Axes, percentiles: list[float], bins: int | None = None):
        model = self.model

        apogees = np.array([flight.apogee for flight in model.flights]) / 1e3

        ax.hist(apogees, bins=bins)

        mean = np.mean(apogees)
        ax.axvline(
            mean, color=colour_palette[1],
            label=f"mean ({mean:.2f} km)"
        )

        ideal_apogee = Flight(model.flight_generator.ideal()).apogee / 1e3
        ax.axvline(
            ideal_apogee, color=colour_palette[2],
            label=f"ideal ({ideal_apogee:.2f} km)"
        )

        for i, percentile in enumerate(percentiles):
            lower, upper = np.percentile(
                apogees, [100 - percentile, percentile])
            label = f"{percentile:.1f}%"
            colour = colour_palette[i+3]
            ax.axvline(lower, color=colour, label=label)
            ax.axvline(upper, color=colour)

        ax.tick_params(axis="y", left=False, labelleft=False)

        ax.set_xlabel("Apogee [km]")

        ax.set_ylim(0, None)

        ax.legend()


class MonteCarloInfo(InfoBase[MonteCarlo]):
    pass
