from __future__ import annotations
from typing import Any, Literal, TypeVar, ParamSpec, Callable, Concatenate, Generic, cast
from dataclasses import dataclass, field
from functools import wraps

from .base import OutputType, Path, count, common_suffix

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import scienceplots

plt.style.use("science")

__all__ = [
    "scientific_plot",
    "scientific_compose",
    "Composition",
    "Axes",
    "colour_palette",
]

colour_palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

T = TypeVar("T", bound="OutputType")
P = ParamSpec("P")

AutoLimits = Literal[None, "x", "y", "both", "0.1scale"]
LegendLocation = Literal[None, "lower", "right", "top-inside"]


@dataclass(frozen=True)
class AutoLayout:
    ncols: int = 1

    def dimensions(self, nplots: int) -> tuple[int, int]:
        if nplots <= 0:
            raise ValueError(
                "Cannot create a layout for zero plots."
            )

        if self.ncols <= 0:
            raise ValueError(
                "ncols must be greater than zero."
            )

        ncols = min(self.ncols, nplots)

        nrows = (
            nplots + ncols - 1
        ) // ncols

        return nrows, ncols


def auto_layout(
    *,
    ncols: int = 1,
) -> AutoLayout:
    return AutoLayout(ncols=ncols)


@dataclass(frozen=True)
class Plot(Generic[T, P]):
    """immutable plot specification without rendering (until `draw` is called)."""
    obj: T
    func: Callable[Concatenate[T, Axes, P], Any]

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    auto_limits: AutoLimits = "both"

    xlim: tuple[Any, Any] | None = None
    ylim: tuple[Any, Any] | None = None

    def draw(self, ax: Axes) -> Any:
        """invoke the decorated plotting function."""
        return self.func(
            self.obj,
            ax,
            *self.args,
            **self.kwargs,
        )


class Plots(Generic[T, P]):

    def __init__(
        self,
        *,
        func: Callable[Concatenate[T, Axes, P], Any] | None = None,
        path: Path | None = None,
        auto_limits: AutoLimits = "both",
        figsize: tuple[float, float] = (6, 4),
        outside_legend: LegendLocation = None,
        outside_legend_cols: int = 1,
        ncols: int = 1,
        sharex: bool = False,
        sharey: bool = False,
        compose_func: Callable[Concatenate[T, P], "Composition"] | None = None,
        plots: list[Plot[T, P]] | None = None,
        superposed_plots: list[Plots[T, P]] | None = None,
        obj: T | None = None,
    ):
        self.func = func
        self.path = path

        self.auto_limits = auto_limits
        self.figsize = figsize
        self.outside_legend = outside_legend
        self.outside_legend_cols = outside_legend_cols

        self.ncols = ncols
        self.sharex = sharex
        self.sharey = sharey

        self.compose_func = compose_func
        self.plots = plots
        self.superposed_plots = [] if superposed_plots is None else superposed_plots
        self.obj = obj

        if (compose_func is None and func is None) or (compose_func is not None and func is not None):
            raise ValueError(
                f"Value Error: Either compose_func or func must be provided"
            )

        if func is not None:
            wraps(func)(self)
        elif compose_func is not None:
            wraps(compose_func)(self)

    def __get__(
            self,
            obj: T | None,
            owner: type[T] | None = None
    ) -> Plots[T, P]:
        if obj is None:
            return self

        return Plots(
            func=self.func,
            path=self.path,
            auto_limits=self.auto_limits,  # type: ignore
            figsize=self.figsize,
            outside_legend=self.outside_legend,  # type: ignore
            outside_legend_cols=self.outside_legend_cols,
            ncols=self.ncols,
            sharex=self.sharex,
            sharey=self.sharey,
            compose_func=self.compose_func,
            plots=self.plots,
            obj=obj
        )

    def __repr__(self) -> str:
        if self.path:
            return f"<Plots {self.path.title}>"
        elif self.func:
            return f"<Plots {self.func.__name__}>"
        elif self.compose_func:
            return f"<Plots {self.compose_func.__name__}>"
        else:
            raise ValueError(
                "neither self.path, self.func, self.compose_func are defined."
            )

    def __call__(
            self,
            *args: P.args,
            show: bool = False,  # type: ignore
            save: bool = True,  # type: ignore
            figsize: tuple[float, float] | None = None,  # type: ignore
            xlim: tuple[Any, Any] | None = None,  # type: ignore
            ylim: tuple[Any, Any] | None = None,  # type: ignore
            extensions: tuple[str, ...] = (  # type: ignore
                ".png", ".pdf", ".svg", ".pgf"),
            suppress_prints: bool = False,  # type: ignore
            **kwargs: P.kwargs,
    ) -> Plots[T, P]:

        if self.obj is None:
            raise TypeError(
                "Plots must be accessed through an instance before calling it"
            )
        obj = self.obj

        if self.path is None:
            if self.func:
                method_name = self.func.__name__
            elif self.compose_func:
                method_name = self.compose_func.__name__
            else:
                raise ValueError(
                    f"Value Error: Either compose_func or func must be provided"
                )

            self.path = Path(
                instance_name=obj.model.name,
                method_name=method_name,
                category_name=obj.model.category,
                output_type=obj.output_type,
                kwargs=kwargs
            )

        if self.plots is None:
            if self.compose_func:
                composition = self.compose_func(
                    obj,
                    *args,
                    **kwargs
                )

                self.plots = composition.plots
                self.figsize = composition.figsize or self.figsize
                self.ncols = composition.ncols or self.ncols
                self.sharex = composition.sharex or self.sharex
                self.sharey = composition.sharey or self.sharey
                self.outside_legend = composition.outside_legend or self.outside_legend
                self.outside_legend_cols = composition.outside_legend_cols or self.outside_legend_cols

            elif self.func:
                self.plots = [self.with_args(
                    *args,
                    xlim=xlim,
                    ylim=ylim,
                    **kwargs
                )]

            else:
                raise ValueError(
                    f"Value Error: Either compose_func or func must be provided"
                )

        if save:
            self.save(
                figsize=figsize,
                xlim=xlim,
                ylim=ylim,
                extensions=extensions,
                suppress_prints=suppress_prints
            )

        if show:
            self.show(
                figsize=figsize,
                xlim=xlim,
                ylim=ylim
            )

        count.figures += 1

        return self

    def __add__(self, other: Plots[T, P]) -> Plots[T, P]:
        if not isinstance(other, Plots):
            raise TypeError(
                f"{type(other)} is not supported for this operation"
            )

        assert self.path is not None
        assert other.path is not None

        return Plots(
            func=self.func,
            path=self.path + other.path,
            auto_limits=self.auto_limits,  # type: ignore
            figsize=self.figsize,
            outside_legend=self.outside_legend,  # type: ignore
            outside_legend_cols=self.outside_legend_cols,
            ncols=self.ncols,
            sharex=self.sharex,
            sharey=self.sharey,
            compose_func=self.compose_func,
            plots=self.plots,
            superposed_plots=[other, *self.superposed_plots,
                              *other.superposed_plots],
            obj=self.obj
        )

    def with_args(
            self,
            *args: P.args,
            xlim: tuple[Any, Any] | None = None,  # type: ignore
            ylim: tuple[Any, Any] | None = None,  # type: ignore
            **kwargs: P.kwargs,
    ) -> Plot[T, P]:
        """immutable plot specification without rendering (until draw is called)."""
        if self.func is None or self.obj is None:
            raise ValueError(
                f"Value Error: {self.func is None=} or {self.obj is None=}")

        return Plot(
            obj=self.obj,
            func=self.func,  # type: ignore
            args=args,
            kwargs=kwargs,
            auto_limits=self.auto_limits,  # type: ignore
            xlim=xlim,
            ylim=ylim,
        )

    def save(
        self,
        figsize: tuple[float, float] | None = None,
        xlim: tuple[Any, Any] | None = None,
        ylim: tuple[Any, Any] | None = None,
        extensions: tuple[str, ...] = (".png", ".pdf", ".svg", ".pgf"),
        suppress_prints: bool = False
    ):
        fig = self.fig(
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
        )

        assert self.path

        self.path.parent.mkdir(parents=True, exist_ok=True)
        for ext in extensions:
            fig.savefig(self.path.to_path(ext))

        if not suppress_prints:
            print(f"Saved images to {self.path.to_stem} {extensions}")

        plt.close(fig)

    def show(
        self,
        figsize: tuple[float, float] | None = None,
        xlim: tuple[Any, Any] | None = None,
        ylim: tuple[Any, Any] | None = None,
        *,
        xkcd: bool = False
    ):
        if xkcd:
            plt_params = plt.rcParams.copy()
            plt.rcParams["text.usetex"] = False
            plt.xkcd()
            plt.rcParams.update({
                "font.size": 9,
                "axes.titlesize": 10,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 8,
                "mathtext.fontset": "custom",
                "mathtext.rm": "xkcd",
                "mathtext.it": "xkcd",
                "mathtext.bf": "xkcd",
                "mathtext.default": "regular",
            })

        fig = self.fig(
            figsize=figsize,
            xlim=xlim,
            ylim=ylim
        )

        if xkcd:
            for ax in fig.get_axes():
                ax.grid(False)

        if self.path:
            fig.suptitle(self.path.method_name.replace("-", " ").title())

        plt.show()
        plt.close(fig)

        if xkcd:
            plt.rcParams.update(plt_params)

    def fig(
        self,
        figsize: tuple[float, float] | None = None,
        xlim: tuple[Any, Any] | None = None,
        ylim: tuple[Any, Any] | None = None,
    ) -> Figure:
        figsize = figsize or self.figsize

        if not self.plots:
            raise ValueError(
                "Plots() requires at least one Plot."
            )
        if not self.path:
            raise ValueError(
                "Path not initiated. Please ensure code is executed through supported channels"
            )

        layout = AutoLayout(ncols=self.ncols)
        nrows, ncols = layout.dimensions(len(self.plots))

        fig, axes_array = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            sharex=self.sharex,
            sharey=self.sharey,
            squeeze=False,
            constrained_layout=True,
        )
        all_axes: list[Axes] = list(axes_array.flat)

        # hide unused axes
        for ax in all_axes[len(self.plots):]:
            ax.set_visible(False)
        axes = all_axes[:len(self.plots)]

        # support manual limits passed from function call
        for plot, ax in zip(self.plots, axes):
            ax.grid(True)

            if plot.xlim is not None:
                ax.set_xlim(plot.xlim)
            if xlim is not None:
                ax.set_xlim(xlim)

            if plot.ylim is not None:
                ax.set_ylim(plot.ylim)
            if ylim is not None:
                ax.set_ylim(ylim)

        # call the plotting function (this is the decorated function)
        for plot, ax in zip(self.plots, axes):
            _ = plot.draw(ax)

            # handle superposed plots
            if self.superposed_plots:

                # change already plotted legend labels
                for line in ax.get_lines():
                    label = str(line.get_label())
                    if not label.startswith("_"):
                        label = f"{label} ({self.path.instance_name})"
                    else:
                        label = self.path.instance_name
                    line.set_label(label)

        if self.superposed_plots:

            for plots in self.superposed_plots:
                assert plots.plots is not None

                for superposed_plot, ax in zip(plots.plots, axes):
                    lines_before = set(ax.get_lines())

                    _ = superposed_plot.draw(ax)

                    new_lines = [
                        line for line in ax.get_lines()
                        if line not in lines_before
                    ]

                    # change already plotted legend labels
                    for line in new_lines:
                        label = str(line.get_label())
                        assert plots.path is not None

                        if not label.startswith("_"):
                            label = f"{label} ({plots.path.instance_name})"
                        else:
                            label = plots.path.instance_name
                        line.set_label(label)

                    ax.legend()

        # remove duplicated x/y labels when sharing x/y axis
        if self.sharex:
            for i, ax in enumerate(axes):
                if (ncols * (nrows - 1)) > i:
                    ax.set_xlabel("")
        if self.sharey:
            for i, ax in enumerate(axes):
                if i % ncols != 0:
                    ax.set_ylabel("")

        self.__apply_auto_limits(self.plots, axes)

        # support shared outside legend
        if self.outside_legend is not None:
            self.__add_shared_legend(
                fig, axes,
                location=self.outside_legend,  # type: ignore
                ncols=self.outside_legend_cols,
            )

        # prettier aligned labels
        fig.align_ylabels(axes)
        fig.align_xlabels(axes)

        # set figure name (in figure window when shown)
        fig_manager = fig.canvas.manager
        if fig_manager and self.path:
            old_title = fig_manager.get_window_title()
            title = f"{old_title}: {self.path.title}"
            fig_manager.set_window_title(title)

        return fig

    @staticmethod
    def __apply_auto_limits(
        plots: list[Plot],
        axes: list[Axes],
    ) -> None:
        """support auto limits and preserve manually set limits"""
        for plot, ax in zip(plots, axes):
            auto_limits = plot.auto_limits

            if auto_limits is None:
                return

            ax.relim()

            if auto_limits == "0.1scale":
                xmin, xmax = ax.dataLim.xmin, ax.dataLim.xmax
                ymin, ymax = ax.dataLim.ymin, ax.dataLim.ymax

                xrange = xmax - xmin
                yrange = ymax - ymin

                ax.set_xlim(xmin - 0.1 * xrange, xmax + 0.1 * xrange)
                ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
                continue

            autoscale_x = (
                auto_limits in ("x", "both")
                and ax.get_autoscalex_on()
            )

            autoscale_y = (
                auto_limits in ("y", "both")
                and ax.get_autoscaley_on()
            )

            ax.margins(
                x=0 if autoscale_x else None,
                y=0 if autoscale_y else None,
            )

            ax.autoscale_view(
                scalex=autoscale_x,
                scaley=autoscale_y,
            )

    @staticmethod
    def __add_shared_legend(
        fig: Figure,
        axes: list[Axes],
        *,
        location: LegendLocation,
        ncols: int,
    ) -> None:

        handles = []
        labels = []

        for ax in axes:
            h, l = ax.get_legend_handles_labels()

            handles.extend(h)
            labels.extend(l)

            legend = ax.get_legend()
            if legend:
                legend.remove()

        unique: dict[str, Any] = {}

        for handle, label in zip(handles, labels):
            if label not in unique:
                unique[label] = handle

        if not unique:
            return

        # rstrip suffix from labels if they're shared
        # suffix = common_suffix(list(unique.keys()))
        # if len(suffix) > 3:
        #     for key in list(unique.keys()):
        #         unique[key.rstrip(suffix)] = unique.pop(key)

        match location:

            case "lower":
                fig.legend(
                    unique.values(),
                    unique.keys(),
                    loc="outside lower center",
                    ncol=ncols,
                )

            case "right":
                fig.legend(
                    unique.values(),
                    unique.keys(),
                    loc="outside right center",
                    ncol=ncols,
                )

            case "top-inside":
                axes[0].legend(
                    unique.values(),
                    unique.keys(),
                )

            case _:
                raise ValueError(
                    f"Unsupported legend location: {location}"
                )


def scientific_plot(
    *,
    auto_limits: AutoLimits = "both",
    figsize: tuple[float, float] = (6, 4),
    outside_legend: LegendLocation = None,
    outside_legend_cols: int = 1,
) -> Callable[
    [Callable[Concatenate[T, Axes, P], Any]],
    Plots[T, P]
]:
    def decorator(
        func: Callable[Concatenate[T, Axes, P], Any]
    ) -> Plots[T, P]:

        return Plots(
            func=func,
            auto_limits=auto_limits,
            figsize=figsize,
            outside_legend=outside_legend,
            outside_legend_cols=outside_legend_cols
        )

    return decorator


@dataclass(frozen=True)
class Composition(Generic[T, P]):
    plots: list[Plot[T, P]]

    figsize: tuple[float, float] = (6, 4)
    ncols: int = 1

    sharex: bool = False
    sharey: bool = False

    outside_legend: LegendLocation = None
    outside_legend_cols: int = 1


def scientific_compose(
    *,
    auto_limits: AutoLimits = "both",
) -> Callable[
    [Callable[Concatenate[T, P], Composition]],
    Plots[T, P]
]:
    def decorator(
        func: Callable[Concatenate[T, P], Composition]
    ) -> Plots[T, P]:

        return Plots(
            compose_func=func,
            auto_limits=auto_limits,
        )

    return decorator


def arrange_matplotlib_figures(
    window_margin: int = 5,
    top_border: int = 25,
    border: int = 10,
):
    """arrange all matplotlib figures onto the largest monitor ready for plt.show()"""

    import matplotlib.pyplot as plt
    from rectpack import newPacker
    from screeninfo import get_monitors

    monitors = get_monitors()
    if not monitors:
        raise RuntimeError("No monitors found.")

    # largest monitor by pixel area.
    monitor = max(monitors, key=lambda m: m.width * m.height)

    def get_monitor_dpi(m):
        width_mm = getattr(m, "width_mm", None)
        height_mm = getattr(m, "height_mm", None)

        if (
            width_mm
            and height_mm
            and width_mm > 0
            and height_mm > 0
        ):
            dpi_x = m.width / (width_mm / 25.4)
            dpi_y = m.height / (height_mm / 25.4)

            return dpi_x, dpi_y

        # screeninfo sometimes exposes a dpi attribute
        dpi = getattr(m, "dpi", None)

        if dpi and dpi > 0:
            return float(dpi), float(dpi)

        # last-resort assumption
        return 96.0, 96.0

    monitor_dpi_x, monitor_dpi_y = get_monitor_dpi(monitor)

    # useage monitor area
    available_w = monitor.width - 2 * window_margin
    available_h = monitor.height - 2 * window_margin

    if available_w <= 0 or available_h <= 0:
        raise RuntimeError(
            "window_margin is larger than the available monitor."
        )

    # get all matplotlib figures
    figures = [
        plt.figure(num)
        for num in plt.get_fignums()
    ]
    if not figures:
        return

    # work out the required space for each fig
    rectangles = []
    for i, fig in enumerate(figures):

        width_in, height_in = fig.get_size_inches()

        canvas_w = round(width_in * monitor_dpi_x)
        canvas_h = round(height_in * monitor_dpi_y)

        manager = fig.canvas.manager

        frame_w = 0
        frame_h = 0

        window = getattr(manager, "window", None)

        if window is not None:

            # TkAgg
            if hasattr(window, "winfo_width"):
                try:
                    client_w = window.winfo_width()
                    client_h = window.winfo_height()

                    outer_w = window.winfo_rootx()
                except Exception:
                    pass

        # use canvis size as window size
        window_w = canvas_w + frame_w
        window_h = canvas_h + frame_h

        # scale down windows if they're largest than the monitor
        scale = min(
            1.0,
            available_w / window_w,
            available_h / window_h,
        )

        if scale < 1.0:
            window_w = max(1, round(window_w * scale))
            window_h = max(1, round(window_h * scale))

        rectangles.append(
            {
                "figure": fig,
                "manager": manager,
                "window": window,
                "width": window_w,
                "height": window_h,
            }
        )

    # pack rectangles
    packer = newPacker(rotation=False)

    for i, rect in enumerate(rectangles):
        packer.add_rect(
            rect["width"] + border,
            rect["height"] + border + top_border,
            rid=i,
        )

    packer.add_bin(available_w, available_h)
    packer.pack()  # type: ignore

    # check everything fitted
    packed_ids = {
        rid
        for _, _, _, _, _, rid in packer.rect_list()
    }

    missing = set(range(len(rectangles))) - packed_ids
    if missing:
        raise RuntimeError(
            f"{len(missing)} figure(s) could not fit on "
            f"the selected monitor."
        )

    # position windows
    for (
        _,
        x,
        y,
        _packed_w,
        _packed_h,
        rid,
    ) in packer.rect_list():

        rect = rectangles[rid]

        manager = rect["manager"]
        window = rect["window"]

        x = int(monitor.x + window_margin + x - border * 0.5)
        y = int(monitor.y + window_margin + y +
                (top_border * 0.5) - (border * 0.5))

        # TkAgg backend logic
        if window is not None and hasattr(window, "geometry"):
            window.geometry(f"{rect['width']}x{rect['height']}+{x}+{y}")
        else:
            raise RuntimeError(
                "Current matplotlib backend does not expose a supported window-positioning API"
            )

    # redraw figures
    for fig in figures:
        fig.canvas.draw_idle()


def _show():
    try:
        arrange_matplotlib_figures()
    except Exception as e:
        import warnings
        warnings.warn(
            f"Encoutered error when arranging figures. {e}. Showing without arranging"
        )
    plt.show()
