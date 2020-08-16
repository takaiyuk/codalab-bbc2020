from dataclasses import dataclass
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Arc, Circle, Rectangle


@dataclass
class BasketCourt:
    """unit is feet"""

    origin_xy: Tuple[int, int] = (0, 0)
    x_length: int = 47
    y_length: int = 50
    hoop_xy: Tuple[float, float] = (4.75, 25.0)
    hoop_radius: float = 0.75


class DrawNBACourt:
    def __init__(self, ax: Axes = None, color: str = "black", lw: int = 1):
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.color = color
        self.lw = lw
        self.outer_lines = None
        self.center_circle = None
        self.corner_three_a = None
        self.corner_three_b = None
        self.three_arc_a = None
        self.three_arc_b = None
        self.outer_box = None
        self.inner_box = None
        self.free_throw_circle_a = None
        self.free_throw_circle_b = None
        self.free_throw_circle_c = None
        self.restricted = None
        self.backboard = None
        self.hoop = None

    def _plot_outline(self) -> None:
        self.outer_lines = Rectangle(
            BasketCourt.origin_xy,
            width=BasketCourt.x_length,
            height=BasketCourt.y_length,
            color=self.color,
            linewidth=self.lw,
            fill=False,
        )

    def _plot_center_circle(self) -> None:
        center_circle_xy = (BasketCourt.x_length, BasketCourt.y_length / 2)
        center_circle_radius = 6
        self.center_circle = Arc(
            center_circle_xy,
            center_circle_radius,
            center_circle_radius,
            theta1=90,
            theta2=270,
            color=self.color,
            linewidth=self.lw,
            fill=False,
        )

    def _plot_three_point_line(self):
        # Corner three-point line
        self.corner_three_a = Rectangle(
            (0, 3), width=14, height=0, color=self.color, linewidth=self.lw, fill=False
        )
        self.corner_three_b = Rectangle(
            (0, 47), width=14, height=0, color=self.color, linewidth=self.lw, fill=False
        )

        # Three-point arc
        self.three_arc_a = Arc(
            BasketCourt.hoop_xy,
            width=23.9 * 2,
            height=23.9 * 2,
            theta1=0,
            theta2=68,
            color=self.color,
            linewidth=self.lw,
            fill=False,
        )
        self.three_arc_b = Arc(
            BasketCourt.hoop_xy,
            width=23.9 * 2,
            height=23.9 * 2,
            theta1=292,
            theta2=0,
            color=self.color,
            linewidth=self.lw,
            fill=False,
        )

    def _plot_paint_box(self):
        self.outer_box = Rectangle(
            (0, 17),
            width=19,
            height=16,
            color=self.color,
            linewidth=self.lw,
            fill=False,
        )
        self.inner_box = Rectangle(
            (0, 19),
            width=19,
            height=12,
            color=self.color,
            linewidth=self.lw,
            fill=False,
        )

    def _plot_free_throw_line(self):
        self.free_throw_circle_a = Arc(
            (19, 25),
            width=6 * 2,
            height=6 * 2,
            theta1=0,
            theta2=90,
            color=self.color,
            linewidth=self.lw,
            fill=False,
        )
        self.free_throw_circle_b = Arc(
            (19, 25),
            width=6 * 2,
            height=6 * 2,
            theta1=90,
            theta2=270,
            color=self.color,
            linewidth=self.lw,
            fill=False,
            linestyle="dashed",
        )
        self.free_throw_circle_c = Arc(
            (19, 25),
            width=6 * 2,
            height=6 * 2,
            theta1=270,
            theta2=0,
            color=self.color,
            linewidth=self.lw,
            fill=False,
        )

    def _plot_back_board(self):
        self.backboard = Rectangle(
            (4, 22), width=0, height=6, color=self.color, linewidth=self.lw, fill=False
        )

    def _plot_hoop(self):
        self.hoop = Circle(
            BasketCourt.hoop_xy,
            radius=BasketCourt.hoop_radius,
            color=self.color,
            linewidth=self.lw,
            fill=False,
        )

    def plot(self) -> Axes:
        # plt.figure(figsize=(12, 11))
        self._plot_outline()
        self._plot_center_circle()
        self._plot_three_point_line()
        self._plot_paint_box()
        self._plot_free_throw_line()
        self._plot_back_board()
        self._plot_hoop()
        # plt.xlim(0 - 5, x_length + 5)
        # plt.ylim(0 - 5, y_length + 5)

        d = self.__dict__
        for k, v in d.items():
            if k in ["ax", "color", "lw"] or v is None:
                continue
            self.ax.add_patch(v)

        return self.ax
