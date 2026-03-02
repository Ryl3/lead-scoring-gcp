"""Shared plot styling and display helpers used across all notebooks."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

PALETTE    = ["#2E6DA4", "#E8734A", "#48A868", "#9B59B6", "#E8C84A"]
CONV_COLOR = "#2E6DA4"
BASE_COLOR = "#AACCE8"
NEG_COLOR  = "#E8734A"


def set_style():
    plt.rcParams.update({
        "figure.facecolor":  "#FAFAFA",
        "axes.facecolor":    "#FAFAFA",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "sans-serif",
        "axes.titlesize":    13,
        "axes.labelsize":    11,
    })


def pct_axis(ax, axis="y"):
    fmt = mtick.PercentFormatter(1.0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def add_bar_labels(ax, bars, values, fmt="{:.0%}", offset=0.005, fontsize=9):
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(v),
            ha="center", fontsize=fontsize,
        )
