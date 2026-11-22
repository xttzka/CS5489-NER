"""Shared typography and palette for the project report figures."""

import matplotlib as mpl


TEXT_WIDTH = 6.5  # inches
METHOD = {
    "M0": "#DDDDDD", "M1": "#DDCC77", "M2": "#999933", "M3": "#117733",
    "M4": "#332288", "Span": "#88CCEE", "GLiNER": "#44AA99", "LLM": "#CC6677",
    "LoRA-LLM": "#882255",
}
ENTITY = {"PER": "#4477AA", "ORG": "#EE8866", "LOC": "#99DDFF", "MISC": "#663333"}
ERROR = {
    "boundary": "#6699CC", "type": "#EECC66", "boundary_and_type": "#889966",
    "missed": "#BBCCEE", "spurious": "#775588",
}
DATASET = {
    "conll2003": "#77AADD", "wnut17": "#EEDD88", "btc": "#666633", "merged": "#222255",
    "mit_restaurant": "#CC6677", "conllpp": "#117733", "wikigold": "#AA4499",
}


def setup():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": "#E5E5E5", "grid.linewidth": 0.6,
        "lines.linewidth": 1.4, "patch.edgecolor": "#444444", "patch.linewidth": 0.6,
        "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
    })
