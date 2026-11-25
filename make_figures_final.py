import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import rp_sindy_radius_final as rp

OUT_DIR = rp.HERE

# Plot style
COLORS = dict(blue="#2a78d6", orange="#eb6834", red="#e34948", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df")
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "mathtext.fontset": "stix",
    "axes.edgecolor": COLORS["ink2"], "xtick.color": COLORS["ink2"], "ytick.color": COLORS["ink2"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": COLORS["grid"], "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight",
})


def figure_radius():
    bubbles = []
    for path in rp.experiment_files():
        t, R_px, ok = rp.read_radius_file(path)
        usable = ok & np.isfinite(R_px) & (R_px > 0)
        manual = usable & (np.abs(2 * R_px - np.round(2 * R_px)) < 1e-9)   # manual Fiji values are multiples of 0.5 px
        bubbles.append((t * 1000, R_px / rp.PX_PER_MM, usable, manual))  # time in microseconds

    bubbles.sort(key=lambda b: b[1][b[2]].max())            # sort by Rmax: light colours for small bubbles
    colors = plt.get_cmap("Blues")(np.linspace(0.35, 0.95, len(bubbles)))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for (t, R, usable, manual), color in zip(bubbles, colors):
        automatic = usable & ~manual
        peak = np.argmax(np.where(usable, R, -1))
        ax_a.plot(t[usable], R[usable], "-", color=color, lw=1.1)
        ax_a.plot(t[automatic], R[automatic], "o", color=color, ms=2.2)
        ax_a.plot(t[manual], R[manual], "o", mfc="white", mec=COLORS["orange"], ms=3.4, mew=0.9)
        ax_b.plot(t[usable] / t[peak], R[usable] / R[peak], "-", color=color, lw=1.1)

    ax_a.plot([], [], "o", color=colors[len(colors) // 2], ms=3, label="automatic (Python)")
    ax_a.plot([], [], "o", mfc="white", mec=COLORS["orange"], ms=3.5, label="manual (Fiji)")
    ax_a.legend(fontsize=7.5, loc="lower center")
    ax_a.set_xlabel(r"$t$ since spark [$\mu$s]")
    ax_a.set_ylabel(r"$R$ [mm]")
    ax_a.set_title(f"(a) first cycle of the {len(bubbles)} bubbles", fontsize=9)
    ax_b.set_xlabel(r"$t/t_{\max}$")
    ax_b.set_ylabel(r"$R/R_{\max}$")
    ax_b.set_title("(b) normalized by the maximum", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_radius.pdf"))


def cv_curve(blocks):
    best = rp.best_subsets(blocks)
    return [1.0] + [best[k][0] for k in range(1, rp.K_MAX + 1)]


def forced_fit(blocks):
    Theta, y = rp.stack(blocks, rp.RP_TERMS, range(len(blocks)))
    xi = rp.least_squares(Theta, y)
    low, high = rp.bootstrap_ci(blocks, rp.RP_TERMS)
    return [(xi[j], low[j], high[j]) for j in (2, 3)]


def figure_results():
    experiments = [rp.library_rows(c) for c in rp.load_experiments()]
    synthetic = [rp.library_rows(c) for c in rp.make_synthetic()]
    k = np.arange(rp.K_MAX + 1)
    error_exp, error_syn = cv_curve(experiments), cv_curve(synthetic)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.4, 3.0))

    ax_a.plot(k, error_exp, "o-", color=COLORS["blue"], label=f"experiments ({len(experiments)} bubbles)")
    ax_a.plot(k, error_syn, "s--", color=COLORS["ink2"], ms=4, label="synthetic RP data")
    for error in (error_exp, error_syn):                        # the two-term model is selected in both cases
        ax_a.plot([2], [error[2]], "o", ms=10, mfc="none", mec=COLORS["red"], mew=1.6)
    ax_a.set_xticks(k)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_xlabel("number of terms $k$")
    ax_a.set_ylabel("cross-validation error $E$")
    ax_a.legend(fontsize=7.5, loc="upper right")
    ax_a.set_title("(a) best $k$-term model (red: selected)", fontsize=9)

    x = np.arange(2)
    ax_b.axhline(0, color=COLORS["ink2"], lw=0.8)
    ax_b.plot(x - 0.25, rp.TEXTBOOK[2:4], "D", color=COLORS["ink2"], ms=6, label="textbook value (water, 20 °C)")
    for offset, blocks, color, label in ((0.0, synthetic, COLORS["orange"], "fitted, synthetic data"),
                                         (0.25, experiments, COLORS["blue"], "fitted, experiments")):
        value, low, high = np.array(forced_fit(blocks)).T
        ax_b.errorbar(x + offset, value, [value - low, high - value], fmt="o", color=color, capsize=3, label=label)
    ax_b.set_yscale("symlog", linthresh=1e-2)                   # symmetric log scale to show both signs
    ax_b.set_ylim(-3000, 300)
    ax_b.set_xlim(-0.6, 1.6)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([r"viscous $\dot R/R$", r"surface tension $1/R$"])
    ax_b.set_ylabel("coefficient")
    ax_b.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 0.02),
                frameon=True, framealpha=0.9, edgecolor="none")
    ax_b.set_title("(b) forced fit of the four RP terms (95% CI)", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_results.pdf"))


if __name__ == "__main__":
    figure_radius()
    figure_results()
