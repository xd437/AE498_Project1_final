import glob
import itertools
import os
import re

import numpy as np
import openpyxl
from scipy.integrate import solve_ivp

# Constants
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "Radius")

PX_PER_MM = 19.23            # pixels per millimetre (assumes one ruler division = 1 mm)
DT_MS = 1e3 / 67500          # frame interval (ms) at 67 500 frames per second

TERM_NAMES = ["1", "Rd^2", "Rd/R", "1/R", "R^-3", "R^-4", "Rd^3", "Rd", "R", "R*Rd", "R^2"]
RP_TERMS = [0, 1, 2, 3]      # the four RP terms: 1, Rd^2, Rd/R, 1/R

# Textbook coefficients for water at 20 °C
TEXTBOOK = np.array([-99.184,     # -(p_inf - p_v)/rho
                     -1.5,        # inertia
                     -0.004008,   # viscosity, -4*nu
                     -0.14589])   # surface tension, -2*sigma/rho

K_MAX = 4          # maximum number of terms in a model
SLACK = 0.02       # model selection
N_BOOT = 400       # number of bootstrap resamples

# Import the experimental data
def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def read_radius_file(path):
    sheet = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    header = [cell.value for cell in sheet[1]]
    rows = [r for r in sheet.iter_rows(min_row=2, values_only=True) if r and r[0] is not None]

    t_ms = np.array([to_float(r[header.index("t_us")]) for r in rows]) / 1000.0
    R_px = np.array([to_float(r[header.index("R_px")]) for r in rows])
    ok = np.array([r[header.index("ok")] == 1 for r in rows])
    return t_ms, R_px, ok


def cut_window(R, pre, stop):
    peak = int(np.argmax(R))
    first = max(0, peak - pre)
    last = peak
    while last + 1 < len(R) and R[last + 1] >= stop * R[peak]:
        last += 1
    return first, last


def experiment_files():
    files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
    return sorted(files, key=lambda f: int(re.findall(r"start_(\d+)", f)[0]))


def load_experiments(pre=5, stop=0.5, min_frames=12):
    clips = []
    for path in experiment_files():
        name = re.findall(r"start_\d+", path)[0]
        t, R_px, ok = read_radius_file(path)
        usable = ok & np.isfinite(R_px) & (R_px > 0)
        t, R_px = t[usable], R_px[usable]

        first, last = cut_window(R_px, pre, stop)
        if last - first + 1 < min_frames:
            continue

        clips.append(dict(name=name, t=t[first:last + 1], R=R_px[first:last + 1] / PX_PER_MM))
    return clips

# Synthetic data
def rp_full(t, z):
    R, Rd = z
    c0, _, c_visc, c_surf = TEXTBOOK
    R_dd = (c0 - 1.5 * Rd**2 + c_visc * Rd / R + c_surf / R) / R
    return [Rd, R_dd]


def make_synthetic(n=20, noise_px=0.03, seed=1, pre=5, stop=0.5):
    rng = np.random.default_rng(seed)
    clips = []
    for i in range(n):
        Rmax = rng.uniform(1.7, 3.0)
        phase = rng.uniform(0.0, 1.0)
        t_collapse = 0.915 * Rmax / np.sqrt(-TEXTBOOK[0])

        t_before = -(pre + 4) * DT_MS
        t_after = 0.95 * t_collapse
        expansion = solve_ivp(rp_full, (0, t_before), [Rmax, 0.0], rtol=1e-10, atol=1e-12, dense_output=True)
        collapse = solve_ivp(rp_full, (0, t_after), [Rmax, 0.0], rtol=1e-10, atol=1e-12, dense_output=True)

        t = (np.arange(-(pre + 3), int(t_after / DT_MS)) + phase) * DT_MS
        R = np.where(t < 0, expansion.sol(np.minimum(t, 0))[0], collapse.sol(np.maximum(t, 0))[0])
        R = R + rng.normal(0.0, noise_px / PX_PER_MM, len(R))

        first, last = cut_window(R, pre, stop)
        clips.append(dict(name=f"synthetic_{i + 1:02d}", t=t[first:last + 1], R=R[first:last + 1]))
    return clips


# Derivatives and library matrix
def derivatives(t, R, half_width=3, order=4):
    index, R_dot, R_ddot = [], [], []
    for k in range(half_width, len(R) - half_width):
        tau = t[k - half_width:k + half_width + 1] - t[k]
        c = np.polyfit(tau, R[k - half_width:k + half_width + 1], order)
        index.append(k)
        R_dot.append(c[-2])
        R_ddot.append(2 * c[-3])
    return np.array(index), np.array(R_dot), np.array(R_ddot)


def library_rows(clip):
    index, V, A = derivatives(clip["t"], clip["R"])
    R = clip["R"][index]
    Theta = np.column_stack([np.ones_like(R), V**2, V / R, 1 / R, R**-3, R**-4,
                             V**3, V, R, R * V, R**2])
    return Theta, R * A


# Least squares, cross-validation, bootstrap
def least_squares(Theta, y):
    scale = np.linalg.norm(Theta, axis=0)
    return np.linalg.lstsq(Theta / scale, y, rcond=None)[0] / scale


def stack(blocks, terms, experiments):
    Theta = np.vstack([blocks[e][0][:, terms] for e in experiments])
    y = np.concatenate([blocks[e][1] for e in experiments])
    return Theta, y


def cv_error(blocks, terms):
    squared_residual, squared_signal = 0.0, 0.0
    for held_out in range(len(blocks)):
        others = [e for e in range(len(blocks)) if e != held_out]
        xi = least_squares(*stack(blocks, terms, others))
        Theta_test, y_test = blocks[held_out]
        residual = y_test - Theta_test[:, terms] @ xi
        squared_residual += residual @ residual
        squared_signal += y_test @ y_test
    return np.sqrt(squared_residual / squared_signal)


def bootstrap_ci(blocks, terms, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(N_BOOT):
        chosen = rng.integers(0, len(blocks), len(blocks))
        samples.append(least_squares(*stack(blocks, terms, chosen)))
    return np.percentile(samples, 2.5, axis=0), np.percentile(samples, 97.5, axis=0)


def best_subsets(blocks):
    best = {}
    for k in range(1, K_MAX + 1):
        scored = [(cv_error(blocks, list(terms)), list(terms))
                  for terms in itertools.combinations(range(len(TERM_NAMES)), k)]
        best[k] = min(scored)
    return best


def formula(xi, terms):
    return " ".join(f"{c:+.4g}[{TERM_NAMES[j]}]" for c, j in zip(xi, terms))

# Full analysis: model selection, confidence intervals, forced RP fit
def analyse(clips, label):
    blocks = [library_rows(c) for c in clips]
    Theta, y = stack(blocks, list(range(len(TERM_NAMES))), range(len(blocks)))
    print(f"\n== {label}: {len(clips)} bubbles, {len(y)} equations")

    # best combination for each k
    best = best_subsets(blocks)
    e_min = min(err for err, _ in best.values())
    k_selected = min(k for k, (err, _) in best.items() if err <= (1 + SLACK) * e_min)
    for k, (err, terms) in best.items():
        mark = "  <== selected" if k == k_selected else ""
        print(f"   k={k}  E={err:.4f}  {formula(least_squares(Theta[:, terms], y), terms)}{mark}")

    # coefficients and confidence intervals of the selected model
    terms = best[k_selected][1]
    xi = least_squares(Theta[:, terms], y)
    low, high = bootstrap_ci(blocks, terms)
    print("   selected model, 95% CI:")
    for j, c, lo, hi in zip(terms, xi, low, high):
        print(f"      {TERM_NAMES[j]:6s} {c:+.4f}  [{lo:+.4f}, {hi:+.4f}]")

    # force all four RP terms into the fit and see what the viscous and surface-tension coefficients become
    xi_rp = least_squares(Theta[:, RP_TERMS], y)
    low_rp, high_rp = bootstrap_ci(blocks, RP_TERMS)
    rms_y = np.sqrt(np.mean(y**2))
    print("   forced RP fit {1, Rd^2, Rd/R, 1/R}:")
    for j, c, lo, hi, tb in zip(RP_TERMS, xi_rp, low_rp, high_rp, TEXTBOOK):
        textbook_share = np.sqrt(np.mean((tb * Theta[:, j])**2)) / rms_y
        print(f"      {TERM_NAMES[j]:6s} {c:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"textbook {tb:+.4f}  textbook RMS share {textbook_share:.2e}")

    return dict(blocks=blocks, Theta=Theta, y=y, best=best, terms=terms, xi=xi,
                xi_rp=xi_rp, ci_rp=(low_rp, high_rp))


def resolution_floor(result):
    low, high = result["ci_rp"]
    print("   resolution floor (largest |CI bound| in the forced fit):")
    for j in (2, 3):
        floor = max(abs(low[j]), abs(high[j]))
        print(f"      {TERM_NAMES[j]:6s} {floor:.3g}  =  {floor / abs(TEXTBOOK[j]):.0f} x textbook")


# Additional checks on the experimental data
def forward_rms_px(clips, terms, xi):
    def rhs(t, z):
        R, Rd = z
        candidates = np.array([1, Rd**2, Rd / R, 1 / R, R**-3, R**-4, Rd**3, Rd, R, R * Rd, R**2])
        return [Rd, candidates[terms] @ xi / R]

    rms = []
    for clip in clips:
        index, V, _ = derivatives(clip["t"], clip["R"])
        t, R_measured = clip["t"][index[0]:], clip["R"][index[0]:]
        solution = solve_ivp(rhs, (t[0], t[-1]), [R_measured[0], V[0]], t_eval=t, rtol=1e-8, atol=1e-10)
        rms.append(np.sqrt(np.mean((solution.y[0] - R_measured)**2)) * PX_PER_MM)
    return np.array(rms)


def experiment_checks(clips, result):
    Theta, y = result["Theta"], result["y"]
    scaled = Theta / np.linalg.norm(Theta, axis=0)
    print(f"   corr(R, R^2) = {np.corrcoef(Theta[:, 8], Theta[:, 10])[0, 1]:.4f}, "
          f"condition number (scaled) = {np.linalg.cond(scaled):.3g}, "
          f"constant of the full 11-term fit = {least_squares(Theta, y)[0]:+.4g}")
    print(f"   radius range in the windows: {Theta[:, 8].min():.2f}-{Theta[:, 8].max():.2f} mm")
    selected = np.median(forward_rms_px(clips, result["terms"], result["xi"]))
    print(f"   forward integration, median RMS deviation: selected {selected:.2f} px")


# Main program
if __name__ == "__main__":
    synthetic = analyse(make_synthetic(), "synthetic RP data (textbook coefficients, 0.03 px noise)")
    resolution_floor(synthetic)

    experiments = load_experiments()

    result = analyse(experiments, "experiments (5 frames before the peak to 0.5 Rmax)")
    experiment_checks(experiments, result)
