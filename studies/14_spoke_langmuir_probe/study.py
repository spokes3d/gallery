"""Langmuir probe in spokes.

What would a Langmuir probe show of a rotating spoke, and what would it miss? A small planar
probe tip (a disc of 0.5 mm radius) sits at three fixed points in the lab frame: on the
racetrack at $R_0 = 3$ cm, 3 mm inside it and 1.5 mm outside it. The background is the made-up
three-spoke pattern of the rotating-spoke test-electron study, built with byob in the frame that
rotates with the spokes and turned at 10 km/s (at $R_0$) in the lab frame: a 25 V radial well,
three 12 V potential humps with a sharp leading edge and a long tail, $T_e$ from 3 eV outside to
7 eV inside a spoke, and the density from Boltzmann's relation. At each instant the probe sees
the plasma at its position rotated back into the spoke frame.

The probe is a wall of byob in the local plasma: the ions reach it at the Bohm flux of the
sheath edge, $n_s u_B$ with $n_s = n\\,e^{-1/2}$, and the electrons with the random flux
$\\frac14 n_s \\bar v_e$ reduced by the Boltzmann factor of byob's sheath drop in front of a
wall biased to $V$. Above the sheath edge potential byob has no electron sheath, so the study
continues the electron current to its saturation at the plasma potential, $\\frac14 e A n \\bar
v_e$ (the standard thin-sheath probe):

$$I(V) = \\tfrac14 e A n \\bar v_e\\, e^{-\\max(V_p - V,\\,0)/T_e} - e A n\\,e^{-1/2} u_B.$$

The first figure shows what a probe at a fixed bias records as the spokes pass: the floating
potential (byob's floating sheath drop), and the ion and electron saturation currents. Then the
bias is swept from $-50$ to $+25$ V in a sawtooth, with a little white noise on the current, and
each sweep is analysed the standard way: a straight line through the ion-saturation part, the
floating potential where the current changes sign, $T_e$ from the slope of $\\ln I_e$ in the
electron-retardation region (the ion line corrected for the electron tail, three passes), and
$V_p$ at the knee where that line meets the saturation line;
$n_e$ follows from the ion saturation current and the fitted $T_e$. The second figure shows sweeps
centred between the spokes, on a leading edge and inside a spoke; the third the fitted $n_e$,
$T_e$ and $V_p$ against time, for sweeps 0.1, 1 and 10 times as long as the leading edge
takes to pass the probe (its 10-90 % width over the spoke speed, 0.26 µs).

Compared: the fitted values with the background at the probe (at the middle of each sweep).
Not compared: any measurement. The probe does not perturb the plasma, its sheath is thin and
collisionless and does not grow with the bias (a flat ion saturation), the electrons are
unmagnetized at the probe (here the electron Larmor radius is a third of the probe radius, which
would lower the electron saturation current), there is no secondary emission and no RF or
circuit effects, and sweeps 0.1 times the passage time (tens of MHz) are faster than real probes
are swept.

Reference: the picture of spokes as rotating ionization zones with a potential hump and a sharp
leading edge, e.g. Hecimovic and von Keudell, J. Phys. D 51 (2018) 453001. The kind of
measurement this mimics: probes in rotating spokes of HiPIMS and DC magnetrons, e.g. Estrin et
al. (2017) and Held et al. (2020).
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

#: The recorded periods of the pattern, the samples of the traces per period and the bias
#: points per sweep.
PRESETS = {
    "smoke": {"periods": 0.5, "samples": 200, "points": 100},
    "quick": {"periods": 2.0, "samples": 600, "points": 300},
}

QE = 1.602176634e-19
ME = 9.1093837015e-31
L = 0.05  # half width of the domain [m]
NG = 201  # grid nodes per axis of the background
R0 = 0.03  # racetrack radius [m]
WR = 0.006  # racetrack half width [m]
B0 = 0.03  # radial field at the racetrack [T]
V_RING = 25.0  # depth of the radial well [V]
V_SPOKE = 12.0  # potential hump of a spoke [V]
TE0, DTE = 3.0, 4.0  # electron temperature outside, and extra inside a spoke [eV]
M_SPOKES = 3
SPOKE_SPEED = 1.0e4  # at R0, in the direction of the Hall drift [m/s]
OMEGA = -SPOKE_SPEED / R0  # [rad/s]; negative: the sense of the Hall drift
PERIOD = 2 * np.pi / (M_SPOKES * abs(OMEGA))  # the pattern repeats after this [s]
EDGE = 0.04  # angular width of the tanh of the leading edge [rad]
#: the leading edge's 10-90 % width at R0 over the spoke speed: the time it takes to pass
PASSAGE = 2 * np.arctanh(0.8) * EDGE * R0 / SPOKE_SPEED
#: lab points (r, 0); the background is symmetric about R0, so the two others are not
PROBES = {"R0 - 3 mm": R0 - 0.003, "R0": R0, "R0 + 1.5 mm": R0 + 0.0015}
PROBE_RADIUS = 0.5e-3  # a planar disc, collecting on one face [m]
AREA = np.pi * PROBE_RADIUS**2
V_MIN, V_MAX = -50.0, 25.0  # bias sweep [V]
NOISE = 1.0e-6  # white noise on the probe current [A]
SWEEP_RATIOS = (0.1, 1.0, 10.0)  # sweep time over the passage time of the leading edge
ITERATIONS = 3  # passes of the ion-line and electron-tail fits
INSTANTS = ("between the spokes", "on the leading edge", "inside the spoke")


def spoke_shape(theta):
    """The sum of the asymmetric spoke profiles in the spoke frame, peak 1."""
    shape = np.zeros_like(theta)
    for k in range(M_SPOKES):
        psi = np.angle(np.exp(1j * (theta - 2 * np.pi * k / M_SPOKES)))
        # the spoke moves towards -theta: a sharp edge at psi = -0.15, a long tail behind
        rise = 0.5 * (1 + np.tanh((psi + 0.15) / EDGE))
        shape += rise * np.exp(-np.clip(psi + 0.15, 0.0, None) / 0.35)
    return shape


def build_background():
    """The made-up spoke-frame background: potential, Te, and n from Boltzmann's relation."""
    from byob import Background, UniformGrid
    from plasmacoll.constants import ATOMIC_MASS

    grid = UniformGrid((-L, -L), (L, L), (NG, NG))
    x, y = np.meshgrid(*grid.axes, indexing="ij")
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    ring = np.exp(-(((r - R0) / WR) ** 2))
    shape = spoke_shape(theta)
    phi = -V_RING * (1 - ring) + V_SPOKE * ring * shape
    te = TE0 + DTE * ring * shape
    bulk = {
        "density": 1e18 * np.exp(phi / te),
        "electron_temperature": te,
        "ion_temperature": np.full_like(te, 0.1),
        "potential": phi,
    }
    return Background(grid, bulk, ion_mass=39.95 * ATOMIC_MASS)


def plasma_at(background, radius: float, t) -> dict:
    """The background at the lab point (radius, 0) at times ``t``: the spoke-frame angle -OMEGA t."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    angle = -OMEGA * t
    points = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    state = background.evaluate(points)
    return {
        "n": state["density"],
        "te": state["electron_temperature"],
        "ti": state["ion_temperature"],
        "phi": state["potential"],
    }


def bohm_speed(te, ti, ion_mass):
    """The Bohm speed [m/s] for temperatures in eV."""
    return np.sqrt(QE * (te + ti) / ion_mass)


def ion_saturation(state: dict, ion_mass: float):
    """The ion current [A] to the probe: the Bohm flux at the sheath edge, n_s = n exp(-1/2)."""
    return (
        QE
        * AREA
        * state["n"]
        * np.exp(-0.5)
        * bohm_speed(state["te"], state["ti"], ion_mass)
    )


def electron_saturation(state: dict):
    """The electron current [A] at the plasma potential: the random flux n v_e / 4."""
    mean_speed = np.sqrt(8 * QE * state["te"] / (np.pi * ME))
    return 0.25 * QE * AREA * state["n"] * mean_speed


def sheath_drop(wall, bulk, ion_mass):
    """byob's sheath drop in front of ``wall`` (floating, or biased to an array of biases)."""
    from byob import sheath_drop as drop

    return drop(wall, bulk, ion_mass)


def floating_potential(state: dict, ion_mass: float):
    """The floating potential [V]: the bulk potential less the presheath and byob's sheath drop."""
    from byob import Wall
    from byob.sheath import PlasmaState

    bulk = PlasmaState(state["n"], state["te"], state["ti"], state["phi"])
    return state["phi"] - 0.5 * state["te"] - sheath_drop(Wall(None), bulk, ion_mass)


def probe_current(bias, state: dict, ion_mass: float):
    """The probe current [A] at ``bias`` (electron collection positive) in the plasma ``state``.

    Below the sheath edge potential the electron current is the random flux at the sheath edge
    reduced by byob's sheath drop; above it, where byob has no electron sheath, the thin-sheath
    electron current rises on to its saturation at the plasma potential.
    """
    from byob import Wall
    from byob.sheath import PlasmaState

    bulk = PlasmaState(state["n"], state["te"], state["ti"], state["phi"])
    drop = sheath_drop(Wall(None, "biased", bias), bulk, ion_mass)
    saturation = electron_saturation(state)
    retarded = saturation * np.exp(-0.5) * np.exp(-drop / state["te"])
    attracted = saturation * np.exp(np.minimum(bias - state["phi"], 0.0) / state["te"])
    electron = np.where(drop > 0, retarded, attracted)
    return electron - ion_saturation(state, ion_mass)


def line_fits(x, y):
    """Least-squares lines through the first k points, for every k: slope, intercept, SSE."""
    k = np.arange(1, x.size + 1)
    sx, sy = np.cumsum(x), np.cumsum(y)
    sxx, sxy, syy = np.cumsum(x * x), np.cumsum(x * y), np.cumsum(y * y)
    cxx = sxx - sx * sx / k
    cxy = sxy - sx * sy / k
    cyy = syy - sy * sy / k
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = cxy / cxx
        sse = cyy - slope * cxy
    return slope, (sy - slope * sx) / k, sse


def fit_characteristic(bias, current, ion_mass: float) -> dict:
    """The standard analysis of a swept I-V: V_f, I_is, Te (retardation), V_p (knee), n_e."""
    nan = {
        k: np.nan for k in ("vf", "te", "vp", "n", "ion_slope", "ion_offset", "ln_ies")
    }
    up = np.flatnonzero((current[:-1] < 0) & (current[1:] >= 0))
    if not up.size:
        return nan
    k = up[0]
    vf = bias[k] - current[k] * (bias[k + 1] - bias[k]) / (current[k + 1] - current[k])
    # ion saturation: a straight line through the lower half of the sweep below V_f, less
    # the tail of the electron current fitted below; three passes
    part = bias < vf - 0.5 * (vf - bias[0])
    if part.sum() < 3:
        return nan
    tail = np.zeros(part.sum())
    for _ in range(ITERATIONS):
        ion_slope, ion_offset = np.polyfit(bias[part], tail - current[part], 1)
        electron = current + ion_slope * bias + ion_offset
        use = (bias >= vf) & (electron > 5 * NOISE)
        x, y = bias[use], np.log(np.where(use, electron, 1.0)[use])
        if x.size < 10:
            return nan
        # two straight lines in ln I_e: retardation and saturation, the knee where they meet
        a1, b1, left = line_fits(x, y)
        a2, b2, right = (v[::-1] for v in line_fits(x[::-1], y[::-1]))
        split = np.arange(4, x.size - 3)  # left: x[:split], right: x[split:]
        best = split[np.nanargmin(left[split - 1] + right[split])]
        s1, c1, s2, c2 = a1[best - 1], b1[best - 1], a2[best], b2[best]
        if not s1 > 0:
            return nan
        tail = np.exp(s1 * bias[part] + c1)
    te = 1.0 / s1
    vp = (c2 - c1) / (s1 - s2)
    ion = ion_slope * vf + ion_offset
    n = ion / (QE * AREA * np.exp(-0.5) * np.sqrt(QE * te / ion_mass))
    return {
        "vf": vf,
        "te": te,
        "vp": vp,
        "n": n,
        "ion_slope": ion_slope,
        "ion_offset": ion_offset,
        "ln_ies": s1 * vp + c1,
        "knee": (s1, c1, s2, c2),
    }


def sweep(background, radius, start, duration, points, rng):
    """One sawtooth sweep from V_MIN to V_MAX starting at ``start``: bias, current, times."""
    bias = np.linspace(V_MIN, V_MAX, points)
    times = start + duration * np.arange(points) / points
    state = plasma_at(background, radius, times)
    current = probe_current(bias, state, background.ion_mass)
    return bias, current + rng.normal(0.0, NOISE, points), times


def edge_time() -> float:
    """The first time after t = 0 at which a leading edge passes the lab angle 0."""
    # the probe sits at the spoke-frame angle -OMEGA t; the edges at -0.15 + 2 pi k / M
    return (2 * np.pi / M_SPOKES - 0.15) / abs(OMEGA)


def rise_time(t, values, t_from, t_to) -> float:
    """The 10-90 % rise time of ``values`` between ``t_from`` and its maximum before ``t_to``."""
    sel = (t >= t_from) & (t <= t_to)
    t, values = t[sel], values[sel]
    if t.size < 2:
        return np.nan
    top = int(np.argmax(values))
    low = float(np.min(values[: top + 1]))
    span = values[top] - low
    if span <= 0:
        return np.nan
    t, values = t[: top + 1], values[: top + 1]
    crossings = []
    for level in (low + 0.1 * span, low + 0.9 * span):
        k = np.flatnonzero((values[:-1] < level) & (values[1:] >= level))
        if not k.size:
            return np.nan
        k = k[-1]
        crossings.append(
            t[k] + (level - values[k]) * (t[k + 1] - t[k]) / (values[k + 1] - values[k])
        )
    return crossings[1] - crossings[0]


def simulate(preset: str = "quick") -> dict:
    """Record the probe traces, the sweeps at three instants and the fitted series."""
    size = PRESETS[preset]
    start = time.time()
    rng = np.random.default_rng(1)
    background = build_background()
    ion_mass = background.ion_mass
    t_edge = edge_time()
    t0 = t_edge - PERIOD / 3  # the window starts between two spokes
    window = size["periods"] * PERIOD
    t = t0 + np.linspace(0.0, window, int(size["samples"] * size["periods"]) + 1)

    # a probe at a fixed bias: floating potential and saturation currents against the truth
    traces = {}
    for name, radius in PROBES.items():
        state = plasma_at(background, radius, t)
        traces[name] = {
            "phi": state["phi"],
            "n": state["n"],
            "te": state["te"],
            "vf": floating_potential(state, ion_mass),
            "iis": ion_saturation(state, ion_mass),
            "ies": electron_saturation(state),
        }
    racetrack = traces["R0"]
    first = (t >= t_edge - 0.25 * PERIOD) & (t <= t_edge + 0.5 * PERIOD)
    between = (t >= t0) & (t <= t_edge)
    phi = racetrack["phi"]
    t_low = t[between][np.argmin(phi[between])]
    t_peak = t[first][np.argmax(phi[first])]
    half = 0.5 * (phi[between].min() + phi[first].max())
    rising = np.flatnonzero((t >= t_low) & (t <= t_peak) & (phi >= half))
    instants = dict(zip(INSTANTS, (t_low, t[rising[0]], t_peak), strict=True))

    # swept characteristics centred on the three instants at R0, short and long sweeps
    characteristics = []
    for label, t_mid in instants.items():
        for ratio in (SWEEP_RATIOS[0], SWEEP_RATIOS[-1]):
            duration = ratio * PASSAGE
            bias, current, times = sweep(
                background, R0, t_mid - 0.5 * duration, duration, size["points"], rng
            )
            truth = plasma_at(background, R0, t_mid)
            characteristics.append(
                {
                    "instant": label,
                    "ratio": ratio,
                    "t": t_mid,
                    "bias": bias.astype(np.float32),
                    "current": current.astype(np.float32),
                    "fit": fit_characteristic(bias, current, ion_mass),
                    "truth": {k: float(v[0]) for k, v in truth.items()},
                }
            )

    # back-to-back sweeps over the window at every probe, for each sweep time
    series = {}
    for name, radius in PROBES.items():
        for ratio in SWEEP_RATIOS:
            duration = ratio * PASSAGE
            starts = t0 + duration * np.arange(int(window // duration))
            fits = [
                fit_characteristic(
                    *sweep(background, radius, s, duration, size["points"], rng)[:2],
                    ion_mass,
                )
                for s in starts
            ]
            truth = plasma_at(background, radius, starts + 0.5 * duration)
            series[(name, ratio)] = {
                "start": starts,
                "duration": duration,
                **{k: np.array([f[k] for f in fits]) for k in ("vf", "te", "vp", "n")},
                "truth": truth,
            }
    return {
        "preset": preset,
        "t0": t0,
        "t": t,
        "traces": {
            name: {k: v.astype(np.float32) for k, v in trace.items()}
            for name, trace in traces.items()
        },
        "instants": instants,
        "characteristics": characteristics,
        "series": series,
        "measured_passage": rise_time(t, phi, t0, t_peak),
        "wall_time": time.time() - start,
    }


def plot_traces(result: dict) -> None:
    """The fixed-bias signals at the three probes against the background there."""
    us = (result["t"] - result["t0"]) * 1e6
    fig, axes = plt.subplots(
        2, 2, figsize=(7.2, 5.6), sharex=True, constrained_layout=True
    )
    for k, (name, trace) in enumerate(result["traces"].items()):
        color = f"C{k}"
        axes[0, 0].plot(
            us, trace["phi"], color=color, label=f"plasma potential, {name}"
        )
        axes[0, 0].plot(
            us, trace["vf"], color=color, ls="--", label=f"floating potential, {name}"
        )
        axes[0, 1].plot(us, trace["te"], color=color, label=name)
        axes[1, 0].plot(us, trace["n"], color=color, label=name)
        axes[1, 1].plot(
            us, trace["iis"] * 1e3, color=color, label=f"ion saturation, {name}"
        )
        axes[1, 1].plot(
            us,
            trace["ies"] * 1e3,
            color=color,
            ls="--",
            label=f"electron saturation, {name}",
        )
    axes[0, 0].set_title("solid: plasma potential, dashed: floating", fontsize=9)
    axes[0, 0].set_ylabel("potential [V]")
    axes[0, 1].set_title("electron temperature", fontsize=9)
    axes[0, 1].set_ylabel("$T_e$ [eV]")
    axes[0, 1].legend(fontsize=8, loc="upper right")
    axes[1, 0].set_title("density", fontsize=9)
    axes[1, 0].set_ylabel("$n_e$ [m$^{-3}$]")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_title("solid: ion, dashed: electron saturation", fontsize=9)
    axes[1, 1].set_ylabel("probe current [mA]")
    axes[1, 1].set_yscale("log")
    for ax in axes[1]:
        ax.set_xlabel("t [µs]")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)


def plot_characteristics(result: dict) -> None:
    """Sweeps centred on the three instants at R0: the I-V and ln I_e with the fitted lines."""
    fig, axes = plt.subplots(
        3, 2, figsize=(7.2, 8.4), sharex=True, constrained_layout=True
    )
    styles = {SWEEP_RATIOS[0]: ("C0", "o"), SWEEP_RATIOS[-1]: ("C3", "s")}
    for row, label in enumerate(INSTANTS):
        left, right = axes[row]
        title = []
        for item in result["characteristics"]:
            if item["instant"] != label:
                continue
            color, marker = styles[item["ratio"]]
            bias = item["bias"].astype(float)
            current = item["current"].astype(float)
            fit, truth = item["fit"], item["truth"]
            name = f"sweep {item['ratio']:g} × passage"
            left.plot(bias, current * 1e3, marker, color=color, ms=2.5, label=name)
            electron = current + fit["ion_slope"] * bias + fit["ion_offset"]
            keep = electron > 5 * NOISE
            right.plot(
                bias[keep],
                electron[keep] * 1e3,
                marker,
                color=color,
                ms=2.5,
                label=name,
            )
            if np.isfinite(fit["te"]):
                s1, c1, s2, c2 = fit["knee"]
                v = np.linspace(fit["vf"], fit["vp"], 20)
                right.plot(v, np.exp(s1 * v + c1) * 1e3, color=color, lw=1.2)
                v = np.linspace(fit["vp"], V_MAX, 20)
                right.plot(v, np.exp(s2 * v + c2) * 1e3, color=color, lw=1.2)
                right.axvline(fit["vp"], color=color, ls=":", lw=1)
            title.append(
                f"{item['ratio']:g}×: $T_e$ {fit['te']:.1f}, $V_p$ {fit['vp']:.1f}"
            )
        right.axvline(truth["phi"], color="k", ls="--", lw=1, label="true $V_p$")
        left.axvline(truth["phi"], color="k", ls="--", lw=1)
        left.axhline(0.0, color="0.5", lw=0.6)
        left.set_title(
            f"{label}: true $T_e$ {truth['te']:.1f} eV, $V_p$ {truth['phi']:.1f} V",
            fontsize=9,
        )
        right.set_title("fit " + "; ".join(title), fontsize=8)
        left.set_ylabel("probe current [mA]")
        right.set_ylabel("electron current [mA]")
        right.set_yscale("log")
        for ax in (left, right):
            ax.grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=7, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("probe bias [V]")


def staircase(entry: dict, key: str):
    """A fitted series as a staircase over the sweeps it comes from (time [µs], value)."""
    starts = entry["start"]
    x = np.stack([starts, starts + entry["duration"]], axis=1).ravel()
    return x, np.repeat(entry[key], 2)


def plot_series(result: dict) -> None:
    """The fitted n_e, Te and V_p at R0 against time, for the three sweep times, and the truth."""
    t0 = result["t0"]
    us = (result["t"] - t0) * 1e6
    truth = result["traces"]["R0"]
    fig, axes = plt.subplots(
        3, 1, figsize=(7.2, 7.0), sharex=True, constrained_layout=True
    )
    rows = [
        ("n", "n", "$n_e$ [m$^{-3}$]"),
        ("te", "te", "$T_e$ [eV]"),
        ("vp", "phi", "$V_p$ [V]"),
    ]
    for ax, (key, true_key, label) in zip(axes, rows, strict=True):
        ax.plot(us, truth[true_key], color="k", lw=1.6, label="background at the probe")
        for k, ratio in enumerate(SWEEP_RATIOS):
            entry = result["series"][("R0", ratio)]
            x, y = staircase(entry, key)
            ax.plot(
                (x - t0) * 1e6,
                y,
                color=f"C{k}",
                lw=1.1,
                label=f"sweep {entry['duration'] * 1e6:.3g} µs = {ratio:g} × passage",
            )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[-1].set_xlabel("t [µs]")


def plot(result: dict) -> None:
    """Draw the fixed-bias traces, the swept characteristics and the fitted series."""
    plot_traces(result)
    plot_characteristics(result)
    plot_series(result)


def errors(entry: dict) -> dict:
    """The errors of a fitted series against the background at the middle of each sweep."""
    truth = entry["truth"]
    ok = np.isfinite(entry["vp"])
    return {
        "n": np.median(np.abs(entry["n"][ok] / truth["n"][ok] - 1)),
        "te": np.median(np.abs(entry["te"][ok] / truth["te"][ok] - 1)),
        "vp": np.sqrt(np.mean((entry["vp"][ok] - truth["phi"][ok]) ** 2)),
        "hump": np.ptp(entry["vp"][ok]) if ok.any() else np.nan,
        "failed": int((~ok).sum()),
        "sweeps": ok.size,
    }


def apparent_rise(entry: dict, t0: float) -> float:
    """The 10-90 % rise time of the fitted V_p over the first leading edge (sweep midpoints)."""
    mid = entry["start"] + 0.5 * entry["duration"]
    ok = np.isfinite(entry["vp"])
    t_edge = edge_time()
    return rise_time(mid[ok], entry["vp"][ok], t0, t_edge + 0.25 * PERIOD)


def rise_text(seconds: float) -> str:
    """A rise time in µs, or why there is none."""
    return (
        f"{seconds * 1e6:.2f} µs"
        if np.isfinite(seconds)
        else "fewer than two sweeps on it"
    )


def table(result: dict) -> str:
    """The probe and the passage time, and the errors of the fits for each sweep time."""
    from byob import debye_length, sheath_width

    lines = [
        "| | value |",
        "|---|---:|",
        f"| spokes, speed at R0, period | {M_SPOKES}, {SPOKE_SPEED / 1e3:.0f} km/s, "
        f"{PERIOD * 1e6:.2f} µs |",
        f"| leading edge 10-90 % width at R0 | {PASSAGE * SPOKE_SPEED * 1e3:.2f} mm |",
        f"| passage time of the edge: width / speed (from the trace) | {PASSAGE * 1e6:.3f} µs "
        f"({result['measured_passage'] * 1e6:.3f} µs) |",
        f"| probe: planar disc, radius, area | {PROBE_RADIUS * 1e3:.1f} mm, {AREA * 1e6:.2f} mm² |",
    ]
    # the sheath at the most negative bias, over the probes and the window
    widths, debyes = [], []
    for trace in result["traces"].values():
        n, te, phi = (trace[k].astype(float) for k in ("n", "te", "phi"))
        ns = n * np.exp(-0.5)
        debyes.append(debye_length(ns, te))
        widths.append(sheath_width(phi - 0.5 * te - V_MIN, ns, te))
    debyes, widths = np.concatenate(debyes), np.concatenate(widths)
    r_larmor = np.sqrt(ME * QE * TE0) / (QE * B0)
    lines += [
        f"| Debye length at the sheath edge | {debyes.min() * 1e6:.0f}-{debyes.max() * 1e6:.0f} µm |",
        f"| sheath width at {V_MIN:.0f} V / probe radius (byob) | "
        f"{widths.min() / PROBE_RADIUS:.2f}-{widths.max() / PROBE_RADIUS:.2f} |",
        f"| electron Larmor radius at {TE0:.0f} eV, {B0 * 1e3:.0f} mT / probe radius | "
        f"{r_larmor / PROBE_RADIUS:.2f} (not included) |",
        f"| wall time | {result['wall_time']:.1f} s |",
        "",
        "Fits at R0 against the background at the middle of each sweep (median relative error "
        "of $n_e$ and $T_e$, rms error of $V_p$):",
        "",
    ]
    stats = {ratio: errors(result["series"][("R0", ratio)]) for ratio in SWEEP_RATIOS}
    header = " | ".join(f"{r:g} × passage" for r in SWEEP_RATIOS)
    lines += [f"| sweep time | {header} |", "|---" + "|---:" * len(SWEEP_RATIOS) + "|"]

    def row(label, values):
        return f"| {label} | " + " | ".join(values) + " |"

    entries = [result["series"][("R0", r)] for r in SWEEP_RATIOS]
    lines += [
        row("sweep time", [f"{e['duration'] * 1e6:.3g} µs" for e in entries]),
        row(
            "sweeps (failed fits)",
            [f"{s['sweeps']} ({s['failed']})" for s in stats.values()],
        ),
        row("$n_e$ error", [f"{100 * s['n']:.1f} %" for s in stats.values()]),
        row("$T_e$ error", [f"{100 * s['te']:.1f} %" for s in stats.values()]),
        row("$V_p$ rms error", [f"{s['vp']:.2f} V" for s in stats.values()]),
        row(
            "rise time of $V_p$ at the edge (true "
            f"{result['measured_passage'] * 1e6:.2f} µs)",
            [rise_text(apparent_rise(e, result["t0"])) for e in entries],
        ),
    ]
    for name in PROBES:
        true = np.ptp(result["traces"][name]["phi"].astype(float))
        lines.append(
            row(
                f"$V_p$ hump at {name} (true {true:.1f} V)",
                [
                    f"{errors(result['series'][(name, r)])['hump']:.1f} V"
                    for r in SWEEP_RATIOS
                ],
            )
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    args = parser.parse_args()
    outcome = simulate(args.preset)
    print(table(outcome))
    plot(outcome)
    plt.show()
