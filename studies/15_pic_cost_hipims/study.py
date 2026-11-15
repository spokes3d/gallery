"""PIC cost against density.

How far can an explicit particle-in-cell simulation of a magnetron get? Its cost is set by
three resolution limits. The cell size must resolve the Debye length
$\\lambda_D = \\sqrt{\\varepsilon_0 T_e / n e^2}$ (else the grid heats the electrons) and, to
represent the electron-scale waves and the gyration in the strong field near the cathode,
the thermal electron gyroradius $\\rho_e = \\sqrt{m_e T_e}/(eB)$: $\\Delta x = \\min(\\lambda_D,
\\rho_e)$. The time step must resolve the plasma oscillation, $\\omega_{pe}\\Delta t \\le 0.2$,
and the gyration in the strongest field, $\\omega_{ce}\\Delta t \\le 0.4$. And every cell needs
enough markers (here 100 per species) for the noise to stay below the physics.

The two limits on the time step cross: at the density of a DC magnetron in Boeuf's
simulation ($2.5\\times10^{15}$ m$^{-3}$) the gyration at 120 mT sets $\\Delta t \\approx 2\\times10^{-11}$ s,
so 20 µs take $10^6$ steps whatever the density; above about $10^{17}$ m$^{-3}$ the plasma
frequency takes over. The cell count grows like $n^{d/2}$ in $d$ dimensions once the Debye
length is the smaller scale, and the steps like $n^{1/2}$: at HiPIMS density
($10^{18}$ to $10^{19}$ m$^{-3}$) the Debye length is a few µm and a 3D run of the ionization
region is out of reach of any allocation, while a 2D slab of a few cm² is a large GPU job.

The cost is markers × steps × the cost of one marker step. That last number is an assumption
here (100 ns on one CPU core, 2 ns on one GPU, including the deposit, gather and collisions;
the field solve is not counted) until the benchmark studies measure it for the codes we use.
The domains: a 2D axial-azimuthal slab of 2 cm × 8 cm (Boeuf's), a 3D sector of 2 cm × 2 cm
× 2 cm, and the full annulus of a 2-inch target (racetrack radius 1.6 cm, 2 cm × 2 cm × 10 cm).

Compared: the cell size and time step these rules give for the two published 2D cases against
the ones their authors chose (table). Not compared: measured run times; the cost per marker
step and the markers per cell are assumptions, implicit or energy-conserving schemes that relax
$\\Delta x < \\lambda_D$ and $\\omega_{pe}\\Delta t < 0.2$ are not considered, nor is mesh refinement
(which helps only where the density is low).

References: Boeuf, arXiv:2211.17051 (the DC magnetron case); Charoy et al., Plasma Sources
Sci. Technol. 28 (2019) 105010 (the axial-azimuthal benchmark).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

PRESETS = {
    "smoke": {"grid": 8},
    "full": {"grid": 90},
}

QE = 1.602176634e-19
ME = 9.1093837015e-31
EPS0 = 8.8541878128e-12

#: The resolution rules and the assumed costs.
WPE_DT = 0.2  # omega_pe dt
WCE_DT = 0.4  # omega_ce dt
MARKERS_PER_CELL = 100  # per species
SPECIES = 2  # electrons and argon ions
CPU_SECONDS_PER_MARKER_STEP = 100e-9  # one core
GPU_SECONDS_PER_MARKER_STEP = 2e-9  # one GPU
TE = 5.0  # electron temperature of the scans [eV]
B_CATHODE = 0.12  # strongest field, at the cathode [T]
WINDOW = 20e-6  # simulated time [s]

#: The domains [m]: the lengths along each simulated axis.
DOMAINS = {
    "2D slab, 2 × 8 cm": (0.02, 0.08),
    "3D sector, 2 × 2 × 2 cm": (0.02, 0.02, 0.02),
    "3D annulus, 2-inch target": (0.02, 0.02, 2 * np.pi * 0.016),
}

#: Published 2D cases: density and temperature where the resolution is tightest, field,
#: domain, simulated time, and the authors' cell size and time step.
CASES = {
    "Boeuf DC magnetron": {
        "density": 2.5e15,
        "te": 10.0,
        "b": 0.12,
        "domain": (0.02, 0.08),
        "window": 20e-6,
        "dx": 0.08 / 1024,
        "dt": 2e-11,
    },
    "Charoy benchmark": {
        # the peak density of the steady state at 50 eV gives the 70 µm Debye length the
        # benchmark's stability limit refers to
        "density": 5.6e17,
        "te": 50.0,
        "b": 0.01,
        "domain": (0.025, 0.0128),
        "window": 20e-6,
        "dx": 5e-5,
        "dt": 5e-12,
    },
}

#: Density ranges of the regimes [m^-3].
DC_RANGE = (1e15, 1e17)
HIPIMS_RANGE = (1e18, 1e19)


def debye_length(density, te):
    """The electron Debye length [m] at ``density`` [m^-3] and ``te`` [eV]."""
    return np.sqrt(EPS0 * te / (density * QE))


def gyroradius(te, b):
    """The thermal electron gyroradius [m] at ``te`` [eV] in the field ``b`` [T]."""
    return np.sqrt(ME * te * QE) / (QE * b)


def plasma_frequency(density):
    """The electron plasma frequency [rad/s]."""
    return np.sqrt(density * QE**2 / (EPS0 * ME))


def cyclotron_frequency(b):
    """The electron cyclotron frequency [rad/s]."""
    return QE * b / ME


def requirements(density, te, b, lengths, window):
    """Cell size, time step, cells, steps, markers and cost of one run.

    Broadcasts over ``density``, ``te`` and ``b``. Returns a dict of arrays; the costs are in
    core-hours and GPU-hours.
    """
    dx = np.minimum(debye_length(density, te), gyroradius(te, b))
    dt_pe = WPE_DT / plasma_frequency(density)
    dt_ce = WCE_DT / cyclotron_frequency(b)
    dt = np.minimum(dt_pe, dt_ce)
    cells = np.prod([np.ceil(length / dx) for length in lengths], axis=0)
    steps = np.ceil(window / dt)
    markers = SPECIES * MARKERS_PER_CELL * cells
    marker_steps = markers * steps
    return {
        "dx": dx,
        "dt": dt,
        "gyration_limited": dt_ce < dt_pe,
        "dt_ratio": dt_ce / dt_pe,
        "cells": cells,
        "steps": steps,
        "markers": markers,
        "core_hours": marker_steps * CPU_SECONDS_PER_MARKER_STEP / 3600,
        "gpu_hours": marker_steps * GPU_SECONDS_PER_MARKER_STEP / 3600,
    }


def reachable_time(density, lengths, budget_seconds, seconds_per_marker_step):
    """The simulated time [s] a budget buys at ``density`` (TE, B_CATHODE)."""
    per_step = requirements(density, TE, B_CATHODE, lengths, window=1.0)
    markers_steps_per_second = per_step["markers"] / per_step["dt"]
    return budget_seconds / (seconds_per_marker_step * markers_steps_per_second)


def simulate(preset: str = "full") -> dict:
    """The requirements against density, field and domain, and the published cases."""
    size = PRESETS[preset]["grid"]
    densities = np.geomspace(1e15, 1e19, size)
    fields = np.geomspace(0.005, 0.3, size)
    by_domain = {
        name: requirements(densities, TE, B_CATHODE, lengths, WINDOW)
        for name, lengths in DOMAINS.items()
    }
    n, b = np.meshgrid(densities, fields, indexing="ij")
    step_map = requirements(n, TE, b, DOMAINS["2D slab, 2 × 8 cm"], WINDOW)
    budgets = {
        "1e6 core-hours": (1e6 * 3600, CPU_SECONDS_PER_MARKER_STEP),
        "1e4 GPU-hours": (1e4 * 3600, GPU_SECONDS_PER_MARKER_STEP),
    }
    reach = {
        (domain, budget): reachable_time(densities, lengths, seconds, per_marker_step)
        for domain, lengths in DOMAINS.items()
        for budget, (seconds, per_marker_step) in budgets.items()
    }
    cases = {}
    for name, case in CASES.items():
        rules = requirements(
            case["density"], case["te"], case["b"], case["domain"], case["window"]
        )
        published_cells = np.prod([np.ceil(x / case["dx"]) for x in case["domain"]])
        cases[name] = {
            **case,
            "rule_dx": float(rules["dx"]),
            "rule_dt": float(rules["dt"]),
            "rule_cells": float(rules["cells"]),
            "rule_steps": float(rules["steps"]),
            "published_cells": float(published_cells),
            "published_steps": float(np.ceil(case["window"] / case["dt"])),
            "rule_core_hours": float(rules["core_hours"]),
        }
    return {
        "densities": densities,
        "fields": fields,
        "by_domain": by_domain,
        "step_map": {key: step_map[key] for key in ("steps", "dt_ratio")},
        "reach": {f"{domain} | {budget}": t for (domain, budget), t in reach.items()},
        "cases": cases,
    }


def shade_regimes(ax):
    """The density ranges of DC and HiPIMS magnetrons."""
    ax.axvspan(*DC_RANGE, color="C0", alpha=0.08, lw=0)
    ax.axvspan(*HIPIMS_RANGE, color="C3", alpha=0.08, lw=0)
    top = ax.get_ylim()[1]
    ax.text(np.sqrt(np.prod(DC_RANGE)), top, "DC", ha="center", va="top")
    ax.text(np.sqrt(np.prod(HIPIMS_RANGE)), top, "HiPIMS", ha="center", va="top")


def plot(result: dict) -> None:
    """Draw the four figures of the study."""
    densities = result["densities"]

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for i, (name, req) in enumerate(result["by_domain"].items()):
        ax.plot(densities, req["core_hours"], color=f"C{i}", label=f"{name}, CPU")
        ax.plot(
            densities, req["gpu_hours"], color=f"C{i}", ls="--", label=f"{name}, GPU"
        )
    for level, label in ((1e5, "a typical allocation"), (1e8, "a national machine")):
        ax.axhline(level, color="0.4", lw=0.8, ls=":")
        ax.text(1.2e15, level * 1.4, label, fontsize=8, color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e1, 1e12)
    shade_regimes(ax)
    ax.set_xlabel("plasma density n [m$^{-3}$]")
    ax.set_ylabel("cost of 20 µs [core-hours or GPU-hours]")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3, which="major")
    fig.caption = (
        "The cost of simulating 20 µs against the plasma density, for the three domains, "
        f"at T_e = {TE:g} eV and {B_CATHODE * 1e3:g} mT at the cathode: core-hours (solid, "
        f"{CPU_SECONDS_PER_MARKER_STEP * 1e9:g} ns per marker step) and GPU-hours (dashed, "
        f"{GPU_SECONDS_PER_MARKER_STEP * 1e9:g} ns). Shaded: the densities of DC and HiPIMS "
        "magnetrons."
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    fields_mt = result["fields"] * 1e3
    steps = result["step_map"]["steps"]
    image = ax.pcolormesh(
        densities, fields_mt, np.log10(steps).T, shading="auto", cmap="viridis"
    )
    ax.contour(
        densities,
        fields_mt,
        np.log10(result["step_map"]["dt_ratio"]).T,
        levels=[0.0],
        colors="white",
        linewidths=1.2,
    )
    for name, case in result["cases"].items():
        ax.plot(case["density"], case["b"] * 1e3, "w*", ms=10)
        ax.annotate(
            name,
            (case["density"], case["b"] * 1e3),
            xytext=(6, 4),
            textcoords="offset points",
            color="white",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("plasma density n [m$^{-3}$]")
    ax.set_ylabel("field at the cathode B [mT]")
    fig.colorbar(image, label="log10 of the steps for 20 µs")
    fig.caption = (
        "The time steps for 20 µs against the density and the strongest field. White line: "
        "where the two limits are equal; above it (strong field, low density) the electron "
        "gyration sets the step, below it the plasma frequency. Stars: the two published cases "
        "(Charoy's at its 50 eV, the map at 5 eV)."
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.8), constrained_layout=True)
    req = next(iter(result["by_domain"].values()))
    left.plot(densities, debye_length(densities, TE) * 1e6, label="Debye length")
    left.axhline(
        gyroradius(TE, B_CATHODE) * 1e6,
        color="C1",
        label=f"gyroradius at {B_CATHODE * 1e3:g} mT",
    )
    left.plot(densities, req["dx"] * 1e6, "k--", lw=1, label="cell size")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("n [m$^{-3}$]")
    left.set_ylabel("length [µm]")
    left.legend(fontsize=7)
    left.grid(True, alpha=0.3)
    for i, (name, req) in enumerate(result["by_domain"].items()):
        right.plot(densities, req["markers"], color=f"C{i}", label=name)
    right.axhline(1e10, color="0.4", lw=0.8, ls=":")
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("n [m$^{-3}$]")
    right.set_ylabel("markers")
    right.legend(fontsize=7)
    right.grid(True, alpha=0.3)
    fig.caption = (
        "Left: the Debye length, the electron gyroradius at the cathode and the cell size "
        "(the smaller of the two). Right: the markers of a run, "
        f"{MARKERS_PER_CELL} per cell and species; dotted: 10^10, about what fits in the "
        "memory of a few GPUs."
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    for key, reach in result["reach"].items():
        domain, budget = key.split(" | ")
        i = list(result["by_domain"]).index(domain)
        ax.plot(
            densities,
            reach * 1e6,
            color=f"C{i}",
            ls="-" if "core" in budget else "--",
            label=f"{domain}, {budget}",
        )
    for t, label in ((10.0, "a spoke period, ~10 µs"), (100.0, "a HiPIMS pulse")):
        ax.axhline(t, color="0.4", lw=0.8, ls=":")
        ax.text(1.2e15, t * 1.5, label, fontsize=8, color="0.3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e5)
    shade_regimes(ax)
    ax.set_xlabel("plasma density n [m$^{-3}$]")
    ax.set_ylabel("simulated time [µs]")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3, which="major")
    fig.caption = (
        "The simulated time a budget buys against the density: a million core-hours (solid) "
        "or ten thousand GPU-hours (dashed), for each domain."
    )


def table(result: dict) -> str:
    """The rules against the published cases, and the cost at the regimes' densities."""
    rows = [
        "| | Δx rule / published [µm] | Δt rule / published [ps] | cells rule / published | steps rule / published |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, case in result["cases"].items():
        rows.append(
            f"| {name} | {case['rule_dx'] * 1e6:.0f} / {case['dx'] * 1e6:.0f} "
            f"| {case['rule_dt'] * 1e12:.0f} / {case['dt'] * 1e12:.0f} "
            f"| {case['rule_cells']:.1e} / {case['published_cells']:.1e} "
            f"| {case['rule_steps']:.1e} / {case['published_steps']:.1e} |"
        )
    rows += [
        "",
        "| domain | 20 µs at 1e16 m⁻³ | at 1e18 m⁻³ | at 1e19 m⁻³ |",
        "|---|---:|---:|---:|",
    ]
    densities = result["densities"]
    for name, req in result["by_domain"].items():
        cells = []
        for n in (1e16, 1e18, 1e19):
            j = int(np.argmin(np.abs(np.log(densities / n))))
            cells.append(
                f"{req['core_hours'][j]:.1e} core-h, {req['gpu_hours'][j]:.1e} GPU-h"
            )
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    outcome = simulate(args.preset)
    print(table(outcome))
    plot(outcome)
    plt.show()
