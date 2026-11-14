"""Design optimisation of an einzel lens: a brute-force scan and the optimiser's path.

A check of beamtracer's design optimisation (`beamtracer.optimization`) on the einzel lens of
`beamtracer.optimization.einzel`, before it designs extraction grids: three coaxial tubes, the
outer two grounded, the middle one at the voltage $V$, focusing a parallel 30 keV proton beam of
2 mm radius; the objective is the rms spot $\\sqrt{\\langle r^2 \\rangle / 2}$ at a plane 80 mm
behind the lens, every design a `BeamOptics` solve (field and trajectories) on a 0.5 mm mesh.
Differential evolution (SciPy) is the optimiser; it never sees the scan.

1. **One parameter, the middle voltage.** A brute-force scan over the whole range of the search
   maps the objective, and the optimiser's evaluations are drawn on it, coloured by the
   generation that asked for them.
   The thin-lens estimate, $1/f = \\frac{3}{16}\\int (\\phi'/\\phi)^2\\,dx$ on the solved axis
   potential ($\\phi$ the protons' kinetic energy per charge), and the paraxial ray integrated
   through the lens are the voltages that focus on the target without aberrations.
2. **Three parameters** (voltage, gap between the tubes, bore radius) with the largest field on
   the electrode surfaces limited to 5 MV/m: the optimiser's evaluations and path, projected on
   two planes.
3. **Space charge.** With a beam current the field and the trajectories are iterated to a
   self-consistent solution (Vlasov iteration); scans and searches at 0, 1 and 3 mA show the
   optimum moving away from the thin-lens one, which knows nothing of space charge.

**Finding.** In one parameter the optimiser lands on the scan's minimum: 14 924 V against
14 923 V from the parabola through the 50 V scan, a 28.4 µm spot, with 121 solves in 15
generations, as many as the scan. The thin-lens and paraxial voltages are 1.7 % and 3.2 %
higher: the 2 mm beam fills 40 % of the bore, its outer rays are focused more strongly
(spherical aberration), and the smallest rms spot lies before the paraxial focus. With three
parameters and the 5 MV/m limit, about 380 solves give 14.35 kV, a 9.2 mm gap and a 3.46 mm bore
(51 µm), near the longer searches of beamtracer's tutorial (14.5 kV, 9.6 to 10 mm, 3.46 mm,
36 µm): the path runs along the field limit, and this short search has not settled the gap.
Space charge defocuses the beam, so the lens needs more voltage: the optimum moves by +9 % at
1 mA and +22 % at 3 mA, where the spot is six times larger, and the scans' minima agree with the
optimiser's to 30 V.

Compared: the optimiser's optimum with the scan's minimum (the scan's step, and a parabola
through $\\mathrm{spot}^2$ at the five scan points around its minimum), and both with the
thin-lens and paraxial voltages. Not compared: a measurement or another code; the 0.5 mm mesh
underestimates the surface field by 10 to 15 % (see beamtracer's tutorial), so the 5 MV/m limit
of step 2 is not a design margin.

Every evaluation is kept as an `xarray.Dataset` (`beamtracer.optimization.to_dataset`). The
same lens with the gap as a second parameter, mapped on a grid with the search's path over it,
is beamtracer's own example `examples/einzel_lens_design_map.py`.

Reference: the einzel lens problem and its thin-lens estimate in `beamtracer.optimization.einzel`
(after M. Reiser, *Theory and Design of Charged Particle Beams*, Wiley 2008).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import plasma_plots
import xarray as xr
from beamtracer.optimization import optimize, scan
from beamtracer.optimization.einzel import EinzelLens

#: scan: points of the one-parameter scan; popsize, maxiter: differential evolution in one
#: parameter; three: the same for the three-parameter search; currents [A], sc_scan,
#: sc_popsize and sc_maxiter: the space-charge scans and searches; workers: processes.
PRESETS = {
    "smoke": {
        "scan": 9,
        "popsize": 4,
        "maxiter": 2,
        "three": {"popsize": 2, "maxiter": 1},
        "currents": [0.0, 1e-3],
        "sc_scan": 3,
        "sc_popsize": 2,
        "sc_maxiter": 1,
        "workers": 1,
    },
    "full": {
        "scan": 121,
        "popsize": 8,
        "maxiter": 30,
        "three": {"popsize": 6, "maxiter": 20},
        "currents": [0.0, 1e-3, 3e-3],
        "sc_scan": 33,
        "sc_popsize": 6,
        "sc_maxiter": 12,
        "workers": 4,
    },
}

SEED = 1
#: The search range of the one-parameter problem, scanned completely [V].
VOLTAGE = (12e3, 18e3)
#: The three-parameter design: voltage [V], gap and bore radius [m].
BOUNDS = {"voltage": (5e3, 25e3), "gap": (1e-3, 12e-3), "aperture": (3e-3, 8e-3)}
#: The voltage range of the space-charge scans and searches [V].
SPACE_CHARGE = (13e3, 21e3)


def search(problem, popsize, maxiter, workers, tol=1e-6):
    """Differential evolution: its evaluations, and the best design after each generation.

    The evaluations are each design once, in the order first asked for, with the generation
    that asked for it; the path starts at the initial population (generation 0).
    """
    start = time.perf_counter()
    result = optimize(
        problem, seed=SEED, popsize=popsize, maxiter=maxiter, tol=tol, workers=workers
    )
    wall = time.perf_counter() - start
    evaluations = result.to_dataset()
    steps = result.path_to_dataset()
    evaluations.attrs.update(
        wall_seconds=wall,
        evaluated=result.evaluated,
        repeats=result.repeats,
        generations=len(result.path) - 1,
        optimum_spot=result.fun,
        **{f"optimum_{k}": v for k, v in result.x.items()},
    )
    return evaluations, steps


def swept(problem, voltages, workers):
    """The scan of the voltage as a dataset along ``voltage``, with its wall time."""
    start = time.perf_counter()
    result = scan(problem, {"voltage": voltages}, workers=workers)
    data = result.to_dataset().swap_dims(evaluation="voltage")
    data.attrs.update(
        wall_seconds=time.perf_counter() - start, evaluated=result.evaluated
    )
    return data.set_coords("voltage")


def refined_minimum(data):
    """The voltage of the scan's minimum: a parabola through spot^2 at five points."""
    k = int(np.argmin(data.spot.values))
    window = slice(max(k - 2, 0), min(k + 3, data.sizes["voltage"]))
    v, s = data.voltage.values[window], data.spot.values[window]
    if v.size < 3:
        return float(data.voltage[k])
    a, b, _ = np.polyfit(v, s**2, 2)
    return float(-b / (2 * a)) if a > 0 else float(data.voltage[k])


def simulate(preset: str = "full") -> dict:
    """Scan and search in one parameter, search in three, and scan and search with space charge."""
    p = PRESETS[preset]
    workers = p["workers"]
    start = time.time()
    lens = EinzelLens()  # 30 keV protons, 2 mm beam, 5 mm bore, 4 mm gaps, 0.5 mm mesh

    # 1. one parameter, without space charge
    problem = lens.problem({"voltage": VOLTAGE})
    one_scan = swept(problem, np.linspace(*VOLTAGE, p["scan"]), workers)
    one_scan.attrs["refined_voltage"] = refined_minimum(one_scan)
    one_search, one_path = search(problem, p["popsize"], p["maxiter"], workers)
    estimates = {
        "thin_lens": lens.thin_lens_voltage(),
        "paraxial": lens.paraxial_voltage(),
    }

    # 2. three parameters with the surface-field limit
    limited = lens.problem(BOUNDS, constraints=("loss", "surface_field"))
    three_search, three_path = search(limited, workers=workers, tol=1e-4, **p["three"])

    # 3. space charge
    voltages = np.linspace(*SPACE_CHARGE, p["sc_scan"])
    scans, optima = [], []
    for current in p["currents"]:
        charged = replace(lens, current=current)
        problem = charged.problem({"voltage": SPACE_CHARGE})
        scans.append(swept(problem, voltages, workers))
        if current == 0.0:  # the search of step 1, the same lens and objective
            optima.append(one_search.attrs)
            continue
        found, _ = search(problem, p["sc_popsize"], p["sc_maxiter"], workers)
        optima.append(found.attrs)
    currents = xr.DataArray(
        p["currents"], dims="current", attrs={"units": "A", "long_name": "beam current"}
    )
    keep = ["spot", "field", "transmission", "iterations", "converged", "wall_time"]
    space_charge = xr.concat(
        [s[keep] for s in scans], dim=currents, coords="minimal"
    ).assign(
        optimum_voltage=("current", [o["optimum_voltage"] for o in optima]),
        optimum_spot=("current", [o["optimum_spot"] for o in optima]),
        evaluated=("current", [o["evaluated"] for o in optima]),
        search_seconds=("current", [o["wall_seconds"] for o in optima]),
    )
    return {
        "preset": preset,
        "lens": {
            "energy": lens.energy,
            "beam_radius": lens.beam_radius,
            "target": lens.target,
            "field_limit": lens.field_limit,
            "cell": lens.cell,
        },
        "one_scan": one_scan,
        "one_search": one_search,
        "one_path": one_path,
        "estimates": estimates,
        "three_search": three_search,
        "three_path": three_path,
        "space_charge": space_charge,
        "runtime": time.time() - start,
    }


def table(result: dict) -> str:
    """The scan's minimum, the optimiser's optimum and the estimates, with their costs."""
    scan_, found = result["one_scan"], result["one_search"].attrs
    k = int(np.argmin(scan_.spot.values))
    step = float(scan_.voltage[1] - scan_.voltage[0])
    v_opt = found["optimum_voltage"]
    lines = [
        "| One parameter | voltage (V) | rms spot (um) | evaluations | wall time (s) |",
        "|---|---|---|---|---|",
        f"| brute-force scan, minimum ({step:.0f} V steps) | {float(scan_.voltage[k]):.0f} | "
        f"{1e6 * float(scan_.spot[k]):.1f} | {scan_.attrs['evaluated']} | "
        f"{scan_.attrs['wall_seconds']:.0f} |",
        f"| scan, parabola through spot^2 | {scan_.attrs['refined_voltage']:.0f} | - | - | - |",
        f"| differential evolution, optimum | {v_opt:.0f} | "
        f"{1e6 * found['optimum_spot']:.1f} | {found['evaluated']} "
        f"({found['generations']} generations) | {found['wall_seconds']:.0f} |",
    ]
    for name, label in (("thin_lens", "thin lens"), ("paraxial", "paraxial ray")):
        v = result["estimates"][name]
        lines.append(f"| {label} ({v / v_opt - 1:+.1%}) | {v:.0f} | - | - | - |")
    three = result["three_search"].attrs
    lines += [
        "",
        "| Three parameters, 5 MV/m | voltage (kV) | gap (mm) | bore (mm) | rms spot (um) "
        "| evaluations | wall time (s) |",
        "|---|---|---|---|---|---|---|",
        f"| differential evolution | {three['optimum_voltage'] / 1e3:.2f} | "
        f"{three['optimum_gap'] * 1e3:.2f} | {three['optimum_aperture'] * 1e3:.2f} | "
        f"{three['optimum_spot'] * 1e6:.1f} | {three['evaluated']} | "
        f"{three['wall_seconds']:.0f} |",
        "",
        "| Space charge, current (mA) | scan minimum, parabola (V) | optimum (V) | shift from 0 mA "
        "| optimum spot (um) | evaluations |",
        "|---|---|---|---|---|---|",
    ]
    sc = result["space_charge"]
    v0 = float(sc.optimum_voltage[0])
    for i in range(sc.sizes["current"]):
        row = sc.isel(current=i)
        v = float(row.optimum_voltage)
        lines.append(
            f"| {1e3 * float(row.current):.0f} | {refined_minimum(row):.0f} | {v:.0f} | "
            f"{v / v0 - 1:+.1%} | {1e6 * float(row.optimum_spot):.1f} | "
            f"{int(row.evaluated)} |"
        )
    return "\n".join(lines)


def micro(data: xr.DataArray) -> xr.DataArray:
    """The spot in micrometres against the voltage in kilovolts, labelled for the plots."""
    out = (1e6 * data).assign_attrs(units="µm", long_name="rms spot at the target")
    return out.assign_coords(
        voltage=(1e-3 * out.voltage).assign_attrs(
            units="kV", long_name="middle voltage"
        )
    )


def plot(result: dict) -> None:
    """The scan and the search, the path, the three-parameter search and space charge."""
    scan_, found, path = result["one_scan"], result["one_search"], result["one_path"]
    thin, paraxial = (result["estimates"][k] / 1e3 for k in ("thin_lens", "paraxial"))
    v_opt = found.attrs["optimum_voltage"] / 1e3
    s_opt = found.attrs["optimum_spot"] * 1e6
    ok = found.feasible.values
    order = found.evaluation.values + 1
    generation = found.generation.values

    # 1. the scan and the optimiser's evaluations
    with plasma_plots.figure(1, 2, figsize=(11, 4)) as fig:
        for ax, zoom in zip((fig[0], fig[1]), (False, True), strict=True):
            micro(scan_.spot).plasma.plot.lineout(x="voltage", ax=ax)
            points = ax.scatter(
                found.voltage[ok] / 1e3,
                found.spot[ok] * 1e6,
                c=generation[ok],
                cmap="viridis",
                s=16,
                zorder=3,
                label="optimiser's evaluations",
            )
            ax.scatter(
                v_opt,
                s_opt,
                marker="*",
                s=200,
                color="tab:red",
                edgecolor="white",
                zorder=4,
                label="optimiser's optimum",
            )
            ax.axvline(thin, color="tab:green", ls="--", label="thin lens")
            ax.axvline(paraxial, color="gray", ls=":", label="paraxial ray")
            if zoom:
                ax.set_xlim(v_opt - 0.5, v_opt + 0.7)
                ax.set_ylim(0, 4 * s_opt + 20)
                ax.set_title("near the minimum")
                ax.legend(fontsize=8, loc="upper right")
            else:
                ax.set_yscale("log")
                ax.set_title("the brute-force scan and the optimiser's evaluations")
        plt.colorbar(points, ax=fig[1], label="generation")

    # 2. the path to the minimum
    step = float(scan_.voltage[1] - scan_.voltage[0]) / 1e3
    refined = scan_.attrs["refined_voltage"] / 1e3
    with plasma_plots.figure(1, 2, figsize=(11, 3.6)) as fig:
        ax = fig[0]
        ax.plot(path.generation, path.voltage / 1e3, marker="o", label="best so far")
        ax.axhspan(refined - step, refined + step, color="tab:blue", alpha=0.15)
        ax.axhline(refined, color="tab:blue", ls="--", label="scan's minimum")
        ax.axhline(thin, color="tab:green", ls="--", label="thin lens")
        ax.set_xlabel("generation")
        ax.set_ylabel("middle voltage [kV]")
        ax.legend(fontsize=8)
        ax = fig[1]
        spots = np.where(ok, found.spot.values, np.inf)
        ax.plot(
            order,
            np.minimum.accumulate(spots) * 1e6,
            label="optimiser, best so far",
        )
        ax.axhline(
            float(scan_.spot.min()) * 1e6,
            color="tab:blue",
            ls="--",
            label=f"scan's minimum ({scan_.attrs['evaluated']} evaluations)",
        )
        ax.set_xlabel("evaluations")
        ax.set_ylabel("rms spot [µm]")
        ax.legend(fontsize=8)

    # 3. three parameters: the projections of the search
    three, steps = result["three_search"], result["three_path"]
    inside = three.feasible.values
    usable = three.ok.values
    norm = plt.Normalize(
        np.log10(three.spot[usable].min() * 1e6),
        np.log10(np.percentile(three.spot[usable] * 1e6, 90)),
    )
    with plasma_plots.figure(1, 2, figsize=(11, 4)) as fig:
        for ax, (name, label) in zip(
            (fig[0], fig[1]),
            (("gap", "gap between the tubes [mm]"), ("aperture", "bore radius [mm]")),
            strict=True,
        ):
            for mask, face in ((usable & inside, True), (usable & ~inside, False)):
                colours = plt.cm.viridis_r(norm(np.log10(three.spot[mask] * 1e6)))
                ax.scatter(
                    three.voltage[mask] / 1e3,
                    three[name][mask] * 1e3,
                    s=18,
                    facecolors=colours if face else "none",
                    edgecolors=colours,
                    label="within 5 MV/m" if face else "above 5 MV/m",
                )
            ax.plot(
                steps.voltage / 1e3,
                steps[name] * 1e3,
                color="tab:red",
                lw=2,
                marker="o",
                ms=4,
                mec="white",
                label="best after each generation",
            )
            ax.scatter(
                steps.voltage[0] / 1e3,
                steps[name][0] * 1e3,
                color="tab:red",
                s=40,
                label="initial population",
            )
            ax.scatter(
                steps.voltage[-1] / 1e3,
                steps[name][-1] * 1e3,
                marker="*",
                s=220,
                color="tab:red",
                edgecolor="white",
                zorder=4,
                label="optimum",
            )
            ax.set_xlabel("middle voltage [kV]")
            ax.set_ylabel(label)
        fig[0].legend(fontsize=8, loc="upper left")
        mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis_r")
        plt.colorbar(mappable, ax=fig[1], label="log10 rms spot [µm]")

    # 4. space charge
    sc = result["space_charge"]
    with plasma_plots.figure(1, 1, figsize=(7, 4)) as fig:
        ax = fig[0]
        spot = micro(sc.spot).assign_coords(current=1e3 * sc.current)
        spot.current.attrs.update(units="mA", long_name="beam current")
        spot.plasma.plot.profiles(x="voltage", over="current", logy=True, ax=ax)
        for i, line in enumerate(ax.get_lines()[: sc.sizes["current"]]):
            line.set_label(f"{1e3 * float(sc.current[i]):.0f} mA")
            ax.scatter(
                float(sc.optimum_voltage[i]) / 1e3,
                float(sc.optimum_spot[i]) * 1e6,
                marker="*",
                s=200,
                color=line.get_color(),
                edgecolor="white",
                zorder=4,
            )
        ax.axvline(thin, color="gray", ls="--", label="thin lens, no space charge")
        ax.scatter([], [], marker="*", s=120, color="gray", label="optimiser's optimum")
        ax.set_title("space charge moves the optimum to higher voltages")
        ax.legend(fontsize=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    output = simulate(args.preset)
    print(table(output))
    plot(output)
    plt.show()
