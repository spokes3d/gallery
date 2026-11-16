"""Secondary electrons.

What becomes of the electrons an ion knocks out of the target? Each is emitted with a few eV,
accelerated to about 300 eV by the cathode sheath and turned back towards the target by the
magnetic field. Without a collision it returns to the target and is recaptured; only an
electron that collides on its way stays in the trap, where it ionizes the gas until it has
spent its energy. The study follows such electrons from their emission to their end.

The fields are those of the study *Magnetron electron orbits*: the balanced 2-inch magnetron,
a 300 V cathode sheath 1 mm thick and a presheath field of 2 kV/m. The electrons leave the
target around the racetrack radius with a cosine angular distribution and a mean energy of 4
eV. They are Boris-pushed (mpiparticles) and collide with argon at 0.3 to 5 Pa (plasmacoll's
Monte Carlo collisions: elastic, one excitation, ionization), and so do the electrons that the
ionizations create. An electron leaves the run when it hits the target, when it reaches 12 mm
above it or the edge of the domain, or when it has cooled below the ionization threshold above
the sheath.

Most electrons return within a few nanoseconds, after one or a few hops along the target. At
0.3 Pa 87 % are recaptured without a collision, at 5 Pa 37 %. An electron that stays makes about
20 ions, twice the sheath voltage over 30 eV per ion: a third of the ionizations happen inside
the sheath, and the electrons they free are accelerated by it too.

Compared: the fraction of electrons that collide before they first return to the target,
against the probability $1 - \\exp(-n\\int\\sigma v\\,dt)$ integrated along the collisionless
orbits (they agree within the statistical error of 150 electrons); and the ions per secondary
electron against the sheath voltage divided by an energy cost of 30 eV per ion, the ideal of
Thornton's model.

Reference: J. A. Thornton, J. Vac. Sci. Technol. 15 (1978) 171; Lieberman and Lichtenberg,
*Principles of Plasma Discharges and Materials Processing*, 2nd ed. (Wiley, 2005), chapter 14.
"""

import argparse
import os
import time

# a few hundred markers: OpenMP threads in the compiled push only cost time (set before the
# kernel library is loaded)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from plasmacoll.constants import BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk

PRESETS = {
    "smoke": {
        "electrons": 20,
        "pressures": [2.0],
        "duration": 20e-9,
        "free_duration": 20e-9,
        "orbits": 1,
    },
    "full": {
        "electrons": 150,
        "pressures": [0.3, 0.5, 1.0, 2.0, 5.0],
        "duration": 1.0e-6,
        "free_duration": 0.2e-6,
        "orbits": 3,
    },
}

MU0 = 4.0e-7 * np.pi
#: Remanence [T], faces [m], radii [m] of the balanced 2-inch magnetron (assumed).
REMANENCE = 1.2
MAGNET_TOP, MAGNET_HEIGHT = -8.0e-3, 8.0e-3
CENTRE_RADIUS, RING_INNER, RING_OUTER = 12.0e-3, 20.0e-3, 23.3e-3
#: Gauss-Legendre nodes per magnet face.
NODES = 8
#: Prescribed potential (assumed): sheath voltage [V] and width [m], presheath field [V/m].
SHEATH_VOLTAGE, SHEATH_WIDTH, PRESHEATH_FIELD = 300.0, 1.0e-3, 2.0e3
#: The domain: height of the top [m] and outer radius [m]; the field table spacing [m].
TOP, OUTER, SPACING = 12.0e-3, 36.0e-3, 5.0e-5
#: Emission (assumed): spread of the radius about the racetrack [m], and the temperature [eV]
#: of the energy distribution E exp(-E/T) (mean 2T).
EMISSION_SPREAD, EMISSION_TEMPERATURE = 2.0e-3, 2.0
#: Time step [s] (omega_ce dt < 0.1 at 0.137 T, the strongest field in the band) and the
#: steps between collisions.
DT, COLLIDE_EVERY = 4.0e-12, 5
#: Argon temperature [K]; ionization threshold [eV]; energy cost per ion of the ideal [eV].
GAS_TEMPERATURE, IONIZATION_ENERGY, ION_COST = 300.0, 15.76, 30.0
#: The orbits drawn: quantiles of the return time, and the longest time drawn [s].
ORBIT_QUANTILES, ORBIT_DURATION = (0.1, 0.5, 0.85), 40e-9
#: The fates of an electron, in the order of the figure.
FATES = ("recaptured", "returned", "cooled", "escaped", "trapped")
FATE_LABELS = {
    "recaptured": "recaptured, no collision",
    "returned": "back to the target after collisions",
    "cooled": "cooled below 15.8 eV",
    "escaped": "left the domain",
    "trapped": "still in the trap",
}


def loop_field(current, radius, height, r, z):
    """Field (B_r, B_z) [T] of circular current loops, exact (elliptic integrals)."""
    r = np.maximum(r, 1e-12)
    dz = z - height
    a2 = (radius + r) ** 2 + dz**2
    d2 = (radius - r) ** 2 + dz**2
    m = 4.0 * radius * r / a2
    k, e = ellipk(m), ellipe(m)
    c = MU0 * current / (2.0 * np.pi * np.sqrt(a2))
    b_r = c * dz / r * (-k + (radius**2 + r**2 + dz**2) / d2 * e)
    b_z = c * (k + (radius**2 - r**2 - dz**2) / d2 * e)
    return b_r, b_z


def magnet_field(r, z):
    """(B_r, B_z) [T] of the magnets at points (r, z) of any shape (loops for the magnets)."""
    x, w = np.polynomial.legendre.leggauss(NODES)
    heights = MAGNET_TOP - 0.5 * MAGNET_HEIGHT * (1.0 - x)
    weights = 0.5 * MAGNET_HEIGHT * w * REMANENCE / MU0
    sheets = ((CENTRE_RADIUS, 1.0), (RING_INNER, 1.0), (RING_OUTER, -1.0))
    current = np.concatenate([sign * weights for _, sign in sheets])[:, None]
    radius = np.repeat([s[0] for s in sheets], NODES)[:, None]
    height = np.tile(heights, len(sheets))[:, None]
    r, z = np.asarray(r, dtype=float), np.asarray(z, dtype=float)
    b_r, b_z = loop_field(current, radius, height, r.ravel()[None], z.ravel()[None])
    return b_r.sum(axis=0).reshape(r.shape), b_z.sum(axis=0).reshape(r.shape)


def potential(z):
    """Prescribed potential [V] against the height z [m] above the target."""
    sheath = np.where(z < SHEATH_WIDTH, (1.0 - z / SHEATH_WIDTH) ** 2, 0.0)
    return PRESHEATH_FIELD * z - SHEATH_VOLTAGE * sheath


def axial_field(z):
    """Prescribed E_z [V/m]: the uniform presheath field plus the linear sheath field."""
    sheath = np.where(z < SHEATH_WIDTH, 1.0 - z / SHEATH_WIDTH, 0.0)
    return -PRESHEATH_FIELD - 2.0 * SHEATH_VOLTAGE / SHEATH_WIDTH * sheath


class Fields:
    """The magnet field tabulated on an (r, z) grid, and E and B at Cartesian points."""

    def __init__(self):
        """Tabulate (B_r, B_z) on the domain."""
        r = np.arange(0.0, OUTER + 2 * SPACING, SPACING)
        z = np.arange(0.0, TOP + 2 * SPACING, SPACING)
        b_r, b_z = magnet_field(*np.meshgrid(r, z, indexing="ij"))
        self.table = np.stack([b_r, b_z])  # (2, nr, nz), uniform spacing from (0, 0)
        self.racetrack = brentq(
            lambda x: magnet_field(x, 0.0)[1], CENTRE_RADIUS, RING_INNER
        )

    def __call__(self, x):
        """E and B [V/m, T] at Cartesian points x (3, n) inside the domain."""
        r = np.hypot(x[0], x[1])
        # bilinear interpolation in the uniform table
        nr, nz = self.table.shape[1:]
        u = np.clip(r / SPACING, 0.0, nr - 1.000001)
        w = np.clip(x[2] / SPACING, 0.0, nz - 1.000001)
        i, j = u.astype(int), w.astype(int)
        u, w = u - i, w - j
        t = self.table
        b_r, b_z = (
            (1 - u) * (1 - w) * t[:, i, j]
            + u * (1 - w) * t[:, i + 1, j]
            + (1 - u) * w * t[:, i, j + 1]
            + u * w * t[:, i + 1, j + 1]
        )
        c, s = x[0] / np.maximum(r, 1e-12), x[1] / np.maximum(r, 1e-12)
        e = np.zeros_like(x)
        e[2] = axial_field(x[2])
        return e, np.array([b_r * c, b_r * s, b_z])


def emit(fields, count, rng):
    """Positions and velocities (3, count) of secondary electrons leaving the target."""
    radius = fields.racetrack + EMISSION_SPREAD * rng.standard_normal(count)
    angle = rng.uniform(0.0, 2 * np.pi, count)
    positions = np.array(
        [radius * np.cos(angle), radius * np.sin(angle), np.full(count, 1e-9)]
    )
    energy = rng.gamma(2.0, EMISSION_TEMPERATURE, count)
    speed = np.sqrt(2.0 * ELEMENTARY_CHARGE * energy / ELECTRON_MASS)
    cos_theta = np.sqrt(rng.random(count))  # a cosine (Lambert) distribution about +z
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = rng.uniform(0.0, 2 * np.pi, count)
    velocities = speed * np.array(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta]
    )
    return positions, velocities


def collisions(pressure, seed=1):
    """Electron-argon Monte Carlo collisions with stand-in cross sections."""
    from plasmacoll import (
        CollisionProcess,
        CrossSection,
        MonteCarloCollisions,
        NeutralBackground,
    )
    from plasmacoll.argon import ARGON_MASS, ARGON_OPAL_ENERGY
    from plasmacoll.constants import kelvin_to_ev
    from plasmacoll.metals import LotzSubshell, lotz_table

    grid = np.geomspace(1e-3, 1e4, 400)

    def elastic(energy):
        return 1.0e-19 * np.minimum(1.0, (20.0 / np.maximum(energy, 1e-30)) ** 0.8)

    def excitation(energy):
        return 1.5e-21 * np.clip(energy - 11.5, 0, None) ** 0.5 * np.exp(-energy / 200)

    processes = [
        CollisionProcess(
            "elastic",
            "Ar",
            CrossSection.from_function(elastic, grid),
            energy_frame="lab",
        ),
        CollisionProcess(
            "excitation",
            "Ar",
            CrossSection.from_function(excitation, grid, threshold=11.5),
            energy_frame="lab",
        ),
        CollisionProcess(
            "ionization",
            "Ar",
            lotz_table([LotzSubshell(IONIZATION_ENERGY, 6)], label="Ar Lotz 3p"),
            energy_frame="lab",
            products={"electron": "e", "ion": "Ar+"},
            energy_sharing="opal",
            sharing_energy=ARGON_OPAL_ENERGY,
        ),
    ]
    argon = NeutralBackground(
        "Ar",
        pressure / (BOLTZMANN * GAS_TEMPERATURE),
        kelvin_to_ev(GAS_TEMPERATURE),
        ARGON_MASS,
    )
    return MonteCarloCollisions(
        species_masses={"e": ELECTRON_MASS, "Ar+": ARGON_MASS},
        backgrounds=[argon],
        processes={"e": processes},
        seed=seed,
    )


def kinetic_energy(v):
    """Kinetic energy [eV] of velocities (3, n)."""
    return 0.5 * ELECTRON_MASS * (v * v).sum(axis=0) / ELEMENTARY_CHARGE


def free_flight(fields, positions, velocities, duration, keep):
    """Collisionless orbits until the electrons first return to the target.

    Returns the return times (nan if not back within ``duration``), the integral of
    sigma_total v dt per unit density at 1 Pa scaled out (the collision frequency over the
    density, integrated), the largest change of the total energy K - e phi [eV], and the
    first ``keep`` orbits.
    """
    from mpiparticles import ParticleStore, boris_push

    mcc = collisions(1.0)
    density = 1.0 / (BOLTZMANN * GAS_TEMPERATURE)
    store = ParticleStore(ndim=3, nvel=3)
    store.add(positions, velocities)
    n = store.length
    q = -ELEMENTARY_CHARGE
    back = np.full(n, np.nan)
    exposure = np.zeros(n)
    flying = np.ones(n, dtype=bool)
    p, v = store.positions[:, :n], store.velocities[:, :n]
    energy0 = kinetic_energy(v) - potential(p[2] - 0.5 * DT * v[2])
    drift = np.zeros(n)
    orbit = [p[:, :keep].copy()]
    steps = int(round(duration / DT))
    for step in range(steps):
        e, b = fields(np.where(flying, p, positions[:, :1]))
        boris_push(store, e, b, DT, q / ELECTRON_MASS)
        speed = np.sqrt((v * v).sum(axis=0))
        exposure[flying] += DT * mcc.total_frequency("e", speed[flying]) / density
        hit = flying & (p[2] < 0.0)
        out = flying & ((p[2] > TOP) | (np.hypot(p[0], p[1]) > OUTER))
        back[hit] = (step + 1) * DT
        live = flying & ~hit & ~out
        # v is at the half step: pair it with the potential half way along the step
        total = kinetic_energy(v) - potential(p[2] - 0.5 * DT * v[2])
        drift[live] = np.maximum(drift[live], np.abs(total[live] - energy0[live]))
        flying &= ~(hit | out)
        if keep:  # the orbit up to and including the step that leaves
            orbit.append(np.where(flying | hit | out, p, np.nan)[:, :keep].copy())
        if not flying.any():
            break
    return {
        "return_time": back,
        "exposure": exposure,
        "energy_drift": float(drift.max()),
        "orbit": np.array(orbit).transpose(2, 1, 0) if keep else None,
    }


def follow(fields, positions, velocities, pressure, duration, seed):
    """Push and collide the electrons and their offspring at one pressure; count the fates."""
    from mpiparticles import ParticleStore, boris_push

    mcc = collisions(pressure, seed)
    store = ParticleStore(ndim=3, nvel=3)
    store.add(positions, velocities)
    primaries = positions.shape[1]
    q = -ELEMENTARY_CHARGE
    # per stored column: a primary or not, collided yet, and the fate (index into FATES)
    primary = np.ones(primaries, dtype=bool)
    collided = np.zeros(primaries, dtype=bool)
    fate = np.full(primaries, -1)
    counts = dict.fromkeys(FATES, 0)
    escaped_energy = []
    ionizations = excitations = in_sheath = 0
    steps = int(round(duration / DT))
    t = 0.0
    for step in range(steps):
        n = store.length
        p, v, alive = store.positions[:, :n], store.velocities[:, :n], store.alive[:n]
        if not alive.any():
            break
        e, b = fields(np.where(alive, p, positions[:, :1]))
        boris_push(store, e, b, DT, q / ELECTRON_MASS)
        t += DT
        p, v = store.positions[:, :n], store.velocities[:, :n]

        hit = alive & (p[2] < 0.0)
        out = alive & ((p[2] > TOP) | (np.hypot(p[0], p[1]) > OUTER))
        cool = (
            alive
            & ~hit
            & ~out
            & (p[2] > SHEATH_WIDTH)
            & (kinetic_energy(v) < IONIZATION_ENERGY)
        )
        for name, mask in (
            ("recaptured", hit & ~collided[:n]),
            ("returned", hit & collided[:n]),
            ("escaped", out),
            ("cooled", cool),
        ):
            fate[mask & primary[:n]] = FATES.index(name)
            counts[name] += int(mask.sum())
        if out.any():
            escaped_energy.extend(kinetic_energy(v[:, out]).tolist())
        gone = hit | out | cool
        if gone.any():
            store.deactivate(gone)
        alive = store.alive[:n]

        if step % COLLIDE_EVERY == 0 and alive.any():
            index = np.flatnonzero(alive)
            result = mcc.collide(
                "e",
                np.ascontiguousarray(p[:, index].T),
                np.ascontiguousarray(v[:, index].T),
                np.ones(index.size),
                COLLIDE_EVERY * DT,
            )
            changed = np.any(result.velocities != v[:, index].T, axis=1)
            collided[index[changed]] = True
            v[:, index] = result.velocities.T
            ionizations += result.diagnostics.counts.get("ionization:Ar", 0)
            excitations += result.diagnostics.counts.get("excitation:Ar", 0)
            for batch in result.created.get("Ar+", []):
                in_sheath += int((batch.positions[:, 2] < SHEATH_WIDTH).sum())
            for batch in result.created.get("e", []):
                if len(batch.weights):
                    store.add(batch.positions.T, batch.velocities.T)
                    k = len(batch.weights)
                    primary = np.concatenate([primary, np.zeros(k, dtype=bool)])
                    collided = np.concatenate([collided, np.ones(k, dtype=bool)])
                    fate = np.concatenate([fate, np.full(k, -1)])
    n = store.length
    trapped = store.alive[:n]
    counts["trapped"] += int(trapped.sum())
    fate[trapped & primary] = FATES.index("trapped")
    return {
        "pressure": pressure,
        "fates": np.array([np.mean(fate[primary] == i) for i in range(len(FATES))]),
        # a primary leaves at its first return, so this is: collided before it
        "collided_first": float(collided[:primaries].mean()),
        "ionizations": ionizations / primaries,
        "excitations": excitations / primaries,
        "in_sheath": in_sheath / max(ionizations, 1),
        "electrons": n,
        "offspring_fates": counts,
        "escaped_energy": float(np.mean(escaped_energy)) if escaped_energy else 0.0,
        "time": t,
    }


def simulate(preset: str = "full") -> dict:
    """Emit the electrons; follow them without collisions, then at every pressure."""
    size = PRESETS[preset]
    rng = np.random.default_rng(23)
    fields = Fields()
    positions, velocities = emit(fields, size["electrons"], rng)
    start = time.time()
    free = free_flight(fields, positions, velocities, size["free_duration"], 0)
    # the orbits drawn: electrons that return early, at the median time and late
    back = np.where(np.isfinite(free["return_time"]), free["return_time"], np.inf)
    order = np.argsort(back)
    chosen = order[
        (np.array(ORBIT_QUANTILES[: size["orbits"]]) * (back.size - 1)).astype(int)
    ]
    shown = free_flight(
        fields,
        positions[:, chosen],
        velocities[:, chosen],
        min(ORBIT_DURATION, size["free_duration"]),
        len(chosen),
    )
    runs = [
        follow(fields, positions, velocities, pressure, size["duration"], seed=i + 1)
        for i, pressure in enumerate(size["pressures"])
    ]
    density = np.array(size["pressures"]) / (BOLTZMANN * GAS_TEMPERATURE)
    predicted = [float(np.mean(1.0 - np.exp(-n * free["exposure"]))) for n in density]
    r_line = np.linspace(4e-3, 30e-3, 300)
    return {
        "preset": preset,
        "seconds": time.time() - start,
        "racetrack": fields.racetrack,
        "pressure": np.array(size["pressures"]),
        "fates": np.array([run["fates"] for run in runs]),
        "collided_first": np.array([run["collided_first"] for run in runs]),
        "predicted_first": np.array(predicted),
        "ionizations": np.array([run["ionizations"] for run in runs]),
        "excitations": np.array([run["excitations"] for run in runs]),
        "in_sheath": np.array([run["in_sheath"] for run in runs]),
        "electrons": np.array([run["electrons"] for run in runs]),
        "escaped_energy": np.array([run["escaped_energy"] for run in runs]),
        "return_time": free["return_time"],
        "energy_drift": free["energy_drift"],
        "orbit": shown["orbit"][:, :, ::2].astype(np.float32),
        "orbit_return": shown["return_time"],
        "duration": size["duration"],
        "field_top": magnet_field(r_line, np.zeros_like(r_line)),
        "r_line": r_line,
    }


def plot(result: dict) -> None:
    """Draw the figures and print the agreement."""
    mm, ns = 1e3, 1e9
    pressure = result["pressure"]

    # 1. collisionless orbits in (r, z)
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    ax.axhspan(0.0, mm * SHEATH_WIDTH, color="0.85", label="cathode sheath")
    ax.axvline(mm * result["racetrack"], color="0.4", ls=":", label="racetrack radius")
    for path, back in zip(result["orbit"], result["orbit_return"], strict=True):
        x, y, z = path
        label = f"back after {ns * back:.1f} ns" if np.isfinite(back) else "not back"
        ax.plot(mm * np.hypot(x, y), mm * np.maximum(z, 0.0), lw=0.6, label=label)
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("height above the target z [mm]")
    ax.set_ylim(0.0, None)
    ax.legend(fontsize=8)

    # 2. return times without collisions
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    back = result["return_time"]
    finite = np.sort(ns * back[np.isfinite(back)])
    ax.step(finite, np.arange(1, finite.size + 1) / back.size, where="post")
    ax.set_xscale("log")
    ax.set_xlabel("time to the first return to the target [ns]")
    ax.set_ylabel("fraction of the electrons back")
    ax.grid(True, alpha=0.3, which="both")

    # 3. collided before the first return: Monte Carlo against the path integral
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    ax.plot(
        pressure,
        result["predicted_first"],
        "-",
        label="1 − ⟨exp(−n∫σv dt)⟩, free orbits",
    )
    count = result["return_time"].size
    spread = np.sqrt(result["collided_first"] * (1 - result["collided_first"]) / count)
    ax.errorbar(
        pressure,
        result["collided_first"],
        yerr=spread,
        fmt="o",
        capsize=3,
        label=f"Monte Carlo collisions ({count} electrons, ±1σ)",
    )
    ax.set_xscale("log")
    ax.set_xlabel("argon pressure [Pa]")
    ax.set_ylabel("collided before the first return")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # 4. fates against the pressure
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for i, name in enumerate(FATES):
        if result["fates"][:, i].any():
            ax.plot(pressure, result["fates"][:, i], "o-", label=FATE_LABELS[name])
    ax.set_xscale("log")
    ax.set_xlabel("argon pressure [Pa]")
    ax.set_ylabel("fraction of the secondary electrons")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # 5. ions per secondary electron
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    ax.plot(
        pressure,
        result["ionizations"],
        "o-",
        label="ionizations per secondary electron",
    )
    ax.plot(
        pressure,
        result["excitations"],
        "s--",
        label="excitations per secondary electron",
    )
    ideal = SHEATH_VOLTAGE / ION_COST
    ax.axhline(
        ideal, color="0.4", ls=":", label=f"{SHEATH_VOLTAGE:g} V / {ION_COST:g} eV"
    )
    ax.set_xscale("log")
    ax.set_xlabel("argon pressure [Pa]")
    ax.set_ylabel("events per secondary electron")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    error = np.abs(result["collided_first"] - result["predicted_first"]).max()
    print(
        f"collided before the first return: Monte Carlo within {error:.3f} of the path "
        f"integral; total energy drift of the free orbits {result['energy_drift']:.2f} eV"
    )


def table(result: dict) -> str:
    """The fates, ionizations and the effective yield factor per pressure."""
    heads = " | ".join(FATE_LABELS[f] for f in FATES)
    lines = [
        f"| p [Pa] | collided before return (MC / theory) | {heads} "
        "| ions per electron |",
        "|---" * (len(FATES) + 3) + "|",
    ]
    for i, p in enumerate(result["pressure"]):
        fates = " | ".join(f"{x:.2f}" for x in result["fates"][i])
        lines.append(
            f"| {p:g} | {result['collided_first'][i]:.2f} / {result['predicted_first'][i]:.2f} "
            f"| {fates} | {result['ionizations'][i]:.2f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    result = simulate(args.preset)
    print(table(result))
    plot(result)
    plt.show()
