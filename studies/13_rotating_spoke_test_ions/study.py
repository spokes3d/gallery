"""Rotating-spoke test ions.

Where do the argon ions made in a spoke go, and with what energy do they reach the target? The
background is that of the rotating-spoke test electrons, unchanged: a 2D slice above the
racetrack ($R_0 = 3$ cm) with a 25 V radial well and three 12 V potential humps rotating at
10 km/s, a radial magnetic field of 30 mT and a uniform axial presheath field
$E_z = -3$ kV/m towards the target.

Ar$^+$ test ions are born with the background's ionization rate $n_e\\,k_{iz}(T_e)$, which peaks
just behind the leading edge of each spoke, at a random height between the cathode sheath edge
and the anode plane. They are Boris-pushed (mpiparticles) through the rotating field gathered by
byob and collide with argon at 0.5 Pa (plasmacoll's Monte Carlo
collisions with Phelps' cross sections, including resonant charge exchange). Each ion is
followed until it reaches the cathode sheath edge (the target), the anode plane or the outer
edge.

The energies are those at the sheath edge; the sheath voltage adds to all of them. The
presheath alone gives at most 30 eV, so an ion above 30 eV has gained energy from the in-plane
field of the well and the spokes.

Compared: nothing; this is a demonstration of the codes on a made-up background.

Reference: the picture of spokes as rotating ionization zones with a potential hump, e.g.
Hecimovic and von Keudell, J. Phys. D 51 (2018) 453001; the Ar$^+$ + Ar cross sections of
Phelps, J. Appl. Phys. 76 (1994) 747.
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

#: Ions born over ``born`` periods of the pattern; the animation records the last period.
PRESETS = {
    "smoke": {"ions": 300, "born": 0.02, "frames": 3},
    "quick": {"ions": 200000, "born": 1.5, "frames": 30},
}

QE = 1.602176634e-19
L = 0.05  # half width of the domain [m]
NG = 201  # grid nodes per axis of the background
NMAP = 101  # nodes per axis of the maps
R0 = 0.03  # racetrack radius [m]
WR = 0.006  # racetrack half width [m]
B0 = 0.03  # radial field at the racetrack [T]
EZ = -3.0e3  # axial (presheath) field [V/m]
LZ = 0.005  # cathode sheath edge at -LZ, anode plane at +LZ [m]
R_EDGE = 0.95 * L  # the outer edge of the slice [m]
V_RING = 25.0  # depth of the radial well [V]
V_SPOKE = 12.0  # potential hump of a spoke [V]
TE0, DTE = 3.0, 4.0  # electron temperature outside, and extra inside a spoke [eV]
M_SPOKES = 3
SPOKE_SPEED = 1.0e4  # at R0, in the direction of the Hall drift [m/s]
OMEGA = -SPOKE_SPEED / R0  # [rad/s]; negative: the sense of the Hall drift
PERIOD = 2 * np.pi / (M_SPOKES * abs(OMEGA))  # the pattern repeats after this [s]
P_AR, T_GAS = 0.5, 300.0  # [Pa], [K]
AR_MASS = 39.948 * 1.66053906660e-27  # [kg]
DT = 1.0e-8  # [s]
MAX_STEPS = 4000  # an ion still inside after 40 µs is dropped (none are)
N_BINS = 60  # azimuthal bins
E_BINS = np.linspace(0.0, 60.0, 61)  # energy bins [eV]
E_HOT = 30.0  # the most the axial presheath gives [eV]
N_SHARE = 51  # cells per axis of the map of the energetic share
Z_BINS = 10  # bins of the birth height
MIN_BORN = 30  # births a cell needs to show its share
MOVIE_MARKERS = 500  # ions per frame of the animation
FATES = ("target", "substrate", "edge")


def spoke_shape(theta):
    """The sum of the asymmetric spoke profiles in the spoke frame, peak 1."""
    shape = np.zeros_like(theta)
    for k in range(M_SPOKES):
        psi = np.angle(np.exp(1j * (theta - 2 * np.pi * k / M_SPOKES)))
        # the spoke moves towards -theta: a sharp edge at psi = -0.15, a long tail behind
        rise = 0.5 * (1 + np.tanh((psi + 0.15) / 0.04))
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


def ionization_rate(background):
    """The ionization rate n_e k_iz(T_e) on the background's grid [m^-3 s^-1]."""
    from plasmacoll import rate_coefficient
    from plasmacoll.constants import ELECTRON_MASS
    from plasmacoll.metals import LotzSubshell, lotz_table

    lotz = lotz_table([LotzSubshell(15.76, 6)], label="Ar Lotz 3p")
    temperatures = np.linspace(TE0 - 0.5, TE0 + DTE + 0.5, 41)
    rates = rate_coefficient(
        lotz, temperatures, ELECTRON_MASS, incident_mass=ELECTRON_MASS
    )
    te = background.bulk["electron_temperature"]
    return background.bulk["density"] * np.interp(te, temperatures, rates)


def b_field(x, y):
    """The radial magnetic field (3, N) [T]."""
    r = np.hypot(x, y) + 1e-12
    br = B0 * (r / R0) * np.exp(-(((r - R0) / 0.015) ** 2))
    return np.stack([br * x / r, br * y / r, np.zeros_like(r)])


def rotate(x, y, angle):
    """Rotate the vectors (x, y) by ``angle``."""
    c, s = np.cos(angle), np.sin(angle)
    return c * x - s * y, s * x + c * y


def e_field(background, positions, t):
    """The lab-frame field (3, N) [V/m]: the spoke-frame field rotated by OMEGA t."""
    xs, ys = rotate(positions[0], positions[1], -OMEGA * t)
    field = background.electric_field(np.stack([xs, ys], axis=1))
    ex, ey = rotate(field[:, 0], field[:, 1], OMEGA * t)
    return np.stack([ex, ey, np.full_like(ex, EZ)])


def spoke_angle(x, y, t):
    """The angle in the spoke frame, in [-pi, pi)."""
    return np.angle(np.exp(1j * (np.arctan2(y, x) - OMEGA * t)))


def collisions():
    """Ar+ + Ar Monte Carlo collisions with Phelps' cross sections."""
    from plasmacoll import MonteCarloCollisions, NeutralBackground
    from plasmacoll.argon import argon_ion_processes
    from plasmacoll.constants import kelvin_to_ev

    argon = NeutralBackground(
        "Ar", P_AR / (1.380649e-23 * T_GAS), kelvin_to_ev(T_GAS), AR_MASS
    )
    return MonteCarloCollisions(
        species_masses={"Ar+": AR_MASS},
        backgrounds=[argon],
        processes={"Ar+": argon_ion_processes()},
        seed=1,
    )


def sample_births(background, weights, rng, count, t):
    """Spoke-frame birth points from ``weights``, rotated into the lab frame at time ``t``."""
    nodes = rng.choice(weights.size, size=count, p=weights.ravel() / weights.sum())
    points = background.grid.points().T[nodes]
    points = points + (rng.random(points.shape) - 0.5) * background.grid.spacing
    x, y = rotate(points[:, 0], points[:, 1], OMEGA * t)
    positions = np.stack([x, y, rng.uniform(-LZ, LZ, count)])
    velocities = rng.normal(size=(3, count)) * np.sqrt(1.380649e-23 * T_GAS / AR_MASS)
    return points, positions, velocities


def run(preset: str) -> dict:
    """Inject, push and collide the ions until all have left; return their fates."""
    from mpiparticles import ParticleStore, boris_push

    size = PRESETS[preset]
    rng = np.random.default_rng(0)
    background = build_background()
    birth_weights = ionization_rate(background)
    mcc = collisions()
    store = ParticleStore(ndim=3, nvel=3)

    total = size["ions"]
    n_inject = int(round(size["born"] * PERIOD / DT))
    per_step = np.diff(np.round(np.linspace(0, total, n_inject + 1)).astype(int))
    # per ion: birth (spoke frame), its fate and what it carries when it leaves
    birth = np.zeros((total, 3))
    fate = np.full(total, -1, dtype=np.int8)
    energy = np.full(total, np.nan)
    incidence = np.full(total, np.nan)
    arrival = np.full(total, np.nan)
    lifetime = np.full(total, np.nan)
    shift = np.full(total, np.nan)  # azimuthal displacement in the lab frame [rad]
    radial = np.full(total, np.nan)  # radial displacement [m]
    born_at = np.zeros(total)
    born_lab = np.zeros(total)  # the lab-frame angle of birth
    counts = {"ar_ion_isotropic": 0, "ar_ion_backscatter": 0}
    frame_every = max(1, int(round(PERIOD / DT / size["frames"])))
    first_frame = max(0, n_inject - size["frames"] * frame_every)
    movie = []
    born = 0
    t = 0.0
    start = time.time()
    for step in range(n_inject + MAX_STEPS):
        if step < n_inject and per_step[step]:
            k = per_step[step]
            points, positions, velocities = sample_births(
                background, birth_weights, rng, k, t
            )
            ids = store.add(positions, velocities)
            birth[ids, :2] = points
            birth[ids, 2] = positions[2]
            born_at[ids] = t
            born_lab[ids] = np.arctan2(positions[1], positions[0])
            born += k
        n = store.length
        if store.num_alive == 0:
            if step >= n_inject:
                break
            t += DT
            continue
        p, v, alive = store.positions[:, :n], store.velocities[:, :n], store.alive[:n]
        e = e_field(background, p, t + 0.5 * DT)
        boris_push(store, e, b_field(p[0], p[1]), DT, QE / AR_MASS)
        t += DT

        index = np.flatnonzero(alive)
        velocities = np.ascontiguousarray(v[:, index].T)
        result = mcc.collide(
            "Ar+",
            np.ascontiguousarray(p[:2, index].T),
            velocities,
            np.ones(index.size),
            DT,
            in_place=True,
        )
        v[:, index] = velocities.T
        for name, value in result.diagnostics.counts.items():
            counts[name] = counts.get(name, 0) + value

        ids = store.ids[:n]
        leaving = {
            0: (p[2] < -LZ) & alive,
            1: (p[2] > LZ) & alive,
            2: (np.hypot(p[0], p[1]) > R_EDGE) & alive,
        }
        dead = np.zeros(n, bool)
        for code, mask in leaving.items():
            if not mask.any():
                continue
            mask &= ~dead
            which = ids[mask]
            speed2 = np.sum(v[:, mask] ** 2, axis=0)
            fate[which] = code
            energy[which] = 0.5 * AR_MASS * speed2 / QE
            incidence[which] = np.degrees(
                np.arccos(np.clip(np.abs(v[2, mask]) / np.sqrt(speed2), 0, 1))
            )
            arrival[which] = spoke_angle(p[0, mask], p[1, mask], t)
            lifetime[which] = t - born_at[which]
            radial[which] = np.hypot(p[0, mask], p[1, mask]) - np.hypot(
                *birth[which, :2].T
            )
            shift[which] = np.angle(
                np.exp(1j * (np.arctan2(p[1, mask], p[0, mask]) - born_lab[which]))
            )
            dead |= mask
        if dead.any():
            store.deactivate(dead)
        store.compact_if_needed()

        if step >= first_frame and (step - first_frame) % frame_every == 0:
            if len(movie) < size["frames"]:
                n = store.length
                live = np.flatnonzero(store.alive[:n])
                show = rng.choice(
                    live, size=min(MOVIE_MARKERS, live.size), replace=False
                )
                vs = store.velocities[:, show]
                movie.append(
                    {
                        "t": t,
                        "x": store.positions[0, show].astype(np.float16),
                        "y": store.positions[1, show].astype(np.float16),
                        "energy": (0.5 * AR_MASS * np.sum(vs * vs, axis=0) / QE).astype(
                            np.float16
                        ),
                        "alive": int(live.size),
                    }
                )
        if step % 200 == 0:
            print(
                f"step {step}  t = {t * 1e6:.2f} us  born {born}  inside {store.num_alive}  "
                f"{time.time() - start:.0f} s"
            )

    sample = background.sample(resolution=NMAP)
    # the birth distribution on the map grid, relative to its maximum
    births, _, _ = np.histogram2d(
        birth[:, 0], birth[:, 1], bins=NMAP, range=[[-L, L], [-L, L]]
    )
    return {
        "preset": preset,
        "born": born,
        "fate": fate,
        "energy": energy,
        "incidence": incidence,
        "arrival": arrival,
        "lifetime": lifetime,
        "shift": shift,
        "radial": radial,
        "birth": birth,
        "births_map": births,
        "counts": counts,
        "movie": movie,
        "axis": sample["x"].values,
        "potential": sample["potential"].values.astype(np.float32),
        "wall_time": time.time() - start,
    }


def compact(full: dict) -> dict:
    """The result for the cache: histograms, profiles and means, and the energetic births."""
    fate, energy = full["fate"], full["energy"]
    born = full["born"]
    target = fate == 0
    edges = np.linspace(-np.pi, np.pi, N_BINS + 1)
    angle, e_target = full["arrival"][target], energy[target]
    births = full["births_map"]
    z_birth = full["birth"][target, 2]
    keep = {
        "preset": full["preset"],
        "born": born,
        # per fate: the share of the ions and their energy distribution (the last bin: above)
        "share": np.array([np.mean(fate == code) for code in range(len(FATES))]),
        "energy_hist": np.array(
            [
                np.histogram(
                    np.clip(energy[fate == code], 0, E_BINS[-1] - 1e-6), E_BINS
                )[0]
                for code in range(len(FATES))
            ]
        ),
        # at the target against the spoke-frame angle: count, energy and the energetic ones
        "profile_count": np.histogram(angle, edges)[0],
        "profile_energy": np.histogram(angle, edges, weights=e_target)[0],
        "profile_hot": np.histogram(angle[e_target > E_HOT], edges)[0],
        "mean_energy": float(e_target.mean()),
        "median_energy": float(np.median(e_target)),
        "presheath_gain": float(np.mean(np.abs(EZ) * (z_birth + LZ))),
        "hot_count": int(np.sum(e_target > E_HOT)),
        "incidence": float(np.mean(full["incidence"][target])),
        "lifetime": float(np.mean(full["lifetime"][target])),
        # the azimuthal displacement from birth to the target, lab frame, along R0 [m]
        "shift_mean": float(np.mean(full["shift"][target]) * R0),
        "shift_rms": float(np.sqrt(np.mean(full["shift"][target] ** 2)) * R0),
        "radial_mean": float(np.mean(full["radial"][target])),
        "radial_rms": float(np.sqrt(np.mean(full["radial"][target] ** 2))),
        "counts": full["counts"],
        "movie": full["movie"],
        "axis": full["axis"],
        "potential": full["potential"].astype(np.float16),
        "births_map": (births / max(births.max(), 1e-30)).astype(np.float16),
        "wall_time": full["wall_time"],
    }
    # where the energetic ions at the target were born: the share above E_HOT of the ions born
    # in a cell of the spoke frame, and against the height of birth
    where = full["birth"][target, :2]
    above = e_target > E_HOT
    box = [[-L, L], [-L, L]]
    count = np.histogram2d(*where.T, bins=N_SHARE, range=box)[0]
    hot_map = np.histogram2d(*where[above].T, bins=N_SHARE, range=box)[0]
    keep["hot_share_map"] = np.where(
        count >= MIN_BORN, hot_map / np.maximum(count, 1), np.nan
    ).astype(np.float16)
    z_edges = np.linspace(-LZ, LZ, Z_BINS + 1)
    keep["hot_share_z"] = np.histogram(z_birth[above], z_edges)[0] / np.maximum(
        np.histogram(z_birth, z_edges)[0], 1
    )
    return keep


def simulate(preset: str = "quick") -> dict:
    """Run the preset; return the fates of the ions, the birth maps and the animation."""
    return compact(run(preset))


def potential_lab(result: dict, t: float):
    """The lab-frame potential on the map grid at time ``t`` (the rotated spoke frame)."""
    from scipy.interpolate import RegularGridInterpolator

    axis = result["axis"]
    interp = RegularGridInterpolator(
        (axis, axis),
        result["potential"].astype(float),
        bounds_error=False,
        fill_value=None,
    )
    x, y = np.meshgrid(axis, axis, indexing="ij")
    xs, ys = rotate(x, y, -OMEGA * t)
    return interp(np.stack([xs, ys], axis=-1))


def potential_ring(result: dict, angles):
    """The potential at R0 against the spoke-frame angle [V]."""
    from scipy.interpolate import RegularGridInterpolator

    axis = result["axis"]
    return RegularGridInterpolator((axis, axis), result["potential"].astype(float))(
        np.stack([R0 * np.cos(angles), R0 * np.sin(angles)], axis=-1)
    )


def steps(edges, values):
    """The x and y of a histogram drawn as a line of steps."""
    return np.repeat(edges, 2)[1:-1], np.repeat(values, 2)


def azimuthal(result: dict) -> dict:
    """Target flux, mean energy and the fraction above E_HOT against the spoke-frame angle."""
    edges = np.linspace(-np.pi, np.pi, N_BINS + 1)
    count = result["profile_count"].astype(float)
    return {
        "edges": edges,
        "angles": 0.5 * (edges[1:] + edges[:-1]),
        "flux": count / max(count.mean(), 1e-30),
        "mean_energy": result["profile_energy"] / np.maximum(count, 1),
        "hot_fraction": result["profile_hot"] / np.maximum(count, 1),
    }


def animate(result: dict):
    """The rotating spokes over one period: the lab-frame potential and the ions."""
    from matplotlib.animation import FuncAnimation

    movie = result["movie"]
    t0 = movie[0]["t"]
    extent = (-L * 100, L * 100, -L * 100, L * 100)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    image = ax.imshow(
        potential_lab(result, t0).T,
        origin="lower",
        extent=extent,
        cmap="cividis",
        vmin=-25,
        vmax=12,
    )
    dots = ax.scatter(
        movie[0]["x"].astype(float) * 100,
        movie[0]["y"].astype(float) * 100,
        c=movie[0]["energy"].astype(float),
        s=6,
        cmap="plasma",
        vmin=0,
        vmax=40,
        lw=0,
    )
    fig.colorbar(image, ax=ax, label="potential [V]", shrink=0.85)
    fig.colorbar(dots, ax=ax, label="ion energy [eV]", shrink=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-L * 100, L * 100)
    ax.set_ylim(-L * 100, L * 100)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    def update(i):
        frame = movie[i]
        image.set_data(potential_lab(result, frame["t"]).T)
        dots.set_offsets(
            np.c_[frame["x"].astype(float), frame["y"].astype(float)] * 100
        )
        dots.set_array(frame["energy"].astype(float))
        return image, dots

    animation = FuncAnimation(fig, update, frames=len(movie), interval=150)
    # the slider of the web figure shows the time
    animation._plasma_sweep = ("t [µs]", [(f["t"] - t0) * 1e6 for f in movie])
    return animation


def plot(result: dict):
    """Draw the energy distributions, the azimuthal profiles, the birth map and the animation."""

    born = result["born"]

    # 1. the energy distributions at the sheath edge and of the ions leaving otherwise
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    width = np.diff(E_BINS)
    for code, label in (
        (0, "target (cathode sheath edge)"),
        (2, "outer edge"),
        (1, "towards the substrate"),
    ):
        if result["share"][code] == 0:
            continue
        xs, ys = steps(E_BINS, result["energy_hist"][code] / born / width)
        ax.plot(xs, ys * 100, label=f"{label}: {result['share'][code]:.2%} of the ions")
    ax.axvline(
        E_HOT, color="k", ls=":", lw=0.9, label="most the presheath gives (30 eV)"
    )
    ax.set_xlim(0, E_BINS[-1])
    ax.set_xlabel("ion energy [eV] (last bin: everything above)")
    ax.set_ylabel("ions per eV [% of the ions born]")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. flux to the target and mean energy against the spoke-frame angle
    profile = azimuthal(result)
    degrees = np.degrees(profile["angles"])
    fine = np.linspace(-np.pi, np.pi, 361)
    phi = potential_ring(result, fine)
    births = result["births_map"].astype(float)
    axis = result["axis"]
    x, y = np.meshgrid(axis, axis, indexing="ij")
    ring = np.abs(np.hypot(x, y) - R0) < WR
    birth_profile = np.histogram(
        np.arctan2(y[ring], x[ring]), profile["edges"], weights=births[ring]
    )[0]
    birth_profile = birth_profile / max(birth_profile.mean(), 1e-30)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7, 5.6), sharex=True, constrained_layout=True
    )
    top.plot(
        *steps(np.degrees(profile["edges"]), profile["flux"]),
        label="ion flux to the target",
    )
    top.plot(
        *steps(np.degrees(profile["edges"]), birth_profile),
        label="ion birth at the racetrack",
    )
    top.plot(
        np.degrees(fine),
        (phi - phi.min()) / max(np.ptp(phi), 1e-30) * 2,
        "k--",
        lw=0.9,
        label="potential at $R_0$ (0 to 12 V)",
    )
    top.set_ylabel("relative to the mean")
    top.legend(fontsize=8, loc="upper right")
    top.grid(True, alpha=0.3)
    bottom.plot(
        degrees, profile["mean_energy"], "o-", ms=3, label="mean energy at the target"
    )
    bottom.plot(
        degrees,
        profile["hot_fraction"] * 100,
        "s-",
        ms=3,
        label=f"share above {E_HOT:.0f} eV [%]",
    )
    bottom.set_xlim(-180, 180)
    bottom.set_xticks(np.arange(-180, 181, 60))
    bottom.set_xlabel(
        "angle in the spoke frame at arrival [deg] (spokes move towards negative angles)"
    )
    bottom.set_ylabel("eV, or %")
    bottom.legend(fontsize=8, loc="upper right")
    bottom.grid(True, alpha=0.3)

    # 3. where the energetic ions were born (spoke frame): the share of the ions born in a cell
    # that reach the target above E_HOT, over the potential, and against the height of birth
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(6.4, 8.4),
        height_ratios=[2.4, 1],
        constrained_layout=True,
    )
    cell = np.linspace(-L, L, N_SHARE + 1) * 100
    image = top.imshow(
        result["hot_share_map"].astype(float).T * 100,
        origin="lower",
        extent=(cell[0], cell[-1], cell[0], cell[-1]),
        cmap="magma",
        vmin=0,
        vmax=100,
    )
    contours = top.contour(
        axis * 100,
        axis * 100,
        result["potential"].astype(float).T,
        levels=[-10, 0, 6],
        colors="#66ccff",
        linewidths=0.7,
    )
    top.clabel(contours, fmt="%g V", fontsize=7)
    fig.colorbar(image, ax=top, label=f"share above {E_HOT:.0f} eV [%]", shrink=0.85)
    top.set_aspect("equal")
    top.set_xlim(-4.5, 4.5)
    top.set_ylim(-4.5, 4.5)
    top.set_xlabel("x of birth [cm] (spoke frame)")
    top.set_ylabel("y of birth [cm]")
    z_edges = np.linspace(-LZ, LZ, Z_BINS + 1) * 1e3
    bottom.plot(*steps(z_edges, result["hot_share_z"] * 100))
    bottom.set_xlim(z_edges[0], z_edges[-1])
    bottom.set_ylim(0, 100)
    bottom.set_xlabel("height of birth z [mm] (target sheath edge at -5 mm)")
    bottom.set_ylabel(f"share above {E_HOT:.0f} eV [%]")
    bottom.grid(True, alpha=0.3)

    # 4. the animation
    return animate(result) if len(result["movie"]) > 1 else None


def table(result: dict) -> str:
    """The fates of the ions, their energies at the target and the run's size."""
    born = result["born"]
    share = result["share"]
    profile = azimuthal(result)
    collided = sum(result["counts"].values()) / max(born, 1)
    cx = result["counts"].get("ar_ion_backscatter", 0) / max(born, 1)
    at_target = max(int(result["profile_count"].sum()), 1)
    rows = [
        "| | value |",
        "|---|---:|",
        f"| ions born | {born} |",
        f"| to the target, the substrate side, the outer edge | {share[0]:.2%}, "
        f"{share[1]:.2%}, {share[2]:.2%} |",
        f"| mean, median energy at the target | {result['mean_energy']:.1f} eV, "
        f"{result['median_energy']:.1f} eV |",
        f"| mean presheath gain e\\|E_z\\|(z_birth + LZ) of those ions | "
        f"{result['presheath_gain']:.1f} eV |",
        f"| at the target above {E_HOT:.0f} eV | {result['hot_count'] / at_target:.1%} |",
        f"| mean angle to the normal at the target | {result['incidence']:.0f}° |",
        f"| collisions per ion (of them charge exchange) | {collided:.2f} ({cx:.2f}) |",
        f"| mean time to the target | {result['lifetime'] * 1e6:.2f} µs |",
        f"| azimuthal shift birth to target at R0 (lab), mean and rms | "
        f"{result['shift_mean'] * 1e3:.1f} mm, {result['shift_rms'] * 1e3:.1f} mm |",
        f"| radial shift birth to target, mean and rms | "
        f"{result['radial_mean'] * 1e3:.1f} mm, {result['radial_rms'] * 1e3:.1f} mm |",
        f"| target flux, max / min over the angle | {profile['flux'].max():.2f} / "
        f"{profile['flux'].min():.2f} |",
        f"| mean energy at the target, max / min over the angle | "
        f"{profile['mean_energy'].max():.1f} / {profile['mean_energy'].min():.1f} eV |",
        f"| period of the pattern | {PERIOD * 1e6:.2f} µs |",
        f"| wall time | {result['wall_time']:.0f} s |",
    ]
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    args = parser.parse_args()
    outcome = simulate(args.preset)
    print(table(outcome))
    animation = plot(outcome)  # an animation plays while it is referenced
    plt.show()
