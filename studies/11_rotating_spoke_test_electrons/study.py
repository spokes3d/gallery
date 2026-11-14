"""Test electrons in a prescribed rotating spoke.

What do electrons do in a spoke that is given, not computed? The background is made up: a
2D slice parallel to a circular target at the height of the racetrack ($R_0 = 3$ cm), built
with byob in the frame that rotates with the spokes. A radial potential well of 25 V keeps the
electrons on the racetrack; on it sit three spokes, each a 12 V potential hump with a sharp
leading edge and a long tail, with a hotter electron temperature (3 eV outside, up to 7 eV
inside) and the density from Boltzmann's relation. In the lab frame the pattern rotates rigidly
at 10 km/s in the direction of the electron Hall drift, so the field at an electron is gathered
(mpifield's compiled kernels, through byob) at its position rotated back into the spoke frame,
and the result is rotated forward again.

The magnetic field is radial, $B_r(r)$ with 30 mT at $R_0$, and a uniform axial field
$E_z = -3$ kV/m (the cathode presheath) drives the Hall drift $E_z/B_r = 10^5$ m/s. The
azimuthal field of a spoke gives an axial drift $E_\\theta \\times B_r$: there the electrons
cross the magnetic field towards the anode ($+z$) and gain $e|E_z|\\,\\Delta z$. The electrons
are stored in an mpiparticles `ParticleStore`, pushed with the Boris scheme
($\\Delta t = 10^{-10}$ s, $\\omega_{ce}\\Delta t = 0.53$) and collide with argon at 0.5 Pa
(plasmacoll's Monte Carlo collisions: elastic, one lumped excitation at 11.5 eV, Lotz
ionization from the 3p shell). Electrons that reach the anode plane $z = +5$ mm are absorbed
and counted as the cross-field current, the cathode sheath at $z = -5$ mm reflects them, and the
population is topped up from the background's density. After a burn-in of 0.4 periods the run
records one period of the pattern ($2\\pi/(3\\Omega) = 6.3\\,\\mu$s).

`python studies/11_rotating_spoke_test_electrons/study.py --movie spoke.mp4` also writes a movie of
the period (needs ffmpeg).

Compared: nothing; this is a demonstration of the codes on a made-up background. Not compared:
any measurement. The background is not self-consistent (no Poisson equation, the electrons do
not change it), the cross sections are rough stand-ins (constant elastic, a made-up excitation,
Lotz ionization), the slab has no axial structure apart from $E_z$, and the secondary electrons
of ionization start at a random height, since the collisions see only $(x, y)$.

Reference: the picture of spokes as rotating ionization zones with a potential hump, a
leading-edge field and enhanced cross-field transport, e.g. Hecimovic and von Keudell,
J. Phys. D 51 (2018) 453001.
"""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

#: Markers, the number of recorded frames per period and the fraction of it recorded.
PRESETS = {
    "smoke": {"markers": 300, "frames": 4, "burn_in": 0.002, "recorded": 0.01},
    "quick": {"markers": 10000, "frames": 120, "burn_in": 0.4, "recorded": 1.0},
}

QE = 1.602176634e-19
L = 0.05  # half width of the domain [m]
NG = 201  # grid nodes per axis of the background
NMAP = 101  # nodes per axis of the ionization maps
R0 = 0.03  # racetrack radius [m]
WR = 0.006  # racetrack half width [m]
B0 = 0.03  # radial field at the racetrack [T]
EZ = -3.0e3  # axial (presheath) field [V/m]
LZ = 0.005  # cathode at -LZ, anode plane at +LZ [m]
V_RING = 25.0  # depth of the radial well [V]
V_SPOKE = 12.0  # potential hump of a spoke [V]
TE0, DTE = 3.0, 4.0  # electron temperature outside, and extra inside a spoke [eV]
M_SPOKES = 3
SPOKE_SPEED = 1.0e4  # at R0, in the direction of the Hall drift [m/s]
OMEGA = -SPOKE_SPEED / R0  # [rad/s]; negative: the sense of the Hall drift
PERIOD = 2 * np.pi / (M_SPOKES * abs(OMEGA))  # the pattern repeats after this [s]
P_AR, T_GAS = 0.5, 300.0  # [Pa], [K]
DT = 1.0e-10  # [s]
COLLIDE_EVERY = 5
N_SHOW = 3000  # markers kept per frame for the figures
N_BINS = 180  # azimuthal bins
SNAPSHOTS = 4


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


def sample_electrons(background, rng, count, t):
    """Positions from the density, Maxwellian velocities at the local Te plus the E x B drift."""
    from plasmacoll.constants import ELECTRON_MASS

    density = background.bulk["density"].ravel()
    nodes = rng.choice(density.size, size=count, p=density / density.sum())
    points = background.grid.points().T[nodes]
    points = points + (rng.random(points.shape) - 0.5) * background.grid.spacing
    te = background.evaluate(points)["electron_temperature"]
    x, y = rotate(points[:, 0], points[:, 1], OMEGA * t)
    positions = np.stack([x, y, rng.uniform(-LZ, LZ, count)])
    velocities = rng.normal(size=(3, count)) * np.sqrt(QE * te / ELECTRON_MASS)
    e, b = e_field(background, positions, t), b_field(x, y)
    velocities += np.cross(e.T, b.T).T / np.sum(b * b, axis=0)
    return positions, velocities


def boris(positions, velocities, e, b, dt):
    """One Boris step of electrons, in place."""
    from plasmacoll.constants import ELECTRON_MASS

    q_m = -QE / ELECTRON_MASS
    v_minus = velocities + 0.5 * dt * q_m * e
    t_vec = 0.5 * dt * q_m * b
    s_vec = 2 * t_vec / (1 + np.sum(t_vec * t_vec, axis=0))
    v_prime = v_minus + np.cross(v_minus.T, t_vec.T).T
    v_plus = v_minus + np.cross(v_prime.T, s_vec.T).T
    velocities[:] = v_plus + 0.5 * dt * q_m * e
    positions += dt * velocities


def collisions():
    """Electron-argon Monte Carlo collisions with stand-in cross sections."""
    from plasmacoll import (
        CollisionProcess,
        CrossSection,
        MonteCarloCollisions,
        NeutralBackground,
    )
    from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, kelvin_to_ev
    from plasmacoll.metals import LotzSubshell, lotz_table

    def excitation(energy):
        return 1.5e-21 * np.clip(energy - 11.5, 0, None) ** 0.5 * np.exp(-energy / 200)

    processes = [
        CollisionProcess(
            "elastic", "Ar", CrossSection.constant(1.0e-19), energy_frame="lab"
        ),
        CollisionProcess(
            "excitation",
            "Ar",
            CrossSection.from_function(
                excitation, np.geomspace(1e-3, 1e4, 400), threshold=11.5
            ),
            energy_frame="lab",
        ),
        CollisionProcess(
            "ionization",
            "Ar",
            lotz_table([LotzSubshell(15.76, 6)], label="Ar Lotz 3p"),
            energy_frame="lab",
            products={"electron": "e", "ion": "Ar+"},
        ),
    ]
    argon = NeutralBackground(
        "Ar", P_AR / (1.380649e-23 * T_GAS), kelvin_to_ev(T_GAS), 39.95 * ATOMIC_MASS
    )
    return MonteCarloCollisions(
        species_masses={"e": ELECTRON_MASS, "Ar+": 39.95 * ATOMIC_MASS},
        backgrounds=[argon],
        processes={"e": processes},
        seed=1,
    )


def map_field():
    """An empty lab-sized mpifield grid for the ionization maps."""
    from mpifield import Field

    return Field(
        ndim=2, lower_bound=(-L, -L), upper_bound=(L, L), num_gridpoints=(NMAP, NMAP)
    )


def spoke_angle(x, y, t):
    """The angle in the spoke frame, in [-pi, pi)."""
    return np.angle(np.exp(1j * (np.arctan2(y, x) - OMEGA * t)))


def run(preset: str) -> dict:
    """Push and collide the electrons; return the per-frame data and the period averages."""
    from mpiparticles import ParticleStore
    from plasmacoll.constants import ELECTRON_MASS

    size = PRESETS[preset]
    rng = np.random.default_rng(0)
    background = build_background()
    mcc = collisions()
    store = ParticleStore(ndim=3, nvel=3)
    store.add(*sample_electrons(background, rng, size["markers"], 0.0))

    n_burn = int(size["burn_in"] * PERIOD / DT)
    per_frame = max(1, int(round(size["recorded"] * PERIOD / DT / size["frames"])))
    edges = np.linspace(-np.pi, np.pi, N_BINS + 1)
    sums = {k: np.zeros(N_BINS) for k in ("ion", "anode", "energy", "count")}
    spoke_ion = map_field()
    frame_ion, anode_count, ion_count = map_field(), 0, 0
    frames = []
    t = 0.0
    start = time.time()
    for step in range(n_burn + size["frames"] * per_frame):
        recording = step >= n_burn
        n = store.length
        p, v, alive = store.positions[:, :n], store.velocities[:, :n], store.alive[:n]
        boris(p, v, e_field(background, p, t + 0.5 * DT), b_field(p[0], p[1]), DT)
        t += DT

        # the cathode sheath reflects; the anode plane and the domain edge absorb
        low = p[2] < -LZ
        p[2, low] = -2 * LZ - p[2, low]
        v[2, low] *= -1
        anode = (p[2] > LZ) & alive
        lost = (np.hypot(p[0], p[1]) > 0.95 * L) & alive
        if recording and anode.any():
            angles = spoke_angle(p[0, anode], p[1, anode], t)
            sums["anode"] += np.histogram(angles, edges)[0]
            anode_count += int(anode.sum())

        if step % COLLIDE_EVERY == 0:
            index = np.flatnonzero(alive & ~anode & ~lost)
            velocities = np.ascontiguousarray(v[:, index].T)
            result = mcc.collide(
                "e",
                np.ascontiguousarray(p[:2, index].T),
                velocities,
                np.ones(index.size),
                COLLIDE_EVERY * DT,
                in_place=True,
            )
            v[:, index] = velocities.T
            ions = result.created.get("Ar+", [])
            if ions and recording:
                where = np.concatenate([np.asarray(m.positions) for m in ions])
                frame_ion.deposit_density(where.T, np.ones(len(where)))
                xs, ys = rotate(where[:, 0], where[:, 1], -OMEGA * t)
                spoke_ion.deposit_density(np.stack([xs, ys]), np.ones(len(where)))
                sums["ion"] += np.histogram(np.arctan2(ys, xs), edges)[0]
                ion_count += len(where)
            for batch in result.created.get("e", []):
                xy = np.asarray(batch.positions)
                z = rng.uniform(-LZ, LZ, len(xy))
                store.add(
                    np.stack([xy[:, 0], xy[:, 1], z]), np.asarray(batch.velocities).T
                )

        dead = anode | lost
        if dead.any():
            store.deactivate(np.concatenate([dead, np.zeros(store.length - n, bool)]))
        store.compact_if_needed()
        missing = size["markers"] - store.num_alive
        if missing > 0:
            store.add(*sample_electrons(background, rng, missing, t))

        if recording and (step - n_burn + 1) % per_frame == 0:
            store.compact()
            n = store.length
            p, v = store.positions[:, :n], store.velocities[:, :n]
            energy = 0.5 * ELECTRON_MASS * np.sum(v * v, axis=0) / QE
            bins = np.clip(
                np.digitize(spoke_angle(p[0], p[1], t), edges) - 1, 0, N_BINS - 1
            )
            near = np.abs(np.hypot(p[0], p[1]) - R0) < WR
            sums["energy"] += np.bincount(bins[near], energy[near], N_BINS)
            sums["count"] += np.bincount(bins[near], minlength=N_BINS)
            show = rng.choice(n, size=min(N_SHOW, n), replace=False)
            frames.append(
                {
                    "t": t,
                    "x": p[0, show].astype(np.float32),
                    "y": p[1, show].astype(np.float32),
                    "energy": energy[show].astype(np.float32),
                    "ion": frame_ion.to_numpy()[..., 0].astype(np.float32),
                    "ions": ion_count,
                    "anode": anode_count,
                }
            )
            frame_ion, anode_count, ion_count = map_field(), 0, 0
            print(
                f"frame {len(frames)}/{size['frames']}  t = {t * 1e6:.2f} us  "
                f"<E> = {energy.mean():.1f} eV  {time.time() - start:.0f} s"
            )

    sample = background.sample(resolution=NMAP)
    return {
        "preset": preset,
        "frames": frames,
        "frame_time": per_frame * DT,
        "angles": 0.5 * (edges[1:] + edges[:-1]),
        "ion_profile": sums["ion"],
        "anode_profile": sums["anode"],
        "energy_profile": sums["energy"] / np.maximum(sums["count"], 1),
        "spoke_ion": spoke_ion.to_numpy()[..., 0].astype(np.float32),
        "axis": sample["x"].values,
        "potential": sample["potential"].values.astype(np.float32),
        "wall_time": time.time() - start,
    }


def compact(full: dict) -> dict:
    """The result for the cache: four snapshots instead of every frame."""
    frames = full["frames"]
    picks = np.linspace(0, len(frames), SNAPSHOTS, endpoint=False).astype(int)
    window = max(1, len(frames) // 24)
    snapshots = []
    for i in picks:
        around = [frames[j % len(frames)] for j in range(i - window, i + window + 1)]
        snapshots.append(
            {**frames[i], "ion": np.mean([f["ion"] for f in around], axis=0)}
        )
    keep = {k: v for k, v in full.items() if k != "frames"}
    keep["snapshots"] = snapshots
    keep["times"] = np.array([f["t"] for f in frames])
    keep["ion_rate"] = np.array([f["ions"] for f in frames]) / full["frame_time"]
    keep["anode_rate"] = np.array([f["anode"] for f in frames]) / full["frame_time"]
    return keep


def simulate(preset: str = "quick") -> dict:
    """Run the preset; return the snapshots, the period averages and the time signals."""
    return compact(run(preset))


def potential_lab(result: dict, t: float):
    """The lab-frame potential on the map grid at time ``t`` (the rotated spoke frame)."""
    from scipy.interpolate import RegularGridInterpolator

    axis = result["axis"]
    interp = RegularGridInterpolator(
        (axis, axis), result["potential"], bounds_error=False, fill_value=None
    )
    x, y = np.meshgrid(axis, axis, indexing="ij")
    xs, ys = rotate(x, y, -OMEGA * t)
    return interp(np.stack([xs, ys], axis=-1))


def smooth_periodic(values, width: int = 5):
    """A running mean over ``width`` bins of a periodic profile."""
    from scipy.ndimage import uniform_filter1d

    return uniform_filter1d(np.asarray(values, dtype=float), width, mode="wrap")


def profiles(result: dict) -> list[tuple[str, np.ndarray]]:
    """The normalized azimuthal profiles at the racetrack in the spoke frame."""
    from scipy.interpolate import RegularGridInterpolator

    angles = result["angles"]
    axis = result["axis"]
    phi = RegularGridInterpolator((axis, axis), result["potential"])(
        np.stack([R0 * np.cos(angles), R0 * np.sin(angles)], axis=-1)
    )

    def norm(values):
        smoothed = smooth_periodic(values)
        return smoothed / max(smoothed.max(), 1e-30)

    return [
        ("potential at $R_0$", (phi - phi.min()) / max(np.ptp(phi), 1e-30)),
        ("mean electron energy", norm(result["energy_profile"])),
        ("ionization rate", norm(result["ion_profile"])),
        ("cross-field e$^-$ flux to the anode", norm(result["anode_profile"])),
    ]


def potential_probe(result: dict):
    """The potential at the lab point (R0, 0): the spoke-frame angle -OMEGA t."""
    from scipy.interpolate import RegularGridInterpolator

    axis = result["axis"]
    interp = RegularGridInterpolator((axis, axis), result["potential"])
    t = result["times"]
    return interp(np.stack([R0 * np.cos(-OMEGA * t), R0 * np.sin(-OMEGA * t)], axis=-1))


def plot(result: dict) -> None:
    """Draw the snapshots, the spoke-frame ionization map, the profiles and the time signals."""
    from scipy.ndimage import gaussian_filter

    extent = (-L * 100, L * 100, -L * 100, L * 100)
    axis_cm = result["axis"] * 100
    snapshots = result["snapshots"]
    t0 = result["times"][0]
    fig, axes = plt.subplots(
        2, len(snapshots), figsize=(3.2 * len(snapshots), 6.6), constrained_layout=True
    )
    axes = np.asarray(axes).reshape(2, -1)
    ion_max = (
        max(np.percentile(gaussian_filter(s["ion"], 1.2), 99.7) for s in snapshots)
        or 1.0
    )
    for top, bottom, snap in zip(axes[0], axes[1], snapshots, strict=True):
        phi = potential_lab(result, snap["t"])
        image = top.imshow(
            phi.T, origin="lower", extent=extent, cmap="cividis", vmin=-12, vmax=12
        )
        dots = top.scatter(
            snap["x"] * 100,
            snap["y"] * 100,
            c=snap["energy"],
            s=0.6,
            cmap="plasma",
            vmin=0,
            vmax=30,
            lw=0,
        )
        top.set_title(f"t = {(snap['t'] - t0) * 1e6:.2f} µs")
        ionization = bottom.imshow(
            gaussian_filter(snap["ion"], 1.2).T,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0,
            vmax=ion_max,
        )
        bottom.contour(
            axis_cm, axis_cm, phi.T, levels=[0, 6], colors="#66ccff", linewidths=0.6
        )
        for ax in (top, bottom):
            ax.set_aspect("equal")
            ax.set_xlim(-4.5, 4.5)
            ax.set_ylim(-4.5, 4.5)
            ax.set_xlabel("x [cm]")
        top.set_ylabel("y [cm]")
        bottom.set_ylabel("y [cm]")
    fig.colorbar(image, ax=axes[0], label="potential [V]", shrink=0.8)
    fig.colorbar(dots, ax=axes[0], label="electron energy [eV]", shrink=0.8)
    fig.colorbar(ionization, ax=axes[1], label="ionization events [arb.]", shrink=0.8)

    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    image = ax.imshow(
        gaussian_filter(result["spoke_ion"], 1.0).T,
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    ax.contour(
        axis_cm,
        axis_cm,
        result["potential"].T,
        levels=[0, 6],
        colors="#66ccff",
        linewidths=0.7,
    )
    fig.colorbar(image, ax=ax, label="ionization events per area [arb.]")
    ax.set_aspect("equal")
    ax.set_xlabel("x [cm] (spoke frame)")
    ax.set_ylabel("y [cm]")

    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    for label, values in profiles(result):
        ax.plot(np.degrees(result["angles"]), values, label=label)
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_ylim(0, 1.3)
    ax.set_xlabel(
        "angle in the spoke frame [deg] (spokes move towards negative angles)"
    )
    ax.set_ylabel("normalized")
    ax.legend(fontsize=8, ncols=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    times = (result["times"] - t0) * 1e6
    width = max(1, len(times) // 13)
    for key, label in (
        ("ion_rate", "total ionization rate"),
        ("anode_rate", "cross-field electron current"),
    ):
        values = smooth_periodic(result[key], width)
        ax.plot(times, values / max(values.mean(), 1e-30), label=label)
    ax.set_xlabel("t [µs]")
    ax.set_ylabel("relative to the mean")
    probe_axis = ax.twinx()
    probe = potential_probe(result)
    probe_axis.plot(times, probe, "k--", lw=0.9, label="probe potential at (R0, 0)")
    probe_axis.set_ylabel("probe potential [V]")
    lines = ax.get_lines() + probe_axis.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)


def table(result: dict) -> str:
    """The spoke-frame peaks of the profiles and the run's size."""
    angles = np.degrees(result["angles"])
    rows = [
        "| | value |",
        "|---|---:|",
        f"| spokes, speed at R0 | {M_SPOKES}, {SPOKE_SPEED / 1e3:.0f} km/s |",
        f"| period of the pattern | {PERIOD * 1e6:.2f} µs |",
        f"| Hall drift E_z/B at R0 | {abs(EZ) / B0 / 1e3:.0f} km/s |",
        f"| markers, frames | {PRESETS[result['preset']]['markers']}, {len(result['times'])} |",
    ]
    ion = smooth_periodic(result["ion_profile"])
    anode = smooth_periodic(result["anode_profile"])
    rows += [
        f"| ionization rate, peak / mean over the ring | {ion.max() / max(ion.mean(), 1e-30):.1f} |",
        f"| cross-field flux, peak / mean over the ring | {anode.max() / max(anode.mean(), 1e-30):.1f} |",
        f"| angle of the first flux peak (spoke frame) | {angles[np.argmax(anode)]:.0f}° |",
        f"| wall time | {result['wall_time']:.0f} s |",
    ]
    return "\n".join(rows)


def movie(full: dict, path: str) -> None:
    """Write the movie of a full run (every frame), played twice so that it loops."""
    from matplotlib.animation import FFMpegWriter
    from scipy.ndimage import gaussian_filter, uniform_filter1d

    frames = full["frames"]
    extent = (-L * 100, L * 100, -L * 100, L * 100)
    axis_cm = full["axis"] * 100
    ion = uniform_filter1d(np.array([f["ion"] for f in frames]), 5, axis=0, mode="wrap")
    ion = np.array([gaussian_filter(m, 1.2) for m in ion])
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 6.6), constrained_layout=True)
    phi = potential_lab(full, frames[0]["t"])
    image = left.imshow(
        phi.T, origin="lower", extent=extent, cmap="cividis", vmin=-12, vmax=12
    )
    dots = left.scatter(
        frames[0]["x"] * 100,
        frames[0]["y"] * 100,
        c=frames[0]["energy"],
        s=3,
        cmap="plasma",
        vmin=0,
        vmax=30,
        lw=0,
    )
    fig.colorbar(image, ax=left, label="potential [V]", shrink=0.75)
    fig.colorbar(dots, ax=left, label="electron energy [eV]", shrink=0.75)
    ionization = right.imshow(
        ion[0].T,
        origin="lower",
        extent=extent,
        cmap="magma",
        vmin=0,
        vmax=np.percentile(ion, 99.7),
    )
    fig.colorbar(ionization, ax=right, label="ionization events [arb.]", shrink=0.75)
    contours = [right.contour(axis_cm, axis_cm, phi.T, levels=[0, 6], colors="#66ccff")]
    left.set_title("potential and test electrons (lab frame)")
    right.set_title("argon ionization by the electrons")
    for ax in (left, right):
        ax.set_aspect("equal")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
    title = fig.suptitle("")
    writer = FFMpegWriter(fps=24, bitrate=6000, extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, path, dpi=120):
        for _ in range(2):
            for i, frame in enumerate(frames):
                phi = potential_lab(full, frame["t"])
                image.set_data(phi.T)
                dots.set_offsets(np.c_[frame["x"] * 100, frame["y"] * 100])
                dots.set_array(frame["energy"])
                ionization.set_data(ion[i].T)
                contours[0].remove()
                contours[0] = right.contour(
                    axis_cm,
                    axis_cm,
                    phi.T,
                    levels=[0, 6],
                    colors="#66ccff",
                    linewidths=0.7,
                )
                title.set_text(
                    f"Electrons in a rotating spoke, t = {(frame['t'] - frames[0]['t']) * 1e6:.2f} µs"
                )
                writer.grab_frame()
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    parser.add_argument("--movie", metavar="PATH", help="also write a movie (mp4)")
    args = parser.parse_args()
    full = run(args.preset)
    if args.movie:
        movie(full, args.movie)
    outcome = compact(full)
    print(table(outcome))
    plot(outcome)
    plt.show()
