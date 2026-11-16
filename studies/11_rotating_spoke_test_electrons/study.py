"""Rotating-spoke test electrons.

What do electrons do in a spoke that is given, not computed? The background, built with byob,
is a 2D slice parallel to a circular target at the height of the racetrack ($R_0 = 3$ cm), in
the frame that rotates with the spokes. A 25 V radial potential well keeps the electrons on the
racetrack; on it sit three spokes, 12 V potential humps with a sharp leading edge and a long
tail, with hotter electrons inside (3 eV outside, up to 7 eV). The pattern rotates rigidly at
10 km/s in the direction of the electron Hall drift.

The magnetic field is radial, 30 mT at $R_0$, and a uniform axial field $E_z = -3$ kV/m drives
the Hall drift. The azimuthal field of a spoke adds an axial drift $E_\\theta \\times B_r$, so
there the electrons cross the magnetic field towards the anode. The electrons are test
particles of picnic's `Vlasov` model: Boris-pushed through the field gathered by byob, colliding
with argon at 0.5 Pa (plasmacoll's Monte Carlo collisions: elastic, excitation, ionization),
kept at a fixed number by a source that draws from the background. Those that reach the anode
plane are counted as the cross-field current. The run records one period of the pattern
(6.3 µs).

`python studies/11_rotating_spoke_test_electrons/study.py --movie spoke.mp4` also writes a movie
(needs ffmpeg).

Compared: nothing; this is a demonstration of the codes on a made-up background.

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
MOVIE_FRAMES = 30  # frames of the animation over the recorded period
N_TRACKS = 8  # electrons followed from the start of the recording
N_TRACKS_SHOWN = 4  # of them, the ones that stay longest, in the figures
# their positions every 0.3 ns, well below the radial bounce in the well (about 13 ns) and the
# gyration (1.2 ns), over the first 0.8 us of the recording
TRACK_EVERY = 3  # steps
TRACK_WINDOW = 8.0e-7  # [s]
TRACK_FRAMES = 60  # frames of the animation of the followed electrons
TRACK_TAIL = 1.0e-7  # length of their trails in it [s]
MOVIE_MARKERS = 1350  # markers per frame of the animation


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
    from picnic.particles.refill_source import sample_background
    from plasmacoll.constants import ELECTRON_MASS

    points, velocities = sample_background(background, rng, count, ELECTRON_MASS)
    x, y = rotate(points[0], points[1], OMEGA * t)
    positions = np.stack([x, y, rng.uniform(-LZ, LZ, count)])
    e, b = e_field(background, positions, t), b_field(x, y)
    velocities += np.cross(e.T, b.T).T / np.sum(b * b, axis=0)
    return positions, velocities


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
    """Push and collide the electrons (picnic's Vlasov); return the frames and averages."""
    import astropy.units as u
    from picnic.models.vlasov import Vlasov
    from picnic.parameters import GridParams, VlasovConfig
    from picnic.parameters.bc_params import ParticleBCs, ParticleBoundaryCondition
    from picnic.particles.refill_source import RefillSource
    from picnic.particles.species_collection import Particles
    from picnic.particles.wall_hits import wall_label
    from picnic.post_processing.rotating_frame import RotatingFrameMoments
    from picnic.simulation import Simulation
    from plasmacoll.constants import ELECTRON_MASS

    size = PRESETS[preset]
    background = build_background()
    n_burn = int(size["burn_in"] * PERIOD / DT)
    per_frame = max(1, int(round(size["recorded"] * PERIOD / DT / size["frames"])))
    t_burn = n_burn * DT
    steps = n_burn + size["frames"] * per_frame

    # the slab between the cathode sheath (reflects) and the anode plane (absorbs); the
    # side faces at +-L absorb (the hand-written loop cut at r = 0.95 L)
    kill, reflect = (
        ParticleBoundaryCondition("kill"),
        ParticleBoundaryCondition("reflect"),
    )
    config = VlasovConfig(
        grid=GridParams(
            ndim=3,
            nvel=3,
            num_gridpoints=(5, 5, 5),
            lower_bound=(-L, -L, -LZ),
            upper_bound=(L, L, LZ),
            coordinates="cartesian",
            domain_decomposition=(False, False, False),
        ),
        bcs_particles=ParticleBCs(kill, kill, kill, kill, reflect, kill),
        dt=DT,
    )
    model = Vlasov(
        config,
        e_field=lambda positions, t: e_field(background, positions.T, t).T,
        b_field=lambda positions: b_field(positions[:, 0], positions[:, 1]).T,
        fields_on_grid=False,
        collisions=collisions(),
        collision_interval=COLLIDE_EVERY,
        record_untracked_products=True,
    )
    electrons = Particles(
        "e",
        name="e",
        number_density=0.0 / u.m**3,
        temperature=0.0 * u.eV,
        macroparticle_weight=1.0,
        injection_method="manual",
        ndim=3,
        nvel=3,
    )
    electrons.set_pusher_args(model.domain.grid)
    model.particles.add(electrons)

    def sample(rng, count, t):
        return sample_electrons(background, rng, count, t)

    model.add_particle_source("e", RefillSource(size["markers"], sample, seed=0))

    # the spoke-frame averages: markers and energy every frame, ionizations every step
    spoke = RotatingFrameMoments("e", OMEGA, (-L, -L), (L, L), NMAP)
    edges = np.linspace(-np.pi, np.pi, N_BINS + 1)
    sums = {k: np.zeros(N_BINS) for k in ("ion", "energy", "count")}
    rng = np.random.default_rng(0)
    frames, tracks = [], []
    state = {"frame_ion": map_field(), "ions": 0, "tracked": None, "hits": 0}

    def ionizations(sim):
        births = sim.model.take_untracked_products().get("Ar+")
        if births is None or sim.model.time <= t_burn:
            return
        where = births.positions
        state["frame_ion"].deposit_density(where[:, :2].T, np.ones(len(where)))
        spoke.add_events("ionization", where, births.times)
        xs, ys = rotate(where[:, 0], where[:, 1], -OMEGA * births.times)
        sums["ion"] += np.histogram(np.arctan2(ys, xs), edges)[0]
        state["ions"] += len(where)

    def follow(sim):
        t = sim.model.time
        if not (t_burn <= t < t_burn + TRACK_WINDOW):
            return
        markers = electrons.markers
        n = markers.current_length
        ids, alive = markers.marker_ids[:n], markers.alive[:n]
        p = markers.positions[:, :n]
        if state["tracked"] is None:
            near = np.flatnonzero(
                alive & (np.abs(np.hypot(p[0], p[1]) - R0) < 0.5 * WR)
            )
            state["tracked"] = ids[rng.choice(near, size=N_TRACKS, replace=False)]
            state["track_start"] = t
        sample_ = np.full((N_TRACKS, 3), np.nan)
        for k, marker in enumerate(state["tracked"]):
            where = np.flatnonzero((ids == marker) & alive)
            if where.size:
                sample_[k] = p[:, where[0]]
        tracks.append(sample_)

    def frame(sim):
        t = sim.model.time
        done = sim.model.iteration - n_burn
        if done <= 0 or done % per_frame:
            return
        spoke(sim)
        markers = electrons.markers
        n = markers.current_length
        alive = markers.alive[:n]
        p, v = markers.positions[:, :n][:, alive], markers.velocities[:, :n][:, alive]
        energy = 0.5 * ELECTRON_MASS * np.sum(v * v, axis=0) / QE
        bins = np.clip(
            np.digitize(spoke_angle(p[0], p[1], t), edges) - 1, 0, N_BINS - 1
        )
        near = np.abs(np.hypot(p[0], p[1]) - R0) < WR
        sums["energy"] += np.bincount(bins[near], energy[near], N_BINS)
        sums["count"] += np.bincount(bins[near], minlength=N_BINS)
        anode_hits = model.wall_hits.get("e")
        total = (
            0
            if anode_hits is None
            else int(np.sum((anode_hits.wall == anode) & (anode_hits.time > t_burn)))
        )
        show = rng.choice(p.shape[1], size=min(N_SHOW, p.shape[1]), replace=False)
        frames.append(
            {
                "t": t,
                "x": p[0, show].astype(np.float32),
                "y": p[1, show].astype(np.float32),
                "energy": energy[show].astype(np.float32),
                "ion": state["frame_ion"].to_numpy()[..., 0].astype(np.float32),
                "ions": state["ions"],
                "anode": total - state["hits"],
            }
        )
        state["frame_ion"], state["ions"], state["hits"] = map_field(), 0, total
        print(
            f"frame {len(frames)}/{size['frames']}  t = {t * 1e6:.2f} us  "
            f"<E> = {energy.mean():.1f} eV  {time.time() - start:.0f} s"
        )

    anode = wall_label(2, "max")
    sim = Simulation(model, t_end=steps * DT)
    sim.add_callback(ionizations, every=1)
    sim.add_callback(follow, every=TRACK_EVERY)
    sim.add_callback(frame, every=1)
    start = time.time()
    sim.run()

    # the anode flux of the recorded part, from the wall hits
    hits = model.wall_hits["e"]
    on = (hits.wall == anode) & (hits.time > t_burn)
    angles = spoke_angle(hits.position[on, 0], hits.position[on, 1], hits.time[on])
    spoke.add_events("anode", hits.position[on], hits.time[on])
    frame_ds = spoke.dataset()
    sample_ = background.sample(resolution=NMAP)
    return {
        "preset": preset,
        "frames": frames,
        "frame_time": per_frame * DT,
        "angles": 0.5 * (edges[1:] + edges[:-1]),
        "ion_profile": sums["ion"],
        "anode_profile": np.histogram(angles, edges)[0].astype(float),
        "energy_profile": sums["energy"] / np.maximum(sums["count"], 1),
        "spoke_ion": frame_ds["ionization"].values.astype(np.float32),
        "spoke_maps": {
            "density": frame_ds["density"].values * frame_ds.attrs["samples"],
            "energy": frame_ds["mean_energy"].fillna(0).values
            * frame_ds["density"].values
            * frame_ds.attrs["samples"],
            "anode": frame_ds["anode"].values,
        },
        "tracks": np.array(tracks, dtype=np.float32).transpose(1, 0, 2),
        "track_start": state.get("track_start", t_burn),
        "track_dt": TRACK_EVERY * DT,
        "axis": sample_["x"].values,
        "potential": sample_["potential"].values.astype(np.float32),
        "wall_time": time.time() - start,
    }


def shrink(keep: dict) -> dict:
    """The large arrays of the cached result in half precision, under the 500 kB limit.

    The followed electrons that stay longest, x and y only (to 15 um); the spoke-frame
    ionization relative to its maximum; the potential map.
    """
    tracks = keep["tracks"]
    kept = np.sort(
        np.argsort(-np.isfinite(tracks[:, :, 0]).sum(axis=1))[:N_TRACKS_SHOWN]
    )
    keep["tracks"] = tracks[kept, :, :2].astype(np.float16)
    keep["potential"] = keep["potential"].astype(np.float16)  # +-25 V to 0.01 V
    ion = keep["spoke_ion"]
    keep["spoke_ion"] = (ion / max(float(ion.max()), 1e-30)).astype(np.float16)
    return keep


def compact(full: dict) -> dict:
    """The result for the cache: four ionization snapshots and the frames of the animation."""
    frames = full["frames"]
    picks = np.linspace(0, len(frames), SNAPSHOTS, endpoint=False).astype(int)
    window = max(1, len(frames) // 24)
    snapshots = []
    for i in picks:
        around = [frames[j % len(frames)] for j in range(i - window, i + window + 1)]
        # the ionization around the instant (the markers are in the animation)
        snapshots.append(
            {
                "t": frames[i]["t"],
                "ion": np.mean([f["ion"] for f in around], axis=0),
            }
        )
    # the ionization maps relative to their common maximum (arbitrary units), in half precision
    peak = max(float(s["ion"].max()) for s in snapshots) or 1.0
    for snapshot in snapshots:
        snapshot["ion"] = (snapshot["ion"] / peak).astype(np.float16)
    keep = {k: v for k, v in full.items() if k not in ("frames", "spoke_maps")}
    maps = full["spoke_maps"]
    count = maps["density"]
    keep["spoke_maps"] = {
        # markers per cell over the period, relative to the maximum
        "density": (count / max(count.max(), 1e-30)).astype(np.float16),
        # the mean energy of the markers in a cell [eV], where there are any
        "energy": np.where(
            count > 0.02 * count.max(),
            maps["energy"] / np.maximum(count, 1e-30),
            np.nan,
        ).astype(np.float16),
        "anode": (maps["anode"] / max(maps["anode"].max(), 1e-30)).astype(np.float16),
    }
    keep["snapshots"] = snapshots
    # the animation: a few dozen frames over the period, a few hundred markers each
    movie = np.linspace(0, len(frames), min(MOVIE_FRAMES, len(frames)), endpoint=False)
    keep["movie"] = [
        {
            "t": frames[i]["t"],
            "x": frames[i]["x"][:MOVIE_MARKERS].astype(np.float16),
            "y": frames[i]["y"][:MOVIE_MARKERS].astype(np.float16),
            "energy": frames[i]["energy"][:MOVIE_MARKERS].astype(np.float16),
        }
        for i in movie.astype(int)
    ]
    keep["times"] = np.array([f["t"] for f in frames])
    keep["ion_rate"] = np.array([f["ions"] for f in frames]) / full["frame_time"]
    keep["anode_rate"] = np.array([f["anode"] for f in frames]) / full["frame_time"]
    return shrink(keep)


def simulate(preset: str = "quick") -> dict:
    """Run the preset; return the snapshots, the period averages and the time signals."""
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


def smooth_periodic(values, width: int = 5):
    """A running mean over ``width`` bins of a periodic profile."""
    from scipy.ndimage import uniform_filter1d

    return uniform_filter1d(np.asarray(values, dtype=float), width, mode="wrap")


def profiles(result: dict) -> list[tuple[str, np.ndarray]]:
    """The normalized azimuthal profiles at the racetrack in the spoke frame."""
    from scipy.interpolate import RegularGridInterpolator

    angles = result["angles"]
    axis = result["axis"]
    phi = RegularGridInterpolator((axis, axis), result["potential"].astype(float))(
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
    interp = RegularGridInterpolator((axis, axis), result["potential"].astype(float))
    t = result["times"]
    return interp(np.stack([R0 * np.cos(-OMEGA * t), R0 * np.sin(-OMEGA * t)], axis=-1))


def animate(result: dict):
    """The rotating spokes over the recorded period: the lab-frame potential and the electrons."""
    from matplotlib.animation import FuncAnimation

    movie = result["movie"]
    t0 = result["times"][0]
    extent = (-L * 100, L * 100, -L * 100, L * 100)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    image = ax.imshow(
        potential_lab(result, movie[0]["t"]).T,
        origin="lower",
        extent=extent,
        cmap="cividis",
        vmin=-12,
        vmax=12,
    )
    dots = ax.scatter(
        movie[0]["x"].astype(float) * 100,
        movie[0]["y"].astype(float) * 100,
        c=movie[0]["energy"].astype(float),
        s=4,
        cmap="plasma",
        vmin=0,
        vmax=30,
        lw=0,
    )
    fig.colorbar(image, ax=ax, label="potential [V]", shrink=0.85)
    fig.colorbar(dots, ax=ax, label="electron energy [eV]", shrink=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
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

    animation = FuncAnimation(fig, update, frames=len(movie), interval=120)
    # the slider of the web figure shows the time
    animation._plasma_sweep = ("t [µs]", [(f["t"] - t0) * 1e6 for f in movie])
    return animation


def track_times(result: dict) -> np.ndarray:
    """The times of the samples of the followed electrons [s]."""
    return result["track_start"] + result["track_dt"] * np.arange(
        result["tracks"].shape[1]
    )


def animate_tracks(result: dict):
    """The followed electrons in the spoke frame over the potential, with their recent trails."""
    from matplotlib.animation import FuncAnimation

    times = track_times(result)
    tracks = result["tracks"].astype(float)
    # in the spoke frame, where the potential stands still (one picture for all frames); drawn
    # every third sample, 0.9 ns, still about 14 points per radial bounce
    xs, ys = rotate(tracks[:, :, 0], tracks[:, :, 1], -OMEGA * times)
    xs, ys, times = xs[:, ::3] * 100, ys[:, ::3] * 100, times[::3]
    # the last sample of each frame, the frames evenly over the recorded samples
    ends = np.linspace(0, len(times) - 1, min(TRACK_FRAMES, len(times))).astype(int)
    tail = int(round(TRACK_TAIL / (times[1] - times[0]))) if len(times) > 1 else 1
    extent = (-L * 100, L * 100, -L * 100, L * 100)
    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    image = ax.imshow(
        result["potential"].astype(float).T,
        origin="lower",
        extent=extent,
        cmap="cividis",
        vmin=-12,
        vmax=12,
    )
    fig.colorbar(image, ax=ax, label="potential [V]", shrink=0.85)
    colors = [f"C{k % 10}" for k in range(len(tracks))]
    trails = [ax.plot([], [], color=color, lw=1.2)[0] for color in colors]
    heads = ax.scatter(xs[:, 0], ys[:, 0], s=24, c=colors, zorder=3, lw=0)
    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("x [cm] (spoke frame)")
    ax.set_ylabel("y [cm]")

    def update(i):
        end = ends[i]
        shown = slice(max(0, end - tail), end + 1)
        for k, trail in enumerate(trails):
            trail.set_data(xs[k, shown], ys[k, shown])
        heads.set_offsets(np.c_[xs[:, end], ys[:, end]])
        return heads, *trails

    animation = FuncAnimation(fig, update, frames=len(ends), interval=120)
    animation._plasma_sweep = ("t [ns]", [(times[e] - times[0]) * 1e9 for e in ends])
    return animation


def plot(result: dict):
    """Draw the animations and the figures of the period; return the animations."""
    from scipy.ndimage import gaussian_filter

    animations = [animate(result), animate_tracks(result)]
    extent = (-L * 100, L * 100, -L * 100, L * 100)
    axis_cm = result["axis"] * 100
    snapshots = result["snapshots"]
    t0 = result["times"][0]
    # the ionization at four instants, in two figures of two stacked maps (the potential and the
    # electrons are in the animation), with one colour scale for all four
    ion_max = (
        max(
            np.percentile(gaussian_filter(s["ion"].astype(float), 1.2), 99.7)
            for s in snapshots
        )
        or 1.0
    )
    for pair in (snapshots[:2], snapshots[2:]):
        fig, axes = plt.subplots(
            2, 1, figsize=(5.6, 9.0), sharex=True, constrained_layout=True
        )
        for ax, snap in zip(axes, pair, strict=True):
            phi = potential_lab(result, snap["t"])
            ionization = ax.imshow(
                gaussian_filter(snap["ion"].astype(float), 1.2).T,
                origin="lower",
                extent=extent,
                cmap="magma",
                vmin=0,
                vmax=ion_max,
            )
            ax.contour(
                axis_cm,
                axis_cm,
                phi.T,
                levels=[0, 6],
                colors="#66ccff",
                linewidths=0.6,
            )
            ax.set_title(f"t = {(snap['t'] - t0) * 1e6:.2f} µs")
            ax.set_aspect("equal")
            ax.set_xlim(-4.5, 4.5)
            ax.set_ylim(-4.5, 4.5)
            ax.set_ylabel("y [cm]")
        axes[-1].set_xlabel("x [cm]")
        fig.colorbar(ionization, ax=axes, label="ionization events [arb.]", shrink=0.8)

    # the followed electrons in the spoke frame (the pattern stands still there): they bounce
    # radially in the well, drift around the racetrack and cross the spokes
    fig, ax = plt.subplots(figsize=(6.4, 5.6), constrained_layout=True)
    image = ax.imshow(
        result["potential"].astype(float).T,
        origin="lower",
        extent=extent,
        cmap="cividis",
        vmin=-12,
        vmax=12,
    )
    times = track_times(result)
    tracks = result["tracks"].astype(float)
    for k, track in enumerate(tracks):
        xs, ys = rotate(track[:, 0], track[:, 1], -OMEGA * times)
        ax.plot(xs * 100, ys * 100, lw=1.0, color=f"C{k % 10}")
        start = np.flatnonzero(np.isfinite(xs))
        if start.size:
            ax.plot(
                xs[start[0]] * 100, ys[start[0]] * 100, "o", ms=4, color=f"C{k % 10}"
            )
    fig.colorbar(image, ax=ax, label="potential [V]", shrink=0.85)
    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("x [cm] (spoke frame)")
    ax.set_ylabel("y [cm]")

    # period averages over the whole plane, in the spoke frame
    maps = result["spoke_maps"]
    panels = [
        (
            gaussian_filter(result["spoke_ion"].astype(float), 1.0),
            "magma",
            "ionization [arb.]",
            "ionization",
        ),
        (
            gaussian_filter(maps["density"].astype(float), 1.0),
            "viridis",
            "electron density [arb.]",
            "density",
        ),
        (
            maps["energy"].astype(float),
            "plasma",
            "mean electron energy [eV]",
            "mean energy",
        ),
        (
            gaussian_filter(maps["anode"].astype(float), 1.0),
            "inferno",
            "cross-field flux to the anode [arb.]",
            "flux to the anode",
        ),
    ]
    for values, cmap, label, title in panels:
        fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
        image = ax.imshow(values.T, origin="lower", extent=extent, cmap=cmap)
        ax.contour(
            axis_cm,
            axis_cm,
            result["potential"].astype(float).T,
            levels=[0, 6],
            colors="#66ccff",
            linewidths=0.6,
        )
        fig.colorbar(image, ax=ax, label=label, shrink=0.85)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
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
    return animations


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
    animation = plot(outcome)  # an animation plays while it is referenced
    plt.show()
