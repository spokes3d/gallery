"""Magnetron electron orbits.

Single electrons in the field of a circular planar magnetron, without collisions: they gyrate
about the field, bounce along the arch of the field lines and drift around the axis, along the
racetrack. The study pushes them with the full Lorentz force (mpiparticles' compiled Boris
scheme, $\\omega_{ce}\\Delta t < 0.1$) and compares their motion with guiding-centre theory.

The magnetic field is that of the balanced 2-inch magnetron of the study *Magnetron magnetic
field*. The electric field is prescribed and axial: a 300 V cathode sheath 1 mm thick and,
above it, a presheath field of 2 kV/m, typical of a DC magnetron. Each electron starts on the
field line whose apex is 4 mm above the target, at the bottom of its potential well
$U = \\mu B - e\\phi$, with a given kinetic energy and pitch angle.

The theory gives the bounce period $\\tau_b = \\oint ds/v_\\parallel$ in that well and the
bounce-averaged $E\\times B$, grad-B and curvature drifts, which all point around the axis in
the direction of $E\\times B$. $E\\times B$ dominates below a few eV, the grad-B drift above.
The push is measured through the guiding centres reconstructed from the orbits.

Compared: the bounce periods and azimuthal drift speeds of the push with the guiding-centre
values, over energies of 1 to 50 eV and pitch angles of 20° to 88°. The drifts agree to better
than 1 % and the bounce periods to about 1 %.

Reference: the guiding-centre drifts and the magnetic mirror in Chen, *Introduction to Plasma
Physics and Controlled Fusion*, 3rd ed. (Springer, 2016), chapter 2; the magnetron and its
electron trap in Lieberman and Lichtenberg, *Principles of Plasma Discharges and Materials
Processing*, 2nd ed. (Wiley, 2005), chapter 14.
"""

import argparse
import os
import time

# a few dozen markers: OpenMP threads in the compiled push only cost time (set before the
# kernel library is loaded)
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
from plasmacoll.constants import ELECTRON_MASS, ELEMENTARY_CHARGE
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ellipe, ellipk

PRESETS = {
    "smoke": {
        "energies": [5.0],
        "pitches": [70.0],
        "duration": 80e-9,
        "orbits": 1,
        "nodes": 6,
    },
    "full": {
        "energies": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0],
        "pitches": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0, 88.0],
        "duration": 1.6e-6,
        "orbits": 3,
        "nodes": 8,
    },
}

MU0 = 4.0e-7 * np.pi
#: Remanence [T], faces [m], radii [m] of the balanced 2-inch magnetron (assumed).
REMANENCE = 1.2
MAGNET_TOP, MAGNET_HEIGHT = -8.0e-3, 8.0e-3
CENTRE_RADIUS, RING_INNER, RING_OUTER = 12.0e-3, 20.0e-3, 23.3e-3
#: Prescribed potential (assumed): sheath voltage [V] and width [m], presheath field [V/m].
SHEATH_VOLTAGE, SHEATH_WIDTH, PRESHEATH_FIELD = 300.0, 1.0e-3, 2.0e3
#: Apex height [m] of the field line the electrons start on.
APEX_HEIGHT = 4.0e-3
#: Time step [s] and recording interval [steps].
DT, RECORD = 5.0e-12, 10
#: Fixed scan values: pitch angle [deg] of the energy scan, energy [eV] of the pitch scan.
SCAN_PITCH, SCAN_ENERGY = 75.0, 5.0
#: The orbits drawn: (energy [eV], pitch angle [deg]); the last one reaches the sheath.
ORBITS = ((5.0, 75.0), (5.0, 30.0), (30.0, 30.0))


def loop_field(current, radius, height, r, z):
    """Field (B_r, B_z) [T] of circular current loops, exact (elliptic integrals).

    ``current``, ``radius`` and ``height`` broadcast against the field points ``r``, ``z``.
    """
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


def magnet_loops(nodes):
    """Currents [A], radii and heights [m] of the loops that replace the magnets."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    heights = MAGNET_TOP - 0.5 * MAGNET_HEIGHT * (1.0 - x)
    weights = 0.5 * MAGNET_HEIGHT * w * REMANENCE / MU0
    # centre: north up (+phi current on its side); ring: south up (+phi inside, -phi outside)
    sheets = ((CENTRE_RADIUS, 1.0), (RING_INNER, 1.0), (RING_OUTER, -1.0))
    current = np.concatenate([sign * weights for _, sign in sheets])
    radius = np.repeat([s[0] for s in sheets], nodes)
    height = np.tile(heights, len(sheets))
    return current, radius, height


def magnet_field(loops, r, z):
    """(B_r, B_z) [T] of the magnets at points (r, z) of any shape."""
    r, z = np.asarray(r, dtype=float), np.asarray(z, dtype=float)
    b_r, b_z = loop_field(
        *(x[:, None] for x in loops), r.ravel()[None], z.ravel()[None]
    )
    return b_r.sum(axis=0).reshape(r.shape), b_z.sum(axis=0).reshape(r.shape)


def potential(z):
    """Prescribed potential [V] against the height z [m] above the target."""
    sheath = np.where(z < SHEATH_WIDTH, (1.0 - z / SHEATH_WIDTH) ** 2, 0.0)
    return PRESHEATH_FIELD * z - SHEATH_VOLTAGE * sheath


def axial_field(z):
    """Prescribed E_z [V/m]: the uniform presheath field plus the linear sheath field."""
    sheath = np.where(z < SHEATH_WIDTH, 1.0 - z / SHEATH_WIDTH, 0.0)
    return -PRESHEATH_FIELD - 2.0 * SHEATH_VOLTAGE / SHEATH_WIDTH * sheath


def cartesian_field(loops, x):
    """E and B [V/m, T] at Cartesian points x (3, n)."""
    r = np.hypot(x[0], x[1])
    b_r, b_z = magnet_field(loops, r, x[2])
    c, s = x[0] / r, x[1] / r
    b = np.array([b_r * c, b_r * s, b_z])
    e = np.zeros_like(x)
    e[2] = axial_field(x[2])
    return e, b


def field_line(loops):
    """The field line with its apex at APEX_HEIGHT, as splines of the arc length s.

    Returns the arc length range and splines of r, z, |B|, the potential, and the azimuthal
    angular drift rates per unit (E x B), per unit mu (grad-B) and per unit m v_par^2 (curvature).
    """
    # the apex: where B_z = 0 at that height, between the magnets
    apex_r = brentq(
        lambda r: magnet_field(loops, r, APEX_HEIGHT)[1], CENTRE_RADIUS, RING_INNER
    )

    def direction(_, y, sign):
        b_r, b_z = magnet_field(loops, y[0], y[1])
        return sign * np.array([b_r, b_z]) / np.hypot(b_r, b_z)

    def target(_, y, sign):
        return y[1]

    target.terminal = True
    halves = []
    for sign in (-1.0, 1.0):
        sol = solve_ivp(
            direction,
            (0.0, 0.1),
            [apex_r, APEX_HEIGHT],
            args=(sign,),
            events=target,
            rtol=1e-11,
            atol=1e-14,
            max_step=2e-5,
        )
        halves.append((sign * sol.t, sol.y))
    s = np.concatenate([halves[0][0][::-1], halves[1][0][1:]])
    r, z = np.concatenate([halves[0][1][:, ::-1], halves[1][1][:, 1:]], axis=1)
    if r[0] > r[-1]:
        s, r, z = -s[::-1], r[::-1], z[::-1]

    # B, grad B and the curvature by centred differences of the exact field
    h = 1e-6
    b_r, b_z = magnet_field(loops, r, z)
    b = np.hypot(b_r, b_z)

    def unit(dr, dz):
        f_r, f_z = magnet_field(loops, r + dr, z + dz)
        n = np.hypot(f_r, f_z)
        return f_r / n, f_z / n, n

    rp, rm = unit(h, 0.0), unit(-h, 0.0)
    zp, zm = unit(0.0, h), unit(0.0, -h)
    grad_r, grad_z = (rp[2] - rm[2]) / (2 * h), (zp[2] - zm[2]) / (2 * h)
    br_hat, bz_hat = b_r / b, b_z / b
    kappa_r = br_hat * (rp[0] - rm[0]) / (2 * h) + bz_hat * (zp[0] - zm[0]) / (2 * h)
    kappa_z = br_hat * (rp[1] - rm[1]) / (2 * h) + bz_hat * (zp[1] - zm[1]) / (2 * h)
    q = -ELEMENTARY_CHARGE
    # azimuthal components: (A x B)_phi = A_z B_r - A_r B_z
    exb = axial_field(z) * b_r / b**2 / r
    grad_b = (b_z * grad_r - b_r * grad_z) / (q * b**2) / r
    curvature = (b_z * kappa_r - b_r * kappa_z) / (q * b**2) / r
    spline = {
        name: CubicSpline(s, values)
        for name, values in (
            ("r", r),
            ("z", z),
            ("b", b),
            ("phi", potential(z)),
            ("exb", exb),
            ("grad_b", grad_b),
            ("curvature", curvature),
        )
    }
    return {"s": s, "r": r, "z": z, "b": b, "spline": spline}


def well_bottom(line, mu):
    """Arc length of the minimum of U = mu B - e phi along the line (above the sheath)."""
    sp = line["spline"]
    s = line["s"]
    inside = s[line["z"] > 1.5 * SHEATH_WIDTH]

    def energy(x):
        return mu * sp["b"](x) - ELEMENTARY_CHARGE * sp["phi"](x)

    return minimize_scalar(
        energy,
        bounds=(inside[0], inside[-1]),
        method="bounded",
        options={"xatol": 1e-10},
    ).x


def launch_point(line, energy_ev, pitch_deg):
    """Arc length, mu and v_par of an electron at the bottom of its well."""
    w = energy_ev * ELEMENTARY_CHARGE
    sin2 = np.sin(np.radians(pitch_deg)) ** 2
    s0 = 0.0
    for _ in range(
        30
    ):  # mu depends on B where the electron starts, which depends on mu
        mu = w * sin2 / line["spline"]["b"](s0)
        s0 = well_bottom(line, mu)
    mu = w * sin2 / line["spline"]["b"](s0)
    v_par = np.sqrt(2.0 * w * (1.0 - sin2) / ELECTRON_MASS)
    return float(s0), float(mu), float(v_par)


def guiding_centre(line, energy_ev, pitch_deg):
    """Bounce period and bounce-averaged drift rates of the guiding-centre theory.

    The parallel motion in U(s) = mu B - e phi is integrated over two bounces, together with the
    azimuthal angle of the three drifts. Also returns the harmonic period at the bottom of the
    well, and the lowest height the bounce reaches.
    """
    sp = line["spline"]
    s0, mu, v0 = launch_point(line, energy_ev, pitch_deg)
    m, e = ELECTRON_MASS, ELEMENTARY_CHARGE

    def rhs(_, y):
        s, v = y[0], y[1]
        force = -mu * sp["b"](s, 1) + e * sp["phi"](s, 1)
        return [
            v,
            force / m,
            sp["exb"](s),
            mu * sp["grad_b"](s),
            m * v * v * sp["curvature"](s),
            0.0,
        ]

    def turning(_, y):
        return y[1]

    turning.direction = 1.0
    turning.terminal = 2

    def outside(_, y):
        return min(y[0] - line["s"][0], line["s"][-1] - y[0])

    outside.terminal = True
    sol = solve_ivp(
        rhs,
        (0.0, 2e-6),
        [s0, v0, 0.0, 0.0, 0.0, 0.0],
        events=(turning, outside),
        rtol=1e-10,
        atol=[1e-14, 1e-6, 1e-12, 1e-12, 1e-12, 1.0],
        max_step=1e-10,
        dense_output=True,
    )
    times = sol.t_events[0]
    if len(times) < 2:  # the electron leaves the line (reaches the target): no bounce
        nan = float("nan")
        keys = ("tau", "omega_exb", "omega_grad_b", "omega_curvature", "omega")
        return dict.fromkeys(keys, nan) | {
            "tau_harmonic": nan,
            "lowest": 0.0,
            "s0": s0,
            "mu": mu,
            "v_par": v0,
        }
    t1, t2 = times[0], times[1]
    y1, y2 = sol.sol(t1), sol.sol(t2)
    tau = t2 - t1
    drift = (y2[2:5] - y1[2:5]) / tau
    grid = sol.sol(np.linspace(t1, t2, 400))
    curvature_u = mu * sp["b"](s0, 2) - e * sp["phi"](s0, 2)
    return {
        "tau": tau,
        "omega_exb": drift[0],
        "omega_grad_b": drift[1],
        "omega_curvature": drift[2],
        "omega": drift.sum(),
        "tau_harmonic": 2.0 * np.pi * np.sqrt(m / curvature_u),
        "lowest": float(sp["z"](grid[0]).min()),
        "s0": s0,
        "mu": mu,
        "v_par": v0,
    }


def initial_state(loops, line, energy_ev, pitch_deg, gyrophase=0.0):
    """Position and velocity (3,) of an electron whose guiding centre starts in the well."""
    sp = line["spline"]
    s0, mu, v_par = launch_point(line, energy_ev, pitch_deg)
    centre = np.array([sp["r"](s0), 0.0, sp["z"](s0)])
    e, b = cartesian_field(loops, centre[:, None])
    e, b = e[:, 0], b[:, 0]
    b2 = b @ b
    b_hat = b / np.sqrt(b2)
    v_e = np.cross(e, b) / b2
    v_perp = np.sqrt(2.0 * mu * np.sqrt(b2) / ELECTRON_MASS)
    e1 = np.array([0.0, 1.0, 0.0])  # the azimuthal direction, perpendicular to b here
    e2 = np.cross(b_hat, e1)
    gyration = v_perp * (np.cos(gyrophase) * e1 + np.sin(gyrophase) * e2)
    q = -ELEMENTARY_CHARGE
    position = centre + ELECTRON_MASS / (q * b2) * np.cross(b, gyration)
    return position, v_par * b_hat + v_e + gyration


def push(loops, line, launches, duration, keep_orbits):
    """Boris-push the electrons; record their guiding centres (and some full orbits)."""
    from mpiparticles import ParticleStore, boris_push

    states = [initial_state(loops, line, w, a) for w, a in launches]
    store = ParticleStore(ndim=3, nvel=3)
    store.add(np.array([p for p, _ in states]).T, np.array([v for _, v in states]).T)
    steps = int(round(duration / DT))
    n = store.length
    q = -ELEMENTARY_CHARGE
    record_r, record_phi, record_z, orbit = [], [], [], []
    for step in range(steps + 1):
        p, v = store.positions[:, :n], store.velocities[:, :n]
        e, b = cartesian_field(loops, p)
        if step % RECORD == 0:
            # guiding centre: x - m (B x v_perp) / (q B^2); v_perp includes the drift (small)
            b2 = (b * b).sum(axis=0)
            centre = p - ELECTRON_MASS / (q * b2) * np.cross(b, v, axis=0)
            record_r.append(np.hypot(centre[0], centre[1]))
            record_phi.append(np.arctan2(centre[1], centre[0]))
            record_z.append(centre[2])
            orbit.append(p[:, -keep_orbits:].copy())
        if step < steps:
            boris_push(store, e, b, DT, q / ELECTRON_MASS)
    return {
        "t": DT * RECORD * np.arange(len(record_r)),
        "r": np.array(record_r).T,
        "phi": np.unwrap(np.array(record_phi).T, axis=1),
        "z": np.array(record_z).T,
        "orbit": np.array(orbit).transpose(2, 1, 0).astype(np.float32),
    }


def measure(t, r, phi):
    """Bounce period and mean angular drift from a guiding-centre record.

    Bounces are the upward crossings of the mean radius (with a hysteresis of a quarter of the
    amplitude); the drift is the angle advanced between the first and the last crossing.
    """
    centre = r - r.mean()
    level = 0.25 * np.abs(centre).max()
    crossings, armed = [], False
    for i in range(1, len(t)):
        if centre[i] < -level:
            armed = True
        if armed and centre[i - 1] < 0.0 <= centre[i]:
            frac = -centre[i - 1] / (centre[i] - centre[i - 1])
            crossings.append(t[i - 1] + frac * (t[i] - t[i - 1]))
            armed = False
    if len(crossings) < 3:
        return np.nan, np.nan, len(crossings)
    first, last = crossings[0], crossings[-1]
    tau = (last - first) / (len(crossings) - 1)
    omega = (np.interp(last, t, phi) - np.interp(first, t, phi)) / (last - first)
    return tau, omega, len(crossings)


def simulate(preset: str = "full") -> dict:
    """Push the scan electrons and the orbit electrons; evaluate the guiding-centre theory."""
    size = PRESETS[preset]
    loops = magnet_loops(size["nodes"])
    line = field_line(loops)
    scans = {
        "energy": [(w, SCAN_PITCH) for w in size["energies"]],
        "pitch": [(SCAN_ENERGY, a) for a in size["pitches"]],
    }
    orbits = list(ORBITS[: size["orbits"]])
    launches = scans["energy"] + scans["pitch"] + orbits
    start = time.time()
    rec = push(loops, line, launches, size["duration"], len(orbits))
    wall = time.time() - start

    result = {"preset": preset, "push_seconds": wall, "steps": len(rec["t"]) * RECORD}
    offset = 0
    for name, launch in scans.items():
        rows = []
        for i, (w, a) in enumerate(launch):
            j = offset + i
            tau, omega, bounces = measure(rec["t"], rec["r"][j], rec["phi"][j])
            theory = guiding_centre(line, w, a)
            rows.append(
                {
                    "energy": w,
                    "pitch": a,
                    "tau": tau,
                    "omega": omega,
                    "bounces": bounces,
                }
                | {f"gc_{k}": v for k, v in theory.items()}
            )
        result[name] = {k: np.array([row[k] for row in rows]) for k in rows[0]}
        offset += len(launch)

    # the orbits: the full path over the first bounce or two, the guiding centre over the run
    every = max(1, int(round(0.5e-9 / (DT * RECORD))))

    def one_turn(k):
        """Indices of the guiding-centre record over the first turn around the axis."""
        phi = rec["phi"][offset + k]
        return slice(
            0, int(np.argmax(np.abs(phi - phi[0]) >= 2 * np.pi)) or None, every
        )

    show = int(min(len(rec["t"]), 120e-9 / (DT * RECORD)))
    result["orbits"] = [
        {
            "energy": w,
            "pitch": a,
            "path": rec["orbit"][k, :, :show],
            "centre_x": (rec["r"][offset + k] * np.cos(rec["phi"][offset + k]))[
                one_turn(k)
            ],
            "centre_y": (rec["r"][offset + k] * np.sin(rec["phi"][offset + k]))[
                one_turn(k)
            ],
            "lowest": float(rec["orbit"][k, 2].min()),
        }
        for k, (w, a) in enumerate(orbits)
    ]
    result["line"] = {"r": line["r"][::20], "z": line["z"][::20]}
    result["duration"] = size["duration"]
    return result


def plot(result: dict) -> None:
    """Draw the figures and print the agreement."""
    mm, ns = 1e3, 1e9
    line = result["line"]

    # 1. orbits in (r, z)
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.plot(
        mm * line["r"],
        mm * line["z"],
        "k--",
        lw=1,
        label=f"field line, apex {mm * APEX_HEIGHT:g} mm",
    )
    ax.axhspan(0.0, mm * SHEATH_WIDTH, color="0.85", label="cathode sheath")
    for orbit in result["orbits"]:
        x, y, z = orbit["path"]
        ax.plot(
            mm * np.hypot(x, y),
            mm * z,
            lw=0.6,
            label=f"{orbit['energy']:g} eV, {orbit['pitch']:g}°",
        )
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("height above the target z [mm]")
    ax.set_ylim(0.0, None)
    ax.legend(fontsize=8)

    # 2. guiding centres seen from above
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    angle = np.linspace(0.0, 2 * np.pi, 200)
    for radius, style in ((CENTRE_RADIUS, ":"), (RING_INNER, ":"), (RING_OUTER, ":")):
        ax.plot(
            mm * radius * np.cos(angle), mm * radius * np.sin(angle), style, color="0.6"
        )
    for orbit in result["orbits"]:
        ax.plot(
            mm * orbit["centre_x"],
            mm * orbit["centre_y"],
            lw=0.6,
            label=f"{orbit['energy']:g} eV, {orbit['pitch']:g}°",
        )
        ax.plot(mm * orbit["centre_x"][0], mm * orbit["centre_y"][0], "ko", ms=4)
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.legend(fontsize=8, loc="upper right")

    # 3. azimuthal drift against the energy
    scan = result["energy"]
    fig, ax = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
    w = scan["energy"]
    unit = 1e-6  # rad/s -> rad/us
    ax.plot(w, -unit * scan["gc_omega_exb"], ":", label="E×B")
    ax.plot(
        w,
        -unit * (scan["gc_omega_exb"] + scan["gc_omega_grad_b"]),
        "-.",
        label="E×B + ∇B",
    )
    ax.plot(w, -unit * scan["gc_omega"], "-", label="E×B + ∇B + curvature")
    ax.plot(w, -unit * scan["omega"], "o", label="Boris push")
    ax.set_xscale("log")
    ax.set_xlabel(f"kinetic energy W [eV] (pitch angle {SCAN_PITCH:g}°)")
    ax.set_ylabel("azimuthal drift −dφ/dt [rad/µs]")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)

    # 4. bounce period against the pitch angle
    scan = result["pitch"]
    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=(7, 6.4), constrained_layout=True
    )
    a = scan["pitch"]
    axes[0].plot(a, ns * scan["gc_tau"], "-", label="guiding centre, ∮ ds / v∥")
    axes[0].plot(a, ns * scan["gc_tau_harmonic"], ":", label="harmonic, 2π (m / U'')^½")
    axes[0].plot(a, ns * scan["tau"], "o", label="Boris push")
    axes[0].set_ylabel("bounce period [ns]")
    axes[0].legend(fontsize=8)
    axes[1].plot(a, -unit * scan["gc_omega_exb"], ":", label="E×B")
    axes[1].plot(a, -unit * scan["gc_omega"], "-", label="E×B + ∇B + curvature")
    axes[1].plot(a, -unit * scan["omega"], "o", label="Boris push")
    axes[1].set_ylabel("−dφ/dt [rad/µs]")
    axes[1].set_xlabel(
        f"pitch angle at the bottom of the well [deg] ({SCAN_ENERGY:g} eV)"
    )
    axes[1].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.grid(True, alpha=0.3)

    for name in ("energy", "pitch"):
        scan = result[name]
        drift = np.abs(scan["omega"] / scan["gc_omega"] - 1.0)
        bounce = np.abs(scan["tau"] / scan["gc_tau"] - 1.0)
        print(
            f"{name} scan: drift within {100 * np.nanmax(drift):.1f} % of the guiding-centre "
            f"value, bounce period within {100 * np.nanmax(bounce):.1f} %"
        )


def table(result: dict) -> str:
    """Measured against guiding-centre values, for every electron of the scans."""
    lines = [
        "| W [eV] | pitch [°] | bounces | τ_b push [ns] | τ_b theory [ns] "
        "| dφ/dt push [rad/µs] | dφ/dt theory [rad/µs] | E×B share | lowest z, theory [mm] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ("energy", "pitch"):
        scan = result[name]
        for i in range(len(scan["energy"])):
            lines.append(
                f"| {scan['energy'][i]:g} | {scan['pitch'][i]:g} | {scan['bounces'][i] - 1} "
                f"| {1e9 * scan['tau'][i]:.2f} | {1e9 * scan['gc_tau'][i]:.2f} "
                f"| {-1e-6 * scan['omega'][i]:.3f} | {-1e-6 * scan['gc_omega'][i]:.3f} "
                f"| {scan['gc_omega_exb'][i] / scan['gc_omega'][i]:.2f} "
                f"| {1e3 * scan['gc_lowest'][i]:.2f} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    plot(simulate(args.preset))
    plt.show()
