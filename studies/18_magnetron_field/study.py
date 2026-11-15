"""Magnetron magnetic field.

The magnetic field of a circular planar magnetron with a 2-inch (50.8 mm) target. Below the
target sit two permanent magnets, magnetized along the axis in opposite directions: a
cylinder in the centre (north up) and a ring at the edge (south up). A uniformly magnetized
magnet has the field of a current sheet on its side faces, $K_\\varphi = M = B_\\mathrm{rem}/\\mu_0$,
so the centre cylinder is one current sheet and the ring two, with opposite currents on its
inner and outer faces. Each sheet is a stack of circular loops (Gauss–Legendre nodes over
the height of the magnet), and each loop has the exact field of a current loop (Biot–Savart,
complete elliptic integrals $K$ and $E$). The field is axisymmetric, so its field lines are
the contours of the flux function $\\psi = r A_\\varphi$, with $B_r = -\\partial_z \\psi / r$ and
$B_z = \\partial_r \\psi / r$.

The field lines leave the target above the centre magnet and return to it above the ring:
they form an arch. Where $B_z = 0$ on the target the field is parallel to it; the electrons
are trapped there (by the arch and the cathode sheath) and drift around the axis, which is
where the target erodes, the racetrack. Two magnet sets are compared. In the *balanced* one
the two magnets carry the same flux (equal pole areas), and nearly all field lines close on
the target. In the *unbalanced* one (type II) the centre magnet is smaller, the ring carries
2.2 times its flux, the excess flux of the ring leaves towards the substrate, and the axis
has a null of the field above the target, with a separatrix between the closed and the open
field lines.

Assumed (typical of a 2-inch magnetron, not a particular one): NdFeB magnets with
$B_\\mathrm{rem} = 1.2$ T, 8 mm high, their top faces 8 mm below the target surface (target,
backing plate and cooling water); the centre cylinder 12 mm in radius (balanced) or 8 mm
(unbalanced), the ring from 20 to 23.3 mm. No iron yoke, no other magnetic material: the
yoke under real magnets strengthens the field above the target, so the numbers are
indicative only. The field line used for the mirror ratio is the one whose apex is 2 mm
above the target, where the electrons of a discharge are concentrated. Its mirror ratio is the
field at a foot on the target over the minimum along the line; the minimum is not at the apex
but towards the ring, so the arch is a strong mirror on the inside and a weak one on the
outside (without a yoke, the narrow ring gives a weaker field than the centre pole).

Compared: the field-line tracing against the flux function ($\\psi$ is constant along the
traced lines to round-off, printed), and the two magnet sets with each other. Not
compared: a measured field of a real magnetron.

Reference: the planar magnetron, its racetrack and the balanced and unbalanced
configurations as in Lieberman and Lichtenberg, *Principles of Plasma Discharges and
Materials Processing*, 2nd ed. (Wiley, 2005), chapter 14; the field of a circular loop in
Jackson, *Classical Electrodynamics*, 3rd ed. (Wiley, 1999), section 5.5.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk

PRESETS = {
    "smoke": {"grid": (25, 17), "wide": (25, 21), "nodes": 6},
    "full": {"grid": (121, 76), "wide": (81, 97), "nodes": 16},
}

MU0 = 4.0e-7 * np.pi
#: Remanence of the magnets [T] (NdFeB, assumed).
REMANENCE = 1.2
#: Top and bottom faces of the magnets [m]; the target surface is z = 0 (assumed).
MAGNET_TOP, MAGNET_HEIGHT = -8.0e-3, 8.0e-3
#: Radii [m]: centre magnet of the two sets, inner and outer radius of the ring (assumed).
CENTRE_RADIUS = {"balanced": 12.0e-3, "unbalanced": 8.0e-3}
RING_INNER, RING_OUTER = 20.0e-3, 23.3e-3
TARGET_RADIUS = 25.4e-3
#: Apex height of the field line used for the mirror ratio [m] (assumed).
APEX_HEIGHT = 2.0e-3
#: Apex heights [m] of the field lines drawn in the mirror figure.
APEX_HEIGHTS = (1.0e-3, 2.0e-3, 4.0e-3, 6.0e-3)


def loop_field(current, radius, height, r, z):
    """Field (B_r, B_z) [T] and flux function r A_phi [T m^2] of a circular current loop.

    The loop of ``radius`` [m] at ``height`` [m] carries ``current`` [A]; exact expressions with
    the complete elliptic integrals (parameter m = k^2).
    """
    r = np.maximum(np.asarray(r, dtype=float), 1e-12)
    dz = np.asarray(z, dtype=float) - height
    a2 = (radius + r) ** 2 + dz**2
    d2 = (radius - r) ** 2 + dz**2
    m = 4.0 * radius * r / a2
    k, e = ellipk(m), ellipe(m)
    c = MU0 * current / (2.0 * np.pi * np.sqrt(a2))
    b_r = c * dz / r * (-k + (radius**2 + r**2 + dz**2) / d2 * e)
    b_z = c * (k + (radius**2 - r**2 - dz**2) / d2 * e)
    psi = MU0 * current / (2.0 * np.pi) * np.sqrt(a2) * ((1.0 - 0.5 * m) * k - e)
    return b_r, b_z, psi


def magnet_field(config, r, z, nodes):
    """Field (B_r, B_z) [T] and flux function [T m^2] of the magnet set ``config``.

    Each side face of a magnet is a current sheet K = B_rem / mu0, integrated over its height
    with ``nodes`` Gauss-Legendre loops.
    """
    x, w = np.polynomial.legendre.leggauss(nodes)
    heights = MAGNET_TOP - 0.5 * MAGNET_HEIGHT * (1.0 - x)
    weights = 0.5 * MAGNET_HEIGHT * w * REMANENCE / MU0
    # centre: north up (current +phi on its side); ring: south up (+phi inside, -phi outside)
    sheets = ((CENTRE_RADIUS[config], 1.0), (RING_INNER, 1.0), (RING_OUTER, -1.0))
    b_r = b_z = psi = 0.0
    for radius, sign in sheets:
        for height, weight in zip(heights, weights, strict=True):
            f = loop_field(sign * weight, radius, height, r, z)
            b_r, b_z, psi = b_r + f[0], b_z + f[1], psi + f[2]
    return b_r, b_z, psi


def racetrack_radius(config, nodes, height=0.0):
    """Radius [m] where B_z = 0 at ``height`` above the target, between the magnets."""
    r = np.linspace(0.5e-3, 40e-3, 400)
    b_z = magnet_field(config, r, height, nodes)[1]
    i = int(np.argmax(b_z < 0.0))
    if i == 0:
        return float("nan")
    return brentq(
        lambda x: magnet_field(config, x, height, nodes)[1], r[i - 1], r[i], xtol=1e-10
    )


def trace(config, start, nodes):
    """Field line through ``start`` (r, z), both ways down to the target (z = 0).

    Returns the arc length s (0 at ``start``), r, z and |B| along the line, from the inner foot
    to the outer one, or None for a branch that leaves the region without coming back.
    """

    def rhs(_, y):
        b_r, b_z, _ = magnet_field(config, y[0], y[1], nodes)
        b = np.hypot(b_r, b_z)
        return [b_r / b, b_z / b]

    def target(_, y):
        return y[1]

    target.terminal = True
    branches = []
    for direction in (-1.0, 1.0):
        sol = solve_ivp(
            lambda s, y, d=direction: d * np.asarray(rhs(s, y)),
            (0.0, 0.2),
            start,
            events=target,
            rtol=1e-10,
            atol=1e-13,
            max_step=2e-4,
        )
        if sol.status != 1:
            return None
        branches.append((direction * sol.t, sol.y))
    s = np.concatenate([branches[0][0][::-1], branches[1][0][1:]])
    rz = np.concatenate([branches[0][1][:, ::-1], branches[1][1][:, 1:]], axis=1)
    if rz[0, 0] > rz[0, -1]:
        s, rz = -s[::-1], rz[:, ::-1]
    b_r, b_z, psi = magnet_field(config, rz[0], rz[1], nodes)
    return {"s": s, "r": rz[0], "z": rz[1], "b": np.hypot(b_r, b_z), "psi": psi}


def apex(config, height, nodes):
    """Apex (r, z) of the field line that is parallel to the target at ``height``."""
    return np.array([racetrack_radius(config, nodes, height), height])


def axis_null(config, nodes):
    """Height [m] of the null of B on the axis, or nan if there is none below 0.2 m."""
    z = np.geomspace(1e-3, 0.2, 400)
    b_z = magnet_field(config, 0.0, z, nodes)[1]
    change = np.nonzero(np.diff(np.sign(b_z)))[0]
    if len(change) == 0:
        return float("nan")
    i = change[0]
    return brentq(
        lambda x: magnet_field(config, 0.0, x, nodes)[1], z[i], z[i + 1], xtol=1e-10
    )


def half_field_height(config, nodes):
    """Height [m] where |B| on the apex line (B_z = 0) is half its value on the target."""
    b0 = abs(magnet_field(config, racetrack_radius(config, nodes), 0.0, nodes)[0])

    def excess(z):
        r = racetrack_radius(config, nodes, z)
        return np.hypot(*magnet_field(config, r, z, nodes)[:2]) - 0.5 * b0

    return brentq(excess, 0.0, 15e-3, xtol=1e-9)


def describe(config, size):
    """The numbers of one magnet set: racetrack, mirror, arch and null."""
    nodes = size["nodes"]
    r_rt = racetrack_radius(config, nodes)
    b_r, b_z, psi = magnet_field(config, r_rt, 0.0, nodes)
    lines = [trace(config, apex(config, h, nodes), nodes) for h in APEX_HEIGHTS]
    line = lines[APEX_HEIGHTS.index(APEX_HEIGHT)]
    b_apex = np.interp(0.0, line["s"], line["b"])
    b_min = line["b"].min()
    psi_edge = magnet_field(config, TARGET_RADIUS, 0.0, nodes)[2]
    edge_closed = psi_edge > 0.0
    edge_apex = np.nan
    if edge_closed:
        # the line from the target edge: its apex on the B_z = 0 line, where psi = psi_edge
        def excess(z):
            return (
                magnet_field(config, racetrack_radius(config, nodes, z), z, nodes)[2]
                - psi_edge
            )

        heights = np.linspace(0.0, 40e-3, 81)
        values = np.array([excess(z) for z in heights])
        i = int(np.argmax(values < 0.0))
        if i > 0 and np.isfinite(values[i]):
            edge_apex = brentq(excess, heights[i - 1], heights[i], xtol=1e-9)
    drift = max(np.ptp(x["psi"]) / abs(x["psi"]).max() for x in lines)
    return {
        "racetrack_radius": r_rt,
        "b_racetrack": float(np.hypot(b_r, b_z)),
        "psi_max": float(psi),
        "lines": lines,
        "apex_field": float(b_apex),
        "min_field": float(b_min),
        "min_height": float(line["z"][np.argmin(line["b"])]),
        "mirror_inner": float(line["b"][0] / b_min),
        "mirror_outer": float(line["b"][-1] / b_min),
        "half_field_height": half_field_height(config, nodes),
        "edge_closed": bool(edge_closed),
        "edge_apex_height": float(edge_apex),
        "null": axis_null(config, nodes),
        "psi_drift": float(drift),
    }


def grid_maps(config, r, z, nodes, axial=True):
    """|B| [mT], psi and (if ``axial``) B_z [mT] on an (r, z) grid, float32 arrays (z, r)."""
    rr, zz = np.meshgrid(r, z)
    b_r, b_z, psi = magnet_field(config, rr, zz, nodes)
    maps = {
        "b": (1e3 * np.hypot(b_r, b_z)).astype(np.float32),
        "psi": psi.astype(np.float32),
    }
    if axial:
        maps["bz"] = (1e3 * b_z).astype(np.float32)
    return maps


def simulate(preset: str = "full") -> dict:
    """Field maps, the target-surface profiles and the numbers of both magnet sets."""
    size = PRESETS[preset]
    nodes = size["nodes"]
    near_r = np.linspace(0.0, 32e-3, size["grid"][0])
    near_z = np.linspace(0.0, 20e-3, size["grid"][1])
    wide_r = np.linspace(0.25e-3, 50e-3, size["wide"][0])
    wide_z = np.linspace(0.0, 60e-3, size["wide"][1])
    surface_r = np.linspace(0.0, 32e-3, 321)
    result = {"preset": preset, "near_r": near_r, "near_z": near_z}
    result |= {"wide_r": wide_r, "wide_z": wide_z, "surface_r": surface_r}
    for config in CENTRE_RADIUS:
        b_r, b_z, _ = magnet_field(config, surface_r, 0.0, nodes)
        result[config] = describe(config, size) | {
            "wide": grid_maps(config, wide_r, wide_z, nodes, axial=False),
            "surface_br": 1e3 * b_r,
            "surface_bz": 1e3 * b_z,
        }
    result["balanced"]["near"] = grid_maps("balanced", near_r, near_z, nodes)
    return result


def draw_magnets(ax, config, scale=1e3):
    """Outline the target face (z = 0) and mark the magnet radii on it."""
    ax.axhline(0.0, color="0.3", lw=2)
    for r0, r1, color in (
        (0.0, CENTRE_RADIUS[config], "tab:red"),
        (RING_INNER, RING_OUTER, "tab:blue"),
    ):
        ax.plot([scale * r0, scale * r1], [0.0, 0.0], color=color, lw=6, alpha=0.6)


def plot(result: dict) -> None:
    """Draw the four figures and print the checks."""
    mm = 1e3
    bal, unb = result["balanced"], result["unbalanced"]

    # 1. field lines and |B| above the target, balanced
    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    near = bal["near"]
    r, z = mm * result["near_r"], mm * result["near_z"]
    image = ax.pcolormesh(
        r, z, np.minimum(near["b"], 200.0), shading="auto", cmap="viridis"
    )
    fig.colorbar(image, ax=ax, label="|B| [mT] (clipped at 200)")
    levels = np.sort(bal["psi_max"] * np.r_[-0.3, -0.1, np.linspace(0.1, 0.95, 9)])
    ax.contour(
        r,
        z,
        near["psi"],
        levels=levels,
        colors="white",
        linewidths=1.0,
        linestyles="solid",
    )
    ax.contour(
        r,
        z,
        near["bz"],
        levels=[0.0],
        colors="tab:orange",
        linestyles="--",
        linewidths=1.5,
    )
    draw_magnets(ax, "balanced")
    ax.plot(
        mm * bal["racetrack_radius"],
        0.0,
        "v",
        color="tab:orange",
        ms=10,
        label="racetrack",
    )
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("height above the target z [mm]")
    ax.set_ylim(0.0, z[-1])
    ax.legend(loc="upper right", fontsize=8)

    # 2. B_r and B_z on the target surface
    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=(7, 5.6), constrained_layout=True
    )
    rs = mm * result["surface_r"]
    for config, style in (("balanced", "-"), ("unbalanced", "--")):
        data = result[config]
        axes[0].plot(rs, data["surface_br"], style, label=config)
        axes[1].plot(rs, data["surface_bz"], style, label=config)
        for ax in axes:
            ax.axvline(mm * data["racetrack_radius"], ls=style, color="0.5", lw=0.8)
    for ax, name in zip(axes, ("$B_r$", "$B_z$"), strict=True):
        ax.axhline(0.0, color="0.7", lw=0.8)
        ax.axvline(mm * TARGET_RADIUS, color="k", lw=0.8, ls=":")
        ax.set_ylabel(f"{name} at z = 0 [mT]")
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xlabel("r [mm] (grey: racetrack, B_z = 0; dotted: target edge)")

    # 3. |B| along field lines (mirror)
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for h, line in zip(APEX_HEIGHTS, bal["lines"], strict=True):
        ax.plot(
            mm * line["s"], mm * line["b"], label=f"balanced, apex at {mm * h:g} mm"
        )
    line = unb["lines"][APEX_HEIGHTS.index(APEX_HEIGHT)]
    ax.plot(
        mm * line["s"],
        mm * line["b"],
        "k--",
        label=f"unbalanced, apex at {mm * APEX_HEIGHT:g} mm",
    )
    ax.set_xlabel("arc length from the apex along the field line [mm] (inner foot < 0)")
    ax.set_ylabel("|B| [mT]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. balanced against unbalanced: closed and open field lines, null
    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=(6.5, 8.5), constrained_layout=True
    )
    r, z = mm * result["wide_r"], mm * result["wide_z"]
    levels = np.sort(
        bal["psi_max"] * np.r_[-0.4, -0.2, -0.1, -0.05, -0.02, np.linspace(0.1, 0.9, 6)]
    )
    for ax, config in zip(axes, ("balanced", "unbalanced"), strict=True):
        data = result[config]
        wide = data["wide"]
        image = ax.pcolormesh(
            r,
            z,
            np.log10(np.maximum(wide["b"], 1e-2)),
            shading="auto",
            cmap="magma",
            vmin=-1.0,
            vmax=2.5,
        )
        ax.contour(
            r,
            z,
            wide["psi"],
            levels=levels,
            colors="white",
            linewidths=0.9,
            linestyles="solid",
        )
        ax.contour(r, z, wide["psi"], levels=[0.0], colors="tab:cyan", linewidths=1.8)
        draw_magnets(ax, config)
        if np.isfinite(data["null"]):
            ax.plot(0.0, mm * data["null"], "X", color="tab:cyan", ms=10, label="null")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel("z [mm]")
        ax.set_title(config, fontsize=10)
        fig.colorbar(image, ax=ax, label="log10 |B| [mT]")
    axes[1].set_xlabel("r [mm]")

    print(
        "field-line tracing: psi constant along the traced lines to "
        f"{max(bal['psi_drift'], unb['psi_drift']):.1e} (relative)"
    )


def table(result: dict) -> str:
    """The racetrack, mirror, arch and null of the two magnet sets."""
    mm = 1e3

    def row(name, fmt):
        cells = [fmt(result[c]) for c in ("balanced", "unbalanced")]
        return f"| {name} | {cells[0]} | {cells[1]} |"

    def edge(d):
        if d["edge_closed"]:
            return f"{mm * d['edge_apex_height']:.1f} mm"
        return "open (escapes)"

    def null(d):
        return (
            "none below 200 mm" if np.isnan(d["null"]) else f"{mm * d['null']:.1f} mm"
        )

    return "\n".join(
        [
            "| | balanced | unbalanced |",
            "|---|---|---|",
            f"| centre magnet radius | {mm * CENTRE_RADIUS['balanced']:g} mm | {mm * CENTRE_RADIUS['unbalanced']:g} mm |",
            row(
                "racetrack radius ($B_z = 0$ on the target)",
                lambda d: f"{mm * d['racetrack_radius']:.1f} mm",
            ),
            row(
                "B at the racetrack (parallel to the target)",
                lambda d: f"{mm * d['b_racetrack']:.0f} mT",
            ),
            row(
                f"B at the apex {mm * APEX_HEIGHT:g} mm above the target",
                lambda d: f"{mm * d['apex_field']:.0f} mT",
            ),
            row(
                "minimum of B along that field line",
                lambda d: f"{mm * d['min_field']:.0f} mT",
            ),
            row(
                "mirror ratio of that field line (B at the foot / minimum), inner / outer foot",
                lambda d: f"{d['mirror_inner']:.2f} / {d['mirror_outer']:.2f}",
            ),
            row(
                "height where B above the racetrack is halved",
                lambda d: f"{mm * d['half_field_height']:.1f} mm",
            ),
            row("apex of the field line from the target edge", edge),
            row("null of B on the axis", null),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    plot(simulate(args.preset))
    plt.show()
