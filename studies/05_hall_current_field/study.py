"""Hall current field.

In a magnetron the electrons drift around the racetrack on E×B and the ions do not, so the
plasma carries an azimuthal Hall current, of the order of the Hall parameter times the
discharge current (Hecimovic and von Keudell 2018). Can the models keep a fixed $B(x)$? Yes if
the field of this current is a few per cent of the magnet's; if it is a quarter, the field
needs a feedback.

Two independent estimates. The first sums the Biot–Savart field of the Hall current
$I_H = \\mathrm{Ha}\\,I_D$ flowing in a torus over the racetrack (16 mm radius, 5 mm above the
target, 5 mm minor radius), for the review's Hall parameters 2, 5 and 16, at the top and
bottom of the torus. The second is the momentum balance $\\delta B = \\mu_0 n e E L / B$ for a
current density $n e E / B$ in a layer of thickness $L$, which needs the density and the
electric field instead of the current.

Compared: the two estimates with each other and with a typical magnet field of 50 to 100 mT at
the racetrack. Not compared: a measured Hall current or field.

Reference: Hecimovic and von Keudell, J. Phys. D 51 (2018) 453001 (Hall parameter, Hall current
and discharge current).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipe, ellipk

PRESETS = {
    "smoke": {"filaments": 6, "currents": 4, "grid": 5},
    "quick": {"filaments": 16, "currents": 12, "grid": 25},
    "full": {"filaments": 28, "currents": 20, "grid": 60},
}

MU0 = 4.0e-7 * np.pi
ELEMENTARY_CHARGE = 1.602176634e-19
#: Assumed geometry of the current torus [m]: racetrack radius, height, minor radius.
RACETRACK_RADIUS, HEIGHT, MINOR_RADIUS = 0.016, 0.005, 0.005


def ring_radial_field(current, radius, height, r, z):
    """Radial field [T] at (r, z) of a circular loop of ``radius`` at ``height``."""
    dz = z - height
    a2 = (radius + r) ** 2 + dz**2
    m = 4.0 * radius * r / a2
    k, e = ellipk(m), ellipe(m)
    return (
        MU0
        * current
        / (2.0 * np.pi)
        * dz
        / (r * np.sqrt(a2))
        * (-k + (radius**2 + r**2 + dz**2) / ((radius - r) ** 2 + dz**2) * e)
    )


def torus_radial_field(current, r, z, filaments):
    """Radial field [T] at (r, z) of a uniform-current torus, summed from filaments."""
    offsets = np.linspace(-MINOR_RADIUS, MINOR_RADIUS, filaments)
    points = [
        (x, y) for x in offsets for y in offsets if x * x + y * y <= MINOR_RADIUS**2
    ]
    return sum(
        ring_radial_field(current / len(points), RACETRACK_RADIUS + x, HEIGHT + y, r, z)
        for x, y in points
    )


def simulate(preset: str = "quick") -> dict:
    """Field of the Hall current against the discharge current and against n and E."""
    size = PRESETS[preset]
    currents = np.geomspace(0.1, 100.0, size["currents"])
    hall_parameters = (2.0, 5.0, 16.0)
    # evaluated slightly outside the torus so that no filament sits on the field point
    top = HEIGHT + 1.2 * MINOR_RADIUS
    field = np.array(
        [
            [
                torus_radial_field(ha * i_d, RACETRACK_RADIUS, top, size["filaments"])
                * 1e3
                for i_d in currents
            ]
            for ha in hall_parameters
        ]
    )
    densities = np.geomspace(1e17, 1e20, size["grid"])
    electric_fields = np.geomspace(1e2, 1e5, size["grid"])
    magnet = 0.05
    layer = 5e-3
    balance = (
        MU0
        * densities[:, None]
        * ELEMENTARY_CHARGE
        * electric_fields[None, :]
        * layer
        / magnet**2
    )
    return {
        "currents": currents,
        "hall_parameters": np.array(hall_parameters),
        "field_mt": field,
        "densities": densities,
        "electric_fields": electric_fields,
        "balance_fraction": balance,
    }


def plot(result: dict) -> None:
    """Draw the two figures of the study."""
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for ha, series in zip(result["hall_parameters"], result["field_mt"], strict=True):
        ax.plot(
            result["currents"], np.abs(series), marker="o", ms=3, label=f"Ha = {ha:g}"
        )
    ax.axhspan(50, 100, color="0.85", label="magnet, 50 to 100 mT")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("discharge current [A]")
    ax.set_ylabel("field of the Hall current, top of the torus [mT]")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    image = ax.pcolormesh(
        result["electric_fields"],
        result["densities"],
        np.log10(result["balance_fraction"]),
        shading="auto",
    )
    contours = ax.contour(
        result["electric_fields"],
        result["densities"],
        result["balance_fraction"],
        levels=[0.01, 0.1, 0.5],
        colors="white",
    )
    ax.clabel(contours, fmt=lambda v: f"{100 * v:g}%")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("electric field E [V/m]")
    ax.set_ylabel("plasma density n [m$^{-3}$]")
    fig.colorbar(image, label="log10 of the field of the Hall current over 50 mT")

    index = {float(ha): i for i, ha in enumerate(result["hall_parameters"])}
    for i_d in (10.0, 70.0):
        j = int(np.argmin(np.abs(result["currents"] - i_d)))
        print(
            f"I_D = {result['currents'][j]:.0f} A, Ha = 5: field of the Hall current "
            f"{abs(result['field_mt'][index[5.0], j]):.1f} mT at the top of the torus"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    args = parser.parse_args()
    plot(simulate(args.preset))
    plt.show()
