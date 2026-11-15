"""Cathode sheath.

Nearly all of the discharge voltage of a magnetron drops across a thin sheath in front of the
target. This study estimates that sheath from textbook formulas for a DC magnetron: 400 V
across the sheath, an ion current density of 100 A/m$^2$ (10 to 1000 A/m$^2$ for the range),
argon at 0.1 to 100 Pa.

**Thickness.** The collisionless Child–Langmuir law (byob's `sheath_width`), or its collisional
form where the ion mean free path falls below about 0.3 of the thickness; the study takes the
thinner of the two. The mean free paths use Phelps' Ar$^+$–Ar cross sections from plasmacoll.

**Ion energy at the target.** An ion that charge-exchanges in the sheath starts again at rest
and gains only the rest of the potential; a fraction $e^{-s/\\lambda_{cx}}$ crosses without
exchange. At 0.5 to 2 Pa the sheath is much thinner than the charge exchange mean free path,
so nearly all ions arrive with the full sheath voltage.

**Secondary electrons.** Thornton's condition for a self-sustained discharge is
$V_\\mathrm{min} = W_0/(\\gamma\\,\\varepsilon_i\\,\\varepsilon_e)$, with $W_0$ the energy per
ion pair, $\\gamma$ the secondary electron yield, and $\\varepsilon_i$, $\\varepsilon_e$ the
fractions of the ions and of the electron energy that are used. With $\\gamma \\approx 0.1$ and
$W_0 = 30$ eV it gives 300 V at best, the low end of DC magnetron voltages.

Compared: byob's Child–Langmuir width with the closed-form law, and Thornton's minimum voltage
with the 300–600 V of DC magnetrons.

Reference: M. A. Lieberman and A. J. Lichtenberg, *Principles of Plasma Discharges and
Materials Processing*, 2nd ed. (Wiley, 2005), ch. 6 (sheaths); J. A. Thornton, J. Vac. Sci.
Technol. 15, 171 (1978) (minimum voltage of a magnetron); A. V. Phelps, J. Appl. Phys. 76, 747
(1994) (Ar$^+$–Ar cross sections).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from byob.sheath import debye_length, sheath_width
from plasmacoll.argon import argon_ion_cross_sections
from plasmacoll.constants import ATOMIC_MASS, BOLTZMANN, ELEMENTARY_CHARGE

PRESETS = {
    "smoke": {"points": 12, "energies": 50},
    "full": {"points": 120, "energies": 400},
}

ION_MASS = 39.948 * ATOMIC_MASS
EPSILON_0 = 8.8541878128e-12
GAS_TEMPERATURE = 300.0  # K
ELECTRON_TEMPERATURE = 3.0  # eV, at the sheath edge
#: Centre-of-mass energy [eV] at which the ion cross sections are taken.
CROSS_SECTION_ENERGY = 50.0
VOLTAGE = 400.0  # V
CURRENT_DENSITIES = (10.0, 100.0, 1000.0)  # A/m^2
BASE_CURRENT = 100.0
PRESSURES = (0.5, 2.0, 5.0, 20.0)  # Pa, for the energy distributions


def cross_sections():
    """Phelps' Ar+ + Ar cross sections [m^2] at ``CROSS_SECTION_ENERGY``: total, charge exchange."""
    tables = argon_ion_cross_sections()
    isotropic = float(tables["isotropic"](CROSS_SECTION_ENERGY))
    backscatter = float(tables["backscatter"](CROSS_SECTION_ENERGY))
    return isotropic + backscatter, backscatter


def gas_density(pressure):
    """Argon density [m^-3] at ``pressure`` [Pa] and ``GAS_TEMPERATURE``."""
    return np.asarray(pressure) / (BOLTZMANN * GAS_TEMPERATURE)


def child_langmuir_width(voltage, current_density):
    """Collisionless sheath thickness [m] from byob, with the Bohm flux ``J = e n_s u_B``."""
    bohm_speed = np.sqrt(ELEMENTARY_CHARGE * ELECTRON_TEMPERATURE / ION_MASS)
    edge_density = np.asarray(current_density) / (ELEMENTARY_CHARGE * bohm_speed)
    return sheath_width(voltage, edge_density, ELECTRON_TEMPERATURE)


def child_langmuir_closed_form(voltage, current_density):
    """``s = (4/9 eps0 sqrt(2e/M) V^1.5 / J)^(1/2)``, for the check of byob."""
    factor = 4.0 / 9.0 * EPSILON_0 * np.sqrt(2.0 * ELEMENTARY_CHARGE / ION_MASS)
    return np.sqrt(factor * np.asarray(voltage) ** 1.5 / np.asarray(current_density))


def collisional_width(voltage, current_density, mean_free_path):
    """Thickness [m] of the collisional (variable-mobility) Child law of Lieberman and Lichtenberg."""
    factor = (
        2.0
        / 3.0
        * (5.0 / 3.0) ** 1.5
        * EPSILON_0
        * np.sqrt(
            2.0 * ELEMENTARY_CHARGE * np.asarray(mean_free_path) / (np.pi * ION_MASS)
        )
    )
    return (factor * np.asarray(voltage) ** 1.5 / np.asarray(current_density)) ** 0.4


def sheath(voltage, current_density, pressure):
    """Thickness [m], potential exponent and mean free paths [m] at ``pressure`` [Pa]."""
    sigma_total, sigma_cx = cross_sections()
    density = gas_density(pressure)
    mfp, mfp_cx = 1.0 / (density * sigma_total), 1.0 / (density * sigma_cx)
    collisionless = child_langmuir_width(voltage, current_density)
    collisional = collisional_width(voltage, current_density, mfp)
    thin = collisional < collisionless
    return {
        "width": np.where(thin, collisional, collisionless),
        "exponent": np.where(thin, 5.0 / 3.0, 4.0 / 3.0),
        "collisionless": collisionless * np.ones_like(mfp),
        "collisional": collisional,
        "mean_free_path": mfp,
        "cx_mean_free_path": mfp_cx,
    }


def energy_distribution(width, exponent, cx_mean_free_path, energies):
    """Ion energy distribution at the target, per unit ``E / eV``, and the unscattered fraction.

    The last charge exchange at ``z`` (from the sheath edge) has the density
    ``exp(-(s - z) / lambda) / lambda``; the ion then arrives with ``1 - (z / s)^p`` of ``eV``.
    """
    ratio = width / cx_mean_free_path
    z = width * (1.0 - energies) ** (1.0 / exponent)
    dz = width / exponent * (1.0 - energies) ** (1.0 / exponent - 1.0)
    density = np.exp(-(width - z) / cx_mean_free_path) / cx_mean_free_path * dz
    return density, np.exp(-ratio)


def mean_energy(width, exponent, cx_mean_free_path, points=2001):
    """Mean ion energy at the target as a fraction of ``eV``, the unscattered ions included."""
    z = np.linspace(0.0, width, points)
    weight = np.exp(-(width - z) / cx_mean_free_path) / cx_mean_free_path
    return np.trapezoid(weight * (1.0 - (z / width) ** exponent), z) + np.exp(
        -width / cx_mean_free_path
    )


def simulate(preset: str = "full") -> dict:
    """Evaluate the sheath estimates of the study."""
    size = PRESETS[preset]
    voltages = np.linspace(100.0, 1000.0, size["points"])
    pressures = np.geomspace(0.1, 100.0, size["points"])
    sigma_total, sigma_cx = cross_sections()

    widths = {
        j: {
            "collisionless": child_langmuir_width(voltages, j),
            "closed_form": child_langmuir_closed_form(voltages, j),
            "at_100pa": sheath(voltages, j, 100.0)["width"],
        }
        for j in CURRENT_DENSITIES
    }
    against_pressure = {j: sheath(VOLTAGE, j, pressures) for j in CURRENT_DENSITIES}
    for data in against_pressure.values():
        data["mean_energy"] = np.array(
            [
                mean_energy(s, p, lam)
                for s, p, lam in zip(
                    data["width"],
                    data["exponent"],
                    data["cx_mean_free_path"],
                    strict=True,
                )
            ]
        )
        data["unscattered"] = np.exp(-data["width"] / data["cx_mean_free_path"])

    energies = np.linspace(0.0, 1.0, size["energies"], endpoint=False)
    distributions = {}
    for p in PRESSURES:
        state = sheath(VOLTAGE, BASE_CURRENT, p)
        density, unscattered = energy_distribution(
            float(state["width"]),
            float(state["exponent"]),
            float(state["cx_mean_free_path"]),
            energies,
        )
        distributions[p] = {
            "density": density,
            "unscattered": unscattered,
            "width": float(state["width"]),
            "cx_mean_free_path": float(state["cx_mean_free_path"]),
            "mean_energy": mean_energy(
                float(state["width"]),
                float(state["exponent"]),
                float(state["cx_mean_free_path"]),
            ),
        }

    gammas = np.geomspace(0.03, 0.3, size["points"])
    #: the energy per ion pair over the efficiencies, W_0 / (eps_i eps_e) [eV]
    thornton = {cost: cost / gammas for cost in (30.0, 60.0, 120.0, 240.0)}

    return {
        "voltages": voltages,
        "pressures": pressures,
        "widths": widths,
        "against_pressure": against_pressure,
        "energies": energies,
        "distributions": distributions,
        "gammas": gammas,
        "thornton": thornton,
        "sigma_total": sigma_total,
        "sigma_cx": sigma_cx,
        "edge_debye_length": float(
            debye_length(
                BASE_CURRENT
                / (
                    ELEMENTARY_CHARGE
                    * np.sqrt(ELEMENTARY_CHARGE * ELECTRON_TEMPERATURE / ION_MASS)
                ),
                ELECTRON_TEMPERATURE,
            )
        ),
    }


def plot(result: dict) -> None:
    """Draw the figures of the study."""
    v = result["voltages"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for i, (j, data) in enumerate(result["widths"].items()):
        ax.plot(
            v,
            1e3 * data["collisionless"],
            color=f"C{i}",
            lw=2,
            label=f"J = {j:g} A/m², collisionless",
        )
        ax.plot(
            v,
            1e3 * data["at_100pa"],
            "--",
            color=f"C{i}",
            label=f"J = {j:g} A/m², at 100 Pa",
        )
    ax.set_xlabel("sheath voltage V [V]")
    ax.set_ylabel("sheath thickness s [mm]")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    p = result["pressures"]
    base = result["against_pressure"][BASE_CURRENT]
    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.plot(
        p,
        1e3 * base["collisionless"],
        "--",
        color="C0",
        label="Child–Langmuir (collisionless)",
    )
    ax.plot(
        p, 1e3 * base["collisional"], "--", color="C1", label="collisional Child law"
    )
    ax.plot(
        p,
        1e3 * base["width"],
        color="k",
        lw=2.5,
        label="sheath thickness used (the thinner)",
    )
    ax.plot(
        p,
        1e3 * base["mean_free_path"],
        ":",
        color="C2",
        lw=2,
        label="ion mean free path λ_i",
    )
    ax.plot(
        p,
        1e3 * base["cx_mean_free_path"],
        "-.",
        color="C3",
        label="charge exchange mean free path",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("argon pressure [Pa]")
    ax.set_ylabel("length [mm]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    energies = result["energies"] * VOLTAGE
    for i, (pressure, data) in enumerate(result["distributions"].items()):
        ax.plot(
            energies,
            data["density"] / VOLTAGE,
            color=f"C{i}",
            lw=2,
            label=f"{pressure:g} Pa: {100 * data['unscattered']:.0f} % arrive with {VOLTAGE:g} eV",
        )
    ax.set_yscale("log")
    ax.set_xlabel("ion energy at the target [eV]")
    ax.set_ylabel("ions per eV (fraction)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=(6.5, 6.5), constrained_layout=True
    )
    for i, (j, data) in enumerate(result["against_pressure"].items()):
        axes[0].plot(
            p,
            data["mean_energy"] * VOLTAGE,
            color=f"C{i}",
            lw=2,
            label=f"J = {j:g} A/m²",
        )
        axes[1].plot(
            p, data["unscattered"], color=f"C{i}", lw=2, label=f"J = {j:g} A/m²"
        )
    axes[0].set_ylabel("mean ion energy at the target [eV]")
    axes[0].legend()
    axes[1].set_ylabel("fraction without charge exchange")
    axes[1].set_xlabel("argon pressure [Pa]")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.grid(True, alpha=0.3, which="both")

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    gammas = result["gammas"]
    for i, (cost, vmin) in enumerate(result["thornton"].items()):
        ax.plot(
            gammas,
            vmin,
            color=f"C{i}",
            lw=2,
            label=f"$W_0/(\\varepsilon_i\\varepsilon_e)$ = {cost:g} eV",
        )
    for voltage, label in ((300.0, "DC magnetrons, 300–600 V"), (600.0, None)):
        ax.plot(
            gammas,
            np.full_like(gammas, voltage),
            color="0.5",
            lw=4,
            alpha=0.5,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("secondary electron yield γ")
    ax.set_ylabel("minimum voltage [V]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")


def table(result: dict) -> str:
    """The base case against pressure, and the check of byob's Child–Langmuir width."""
    check = max(
        np.abs(data["collisionless"] / data["closed_form"] - 1.0).max()
        for data in result["widths"].values()
    )
    rows = [
        f"V = {VOLTAGE:g} V, ion current density {BASE_CURRENT:g} A/m², T_e = "
        f"{ELECTRON_TEMPERATURE:g} eV; Phelps' cross sections at {CROSS_SECTION_ENERGY:g} eV: "
        f"total {result['sigma_total']:.2e} m², charge exchange {result['sigma_cx']:.2e} m²; "
        f"Debye length at the sheath edge {1e6 * result['edge_debye_length']:.0f} µm; byob's "
        f"width against the closed form: largest relative difference {check:.1e}.",
        "",
        "| pressure [Pa] | λ_cx [mm] | s [mm] | s / λ_cx | without charge exchange | mean energy [eV] |",
        "|---|---|---|---|---|---|",
    ]
    for pressure, data in result["distributions"].items():
        rows.append(
            f"| {pressure:g} | {1e3 * data['cx_mean_free_path']:.2f} | {1e3 * data['width']:.2f} "
            f"| {data['width'] / data['cx_mean_free_path']:.2f} | {100 * data['unscattered']:.0f} % "
            f"| {VOLTAGE * data['mean_energy']:.0f} |"
        )
    rows += [
        "",
        "| γ | V_min, W₀ = 30 eV, ε_i ε_e = 1 | 0.5 | 0.25 |",
        "|---|---|---|---|",
    ]
    for gamma in (0.05, 0.1, 0.15):
        rows.append(
            f"| {gamma:g} | "
            + " | ".join(f"{30.0 / (gamma * eff):.0f} V" for eff in (1.0, 0.5, 0.25))
            + " |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    result = simulate(args.preset)
    print(table(result))
    plot(result)
    plt.show()
