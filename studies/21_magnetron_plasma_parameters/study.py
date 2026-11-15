"""Magnetron plasma parameters.

The length, time and transport scales of DC and HiPIMS magnetrons in argon, over densities of
$10^{15}$ to $10^{19}$ m$^{-3}$, fields of 10 to 100 mT, pressures of 0.1 to 5 Pa and electron
temperatures of 1 to 10 eV. They are textbook formulas, evaluated with `plasma_plots.theory`
and the argon data of plasmacoll: the Debye length, plasma and gyrofrequencies, thermal
gyroradii, collision frequencies and mean free paths in the gas, the Coulomb collision
frequency, Hall parameters, and the classical and Bohm cross-field diffusion coefficients. The
table gives them for a typical DC and a typical HiPIMS case.

What it shows: the plasma is quasineutral on every scale of the device. The electrons are
strongly magnetised (gyroradius about 0.1 mm, Hall parameter of hundreds) while the ions are
not: the magnetic field holds the electrons and the electric field the ions. At 0.5 Pa the
electron mean free path exceeds the gap, while an ion collides a few times crossing it. At
HiPIMS densities Coulomb collisions outnumber those with the gas and lower the electron Hall
parameter several times. Classical cross-field diffusion is 8 to 80 times below Bohm's at
0.5 Pa; electron transport near Bohm's is what is called anomalous.

Compared: the scales with each other and with the assumed geometry, and classical with Bohm
diffusion.

Reference: Lieberman and Lichtenberg, *Principles of Plasma Discharges and Materials
Processing* (Wiley, 2nd ed. 2005); Chen, *Introduction to Plasma Physics and Controlled
Fusion* (Springer, 3rd ed. 2016), chapters 1, 2 and 5; the NRL Plasma Formulary (Huba, 2019);
Phelps, J. Appl. Phys. 76, 747 (1994) (Ar$^+$ + Ar cross sections).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from plasma_plots.theory import collisions, magnetron, parameters, sheaths
from plasmacoll import argon_ion_processes
from plasmacoll.argon import ARGON_MASS
from plasmacoll.constants import BOLTZMANN, ELECTRON_MASS, PROTON_MASS

PRESETS = {"smoke": {"points": 12, "map": 10}, "full": {"points": 160, "map": 90}}

#: Gas temperature [K].
GAS_TEMPERATURE = 300.0
#: Electron-argon momentum-transfer cross section [m^2]: a constant stand-in, of the size of
#: the measured one at a few eV (no electron-argon set ships with plasmacoll).
ELECTRON_CROSS_SECTION = 1.0e-19
#: Ion temperature [eV]: warm ions, a few times the gas temperature (an assumption).
ION_TEMPERATURE = 0.1
#: Assumed geometry [m]: the target-anode gap and the width of the racetrack.
GAP, RACETRACK_WIDTH = 0.015, 0.010
#: Typical (not measured) conditions: the DC case; the HiPIMS case differs in the density.
DC = {"density": 1e16, "electron_temperature": 3.0, "field": 0.030, "pressure": 0.5}
HIPIMS = {**DC, "density": 1e19}
#: The ranges the figures cover.
DENSITIES, FIELDS, PRESSURES, TEMPERATURES = (
    (1e15, 1e19),
    (0.01, 0.1),
    (0.1, 5.0),
    (1, 10),
)


def ion_momentum_cross_section(lab_energy):
    """Phelps' Ar+ + Ar momentum-transfer cross section Q_i + 2 Q_b [m^2] at the ion energy [eV]."""
    isotropic, backscatter = (p.cross_section for p in argon_ion_processes())
    # the tables are against the centre-of-mass energy: half the ion's on a target at rest
    energy = np.log(0.5 * np.asarray(lab_energy, dtype=float))

    def table(cross_section):
        return np.interp(energy, np.log(cross_section.energy), cross_section.sigma)

    return table(isotropic) + 2.0 * table(backscatter)


def mean_speed(temperature, mass):
    """The mean speed sqrt(8 T / (pi m)) [m/s] of a Maxwellian of ``temperature`` [eV]."""
    return parameters.thermal_speed(temperature, mass, convention="mean")


def scales(density, electron_temperature, field, pressure):
    """Every plasma parameter of the study; the arguments broadcast against each other."""
    n, te, b, p = np.broadcast_arrays(
        *(
            np.asarray(v, dtype=float)
            for v in (density, electron_temperature, field, pressure)
        )
    )
    gas = p / (BOLTZMANN * GAS_TEMPERATURE)
    sigma_i = ion_momentum_cross_section(ION_TEMPERATURE)
    nu_en = gas * ELECTRON_CROSS_SECTION * mean_speed(te, ELECTRON_MASS)
    nu_in = gas * sigma_i * mean_speed(ION_TEMPERATURE, ARGON_MASS)
    nu_ei = collisions.collision_frequencies(
        n, te, ion_temperature=ION_TEMPERATURE, mass_number=ARGON_MASS / PROTON_MASS
    )["electron_ion"]
    hall_e = magnetron.hall_parameter(b, nu_en)
    diffusion_e = te / (ELECTRON_MASS / parameters.elementary_charge * nu_en)
    return {
        "debye_length": parameters.debye_length(n, te),
        "plasma_parameter": parameters.plasma_parameter(n, te),
        "electron_plasma_frequency": parameters.plasma_frequency(n),
        "ion_plasma_frequency": parameters.plasma_frequency(n, ARGON_MASS),
        "electron_gyrofrequency": parameters.cyclotron_frequency(b),
        "ion_gyrofrequency": parameters.cyclotron_frequency(b, ARGON_MASS),
        "electron_gyroradius": parameters.larmor_radius(b, temperature=te),
        "ion_gyroradius": parameters.larmor_radius(
            b, temperature=ION_TEMPERATURE, mass=ARGON_MASS
        ),
        "gas_density": gas,
        "electron_neutral_frequency": nu_en,
        "ion_neutral_frequency": nu_in,
        "electron_ion_frequency": nu_ei,
        "electron_mean_free_path": sheaths.mean_free_path(
            ELECTRON_CROSS_SECTION, density=gas
        ),
        "ion_mean_free_path": sheaths.mean_free_path(sigma_i, density=gas),
        "electron_hall_parameter": hall_e,
        "electron_hall_parameter_coulomb": magnetron.hall_parameter(b, nu_en + nu_ei),
        "ion_hall_parameter": magnetron.hall_parameter(b, nu_in, mass=ARGON_MASS),
        "classical_diffusion": diffusion_e / (1.0 + hall_e**2),
        "bohm_diffusion": sheaths.bohm_diffusion(te, b),
    }


def simulate(preset: str = "full") -> dict:
    """Evaluate the scales over the density, the field and the pressure, and for the two cases."""
    size = PRESETS[preset]
    densities = np.geomspace(*DENSITIES, size["points"])
    fields = np.geomspace(*FIELDS, size["points"])
    temperatures = np.array([1.0, 3.0, 10.0])
    pressures = np.array([0.1, 0.5, 5.0])
    map_fields = np.geomspace(*FIELDS, size["map"])
    map_pressures = np.geomspace(*PRESSURES, size["map"])
    against_density = {
        f"{te:g}": scales(densities, te, DC["field"], DC["pressure"])
        for te in temperatures
    }
    against_field = {
        f"{p:g}": scales(DC["density"], DC["electron_temperature"], fields, p)
        for p in pressures
    }
    hall_map = scales(
        DC["density"],
        DC["electron_temperature"],
        map_fields[None, :],
        map_pressures[:, None],
    )["electron_hall_parameter"]
    cases = {
        name: {**case, **{k: float(v) for k, v in scales(**case).items()}}
        for name, case in (("DC", DC), ("HiPIMS", HIPIMS))
    }
    return {
        "densities": densities,
        "fields": fields,
        "temperatures": temperatures,
        "pressures": pressures,
        "against_density": against_density,
        "against_field": against_field,
        "high_density_frequencies": {
            f"{n:g}": scales(n, DC["electron_temperature"], fields, DC["pressure"])
            for n in (DENSITIES[0], DENSITIES[1])
        },
        "map_fields": map_fields,
        "map_pressures": map_pressures,
        "hall_map": hall_map,
        "cases": cases,
        "assumptions": {
            "gas_temperature": GAS_TEMPERATURE,
            "electron_cross_section": ELECTRON_CROSS_SECTION,
            "ion_temperature": ION_TEMPERATURE,
            "gap": GAP,
            "racetrack_width": RACETRACK_WIDTH,
        },
    }


def _constant(x, value):
    return np.full_like(x, float(value))


def plot_lengths(result: dict) -> None:
    """The length scales against the density, with the gap and the racetrack width."""
    n = result["densities"]
    dc = result["cases"]["DC"]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for te, s in result["against_density"].items():
        ax.plot(n, s["debye_length"] * 1e3, label=f"Debye length, $T_e$ = {te} eV")
    lines = [
        ("electron gyroradius", dc["electron_gyroradius"], "-."),
        ("ion gyroradius", dc["ion_gyroradius"], "-."),
        ("electron mean free path", dc["electron_mean_free_path"], ":"),
        ("ion mean free path", dc["ion_mean_free_path"], ":"),
        ("gap", result["assumptions"]["gap"], "--"),
        ("racetrack width", result["assumptions"]["racetrack_width"], "--"),
    ]
    for label, value, style in lines:
        ax.plot(n, _constant(n, value * 1e3), style, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("plasma density $n$ [m$^{-3}$]")
    ax.set_ylabel("length [mm]")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")


def plot_frequencies_density(result: dict) -> None:
    """The frequencies against the density at the DC field and pressure."""
    n = result["densities"]
    s = result["against_density"]["3"]
    dc = result["cases"]["DC"]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(n, s["electron_plasma_frequency"], label=r"$\omega_{pe}$")
    ax.plot(n, s["ion_plasma_frequency"], label=r"$\omega_{pi}$")
    ax.plot(n, s["electron_ion_frequency"], label=r"$\nu_{ei}$ (Coulomb)")
    lines = [
        (r"$\omega_{ce}$", dc["electron_gyrofrequency"], "-."),
        (r"$\omega_{ci}$", dc["ion_gyrofrequency"], "-."),
        (r"$\nu_{en}$", dc["electron_neutral_frequency"], ":"),
        (r"$\nu_{in}$", dc["ion_neutral_frequency"], ":"),
    ]
    for label, value, style in lines:
        ax.plot(n, _constant(n, value), style, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("plasma density $n$ [m$^{-3}$]")
    ax.set_ylabel("frequency [s$^{-1}$]")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")


def plot_frequencies_field(result: dict) -> None:
    """The gyrofrequencies against the field, with the collision and plasma frequencies."""
    b = result["fields"] * 1e3
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    any_case = next(iter(result["against_field"].values()))
    ax.plot(b, any_case["electron_gyrofrequency"], label=r"$\omega_{ce}$")
    ax.plot(b, any_case["ion_gyrofrequency"], label=r"$\omega_{ci}$")
    for p, s in result["against_field"].items():
        ax.plot(b, s["electron_neutral_frequency"], ":", label=rf"$\nu_{{en}}$, {p} Pa")
    for p, s in result["against_field"].items():
        ax.plot(b, s["ion_neutral_frequency"], ":", label=rf"$\nu_{{in}}$, {p} Pa")
    for n, s in result["high_density_frequencies"].items():
        exponent = int(round(np.log10(float(n))))
        ax.plot(
            b,
            s["electron_plasma_frequency"],
            "--",
            label=rf"$\omega_{{pe}}$, $10^{{{exponent}}}$ m$^{{-3}}$",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("magnetic field $B$ [mT]")
    ax.set_ylabel("frequency [s$^{-1}$]")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")


def plot_hall_map(result: dict) -> None:
    """The electron Hall parameter over (B, p), with the DC case marked."""
    b = result["map_fields"] * 1e3
    p = result["map_pressures"]
    hall = result["hall_map"]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = ax.pcolormesh(b, p, np.log10(hall), shading="auto", cmap="viridis")
    contours = ax.contour(
        b, p, hall, levels=[10, 30, 100, 300, 1000, 3000], colors="white"
    )
    ax.clabel(contours, fmt=lambda v: f"{v:g}")
    dc = result["cases"]["DC"]
    ax.plot(
        [dc["field"] * 1e3],
        [dc["pressure"]],
        "o",
        color="red",
        label="typical DC case",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("magnetic field $B$ [mT]")
    ax.set_ylabel("argon pressure $p$ [Pa]")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(image, label=r"$\log_{10}(\omega_{ce}/\nu_{en})$")


def plot_diffusion(result: dict) -> None:
    """Classical against Bohm cross-field diffusion of the electrons."""
    b = result["fields"] * 1e3
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    any_case = next(iter(result["against_field"].values()))
    ax.plot(b, any_case["bohm_diffusion"], "k-", label=r"Bohm, $T_e/(16 e B)$")
    for p, s in result["against_field"].items():
        ax.plot(b, s["classical_diffusion"], "--", label=f"classical, {p} Pa")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("magnetic field $B$ [mT]")
    ax.set_ylabel(r"electron cross-field diffusion $D_\perp$ [m$^2$/s]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")


def plot(result: dict) -> None:
    """Draw the five figures of the study."""
    plot_lengths(result)
    plot_frequencies_density(result)
    plot_frequencies_field(result)
    plot_hall_map(result)
    plot_diffusion(result)


def _num(value: float) -> str:
    """A number in LaTeX, with a power of ten outside 0.01 to 1000."""
    if value == 0 or 1e-2 <= abs(value) < 1e3:
        return f"${value:.3g}$"
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${float(mantissa):.2g}\cdot10^{{{int(exponent)}}}$"


def table(result: dict) -> str:
    """The parameters of the typical DC and HiPIMS cases as a Markdown table."""
    dc, hipims = result["cases"]["DC"], result["cases"]["HiPIMS"]
    rows = [
        ("plasma density $n$ (m$^{-3}$)", "density", 1.0),
        ("electron temperature $T_e$ (eV)", "electron_temperature", 1.0),
        ("magnetic field $B$ (mT)", "field", 1e3),
        ("argon pressure $p$ (Pa)", "pressure", 1.0),
        ("gas density $n_g$ (m$^{-3}$)", "gas_density", 1.0),
        (r"Debye length $\lambda_D$ (µm)", "debye_length", 1e6),
        (r"electrons per Debye sphere $N_D$", "plasma_parameter", 1.0),
        (r"$\omega_{pe}/2\pi$ (GHz)", "electron_plasma_frequency", 1e-9 / (2 * np.pi)),
        (r"$\omega_{pi}/2\pi$ (MHz)", "ion_plasma_frequency", 1e-6 / (2 * np.pi)),
        (r"$\omega_{ce}/2\pi$ (GHz)", "electron_gyrofrequency", 1e-9 / (2 * np.pi)),
        (r"$\omega_{ci}/2\pi$ (kHz)", "ion_gyrofrequency", 1e-3 / (2 * np.pi)),
        (r"electron gyroradius $\rho_e$ (mm)", "electron_gyroradius", 1e3),
        (r"ion gyroradius $\rho_i$ (mm)", "ion_gyroradius", 1e3),
        (r"$\nu_{en}$ (s$^{-1}$)", "electron_neutral_frequency", 1.0),
        (r"$\nu_{ei}$, Coulomb (s$^{-1}$)", "electron_ion_frequency", 1.0),
        (r"$\nu_{in}$ (s$^{-1}$)", "ion_neutral_frequency", 1.0),
        ("electron mean free path (mm)", "electron_mean_free_path", 1e3),
        ("ion mean free path (mm)", "ion_mean_free_path", 1e3),
        (
            r"electron Hall parameter $\omega_{ce}/\nu_{en}$",
            "electron_hall_parameter",
            1.0,
        ),
        (
            r"with Coulomb collisions, $\omega_{ce}/(\nu_{en} + \nu_{ei})$",
            "electron_hall_parameter_coulomb",
            1.0,
        ),
        (r"ion Hall parameter $\omega_{ci}/\nu_{in}$", "ion_hall_parameter", 1.0),
        (r"classical $D_\perp$ (m$^2$/s)", "classical_diffusion", 1.0),
        (r"Bohm $D_B$ (m$^2$/s)", "bohm_diffusion", 1.0),
    ]
    lines = [
        "Typical values, chosen here, not measured; the HiPIMS case differs from the DC case "
        "only in the density. Collisions with the gas only, unless stated.\n",
        "| Quantity | DC | HiPIMS |",
        "|---|---|---|",
    ]
    lines += [
        f"| {label} | {_num(dc[key] * scale)} | {_num(hipims[key] * scale)} |"
        for label, key, scale in rows
    ]
    a = result["assumptions"]
    lines.append(
        f"\nAssumed: gas at {a['gas_temperature']:g} K, electron–argon "
        f"$\\sigma_m = {_num(a['electron_cross_section'])[1:-1]}$ m$^2$, ions at "
        f"{a['ion_temperature']:g} eV, gap {a['gap'] * 1e3:g} mm, racetrack width "
        f"{a['racetrack_width'] * 1e3:g} mm."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    result = simulate(args.preset)
    plot(result)
    print(table(result))
    plt.show()
