"""Gradient-drift instability.

The textbook candidate for the spokes of a magnetron is the gradient-drift (Simon–Hoh)
instability: the magnetised electrons drift on $\\mathbf E\\times\\mathbf B$ along the racetrack
while the unmagnetised ions do not, and a density ripple along the drift grows when the
equilibrium field and the density gradient point the same way, $\\mathbf E\\cdot\\nabla n > 0$.

The study solves the local (WKB) dispersion relation of wellbalanced (`LocalDispersion`), with
electron inertia, collisions and Poisson's equation, for the ionisation region of a DC
magnetron: $B = 30$ mT, $E = 10^4$ V/m, $n = 10^{17}$ m$^{-3}$, $T_e = 3$ eV, argon, and a
density gradient length $L_n$ of 2 to 10 mm ($x$ away from the target, $y$ along
$\\mathbf E\\times\\mathbf B$, so $E_0 < 0$ and $L_n = n/n' < 0$). In the Simon–Hoh limit (cold
collisionless ions, quasineutrality, no electron inertia) it reduces to
$$\\omega_*\\,\\omega^2 = k_y^2 c_s^2\\,(\\omega - \\omega_0), \\qquad
\\omega_0 = -\\frac{k_y E_0}{B}, \\quad \\omega_* = -\\frac{k_y T_e}{e B L_n}, \\quad
c_s^2 = \\frac{T_e}{M},$$
with growth $\\gamma \\approx k_y\\sqrt{e E_0 L_n/M}$. Electron inertia cuts the mode off near
$\\varepsilon = k_y^2\\rho_e^2\\,\\omega_0/\\omega_* \\approx 1$.

The figures show the growth rate and phase speed against $k_y$ for three gradient lengths and
three fields, the growth rate across the sign of $\\mathbf E\\cdot\\nabla n$, and the effect of
electron collisions and warm ions.

Compared: the solver in the Simon–Hoh limit against the roots of the quadratic, which it
approaches as $\\varepsilon \\to 0$ (relative difference about $\\varepsilon/2$ in the growth
rate), and against Boeuf's closed-form relation with electron inertia (within 0.2 % for
$\\varepsilon < 0.1$).

Reference: A. Simon, Phys. Fluids 6, 382 (1963) and F. C. Hoh, Phys. Fluids 6, 1184 (1963)
for the instability; Boeuf's local relation is Eq. (1) of J.-P. Boeuf, arXiv:2211.17051.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from plasmacoll.constants import ATOMIC_MASS, ELECTRON_MASS, ELEMENTARY_CHARGE
from wellbalanced.dispersion import ColdIons, KineticIons, LocalDispersion, Plasma
from wellbalanced.driftfluid2d import boeuf_dispersion_roots

PRESETS = {
    "smoke": {"wavenumbers": 3, "fields": 2, "signs": 2},
    "quick": {"wavenumbers": 14, "fields": 3, "signs": 8},
    "full": {"wavenumbers": 28, "fields": 6, "signs": 16},
}

ION_MASS = 39.948 * ATOMIC_MASS - ELECTRON_MASS
#: The base case: field [T], electric field [V/m], density [m^-3], T_e [eV], L_n [m].
B0, E0, N0, TE, LN = 0.03, -1.0e4, 1.0e17, 3.0, -5.0e-3
GRADIENT_LENGTHS = (-2.0e-3, -5.0e-3, -10.0e-3)
FIELDS = (0.015, 0.03, 0.06)
#: Electron collision frequencies [1/s] and ion temperatures [eV] of the last figure.
COLLISION_FREQUENCIES = (1.0e8, 1.0e9)
ION_TEMPERATURES = (0.5, 2.0)


def simon_hoh_roots(ky, e_field, b_field, electron_temperature, density_length):
    """Roots of the Simon–Hoh quadratic ``omega_* omega^2 = k^2 c_s^2 (omega - omega_0)``."""
    omega_0 = -ky * e_field / b_field
    omega_star = -ky * electron_temperature / (b_field * density_length)
    cs2 = ELEMENTARY_CHARGE * electron_temperature / ION_MASS
    return np.roots([omega_star, -(ky**2) * cs2, ky**2 * cs2 * omega_0])


def fastest(roots):
    """The root with the largest imaginary part."""
    roots = np.asarray(roots)
    return complex(roots[np.argmax(roots.imag)])


def scan(plasma, ky, ions=None, quasineutral=False):
    """Growth rate and frequency of the gradient-drift branch over ``ky``."""
    ions = ColdIons() if ions is None else ions
    result = LocalDispersion(plasma, ions, quasineutral).scan(ky)
    return result["growth_rate"].values, result["frequency"].values


def growth_across_sign(ky, inverse_lengths):
    """The largest Im omega of the drift ordering at ``ky`` against the signed ``1/L_n``."""
    growth = []
    for inverse in inverse_lengths:
        plasma = Plasma(N0, E0, B0, TE, 1.0 / inverse)
        roots = LocalDispersion(plasma, ColdIons()).roots(0.0, ky).drift(plasma)
        growth.append(max(roots.growth_rate, 0.0) if roots.omega.size else np.nan)
    return np.array(growth)


def simulate(preset: str = "full") -> dict:
    """Solve the local relation for the verification and the parameter scans."""
    size = PRESETS[preset]
    ky = np.geomspace(100.0, 3000.0, size["wavenumbers"])

    # verification: quasineutral, collisionless, cold ions, uniform B
    base = Plasma(N0, E0, B0, TE, LN)
    growth, frequency = scan(base, ky, quasineutral=True)
    quadratic = np.array([fastest(simon_hoh_roots(k, E0, B0, TE, LN)) for k in ky])
    # Boeuf's relation without the grad-B term; a huge density makes it quasineutral
    boeuf = np.array(
        [
            fastest(
                boeuf_dispersion_roots(
                    k, 1e30, E0, B0, TE, LN, np.inf, 0.0, grad_b=False
                )
            )
            for k in ky
        ]
    )
    verification = {
        "growth": growth,
        "frequency": frequency,
        "quadratic": quadratic,
        "approximation": ky * np.sqrt(ELEMENTARY_CHARGE * E0 * LN / ION_MASS),
        "boeuf": boeuf,
    }

    # convergence to the Simon-Hoh limit as electron inertia is made small by raising B
    fields = np.geomspace(0.03, 0.48, size["fields"])
    convergence = {"epsilon": [], "growth_error": [], "frequency_error": [], "ky": []}
    for b in fields:
        plasma = Plasma(N0, E0, b, TE, LN)
        for k in (300.0, 1000.0):
            roots = (
                LocalDispersion(plasma, ColdIons(), True).roots(0.0, k).drift(plasma)
            )
            omega = roots.omega[0]
            exact = fastest(simon_hoh_roots(k, E0, b, TE, LN))
            rho2 = TE * ELECTRON_MASS / (ELEMENTARY_CHARGE * b**2)
            convergence["epsilon"].append(k**2 * rho2 * E0 * LN / TE)
            convergence["growth_error"].append(abs(omega.imag / exact.imag - 1.0))
            convergence["frequency_error"].append(abs(omega.real / exact.real - 1.0))
            convergence["ky"].append(k)
    convergence = {key: np.array(value) for key, value in convergence.items()}

    # Poisson closure: gradient lengths and fields
    lengths = {}
    for length in GRADIENT_LENGTHS:
        g, f = scan(Plasma(N0, E0, B0, TE, length), ky)
        quad = np.array([fastest(simon_hoh_roots(k, E0, B0, TE, length)) for k in ky])
        lengths[length] = {"growth": g, "frequency": f, "quadratic": quad}
    fields_scan = {}
    for b in FIELDS:
        g, f = scan(Plasma(N0, E0, b, TE, LN), ky)
        fields_scan[b] = {"growth": g, "frequency": f}

    # the sign of E . grad n at one wavenumber
    inverse_lengths = np.concatenate(
        [
            -np.geomspace(200.0, 20.0, size["signs"]),
            np.geomspace(20.0, 200.0, size["signs"]),
        ]
    )
    sign_ky = 300.0
    sign = {
        "inverse_lengths": inverse_lengths,
        "ky": sign_ky,
        "growth": growth_across_sign(sign_ky, inverse_lengths),
        "quadratic": np.array(
            [
                max(fastest(simon_hoh_roots(sign_ky, E0, B0, TE, 1.0 / s)).imag, 0.0)
                for s in inverse_lengths
            ]
        ),
    }

    # electron collisions and kinetic ions
    variants = {"cold, collisionless": scan(base, ky)}
    for nu in COLLISION_FREQUENCIES:
        plasma = Plasma(N0, E0, B0, TE, LN, electron_collision_frequency=nu)
        variants[f"ν_e = {nu:.0e} /s"] = scan(plasma, ky)
    for ti in ION_TEMPERATURES:
        variants[f"kinetic ions, T_i = {ti:g} eV"] = scan(
            base, ky, KineticIons(temperature=ti)
        )

    return {
        "ky": ky,
        "verification": verification,
        "convergence": convergence,
        "lengths": lengths,
        "fields": fields_scan,
        "sign": sign,
        "variants": {
            k: {"growth": g, "frequency": f} for k, (g, f) in variants.items()
        },
        "omega_ce": base.omega_ce,
    }


def growing(growth, frequency, ky):
    """Growth rate and phase speed where the mode grows (``nan`` where it does not)."""
    growth, frequency = np.asarray(growth), np.asarray(frequency)
    grows = growth > 1e-6 * np.nanmax(growth)
    return np.where(grows, growth, np.nan), np.where(grows, frequency / ky, np.nan)


def plot(result: dict) -> None:
    """Draw the figures of the study."""
    ky = result["ky"]
    ver = result["verification"]

    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=(6.5, 6.5), constrained_layout=True
    )
    solver = growing(ver["growth"], ver["frequency"], ky)
    boeuf = growing(ver["boeuf"].imag, ver["boeuf"].real, ky)
    quadratic = growing(ver["quadratic"].imag, ver["quadratic"].real, ky)
    axes[0].plot(ky, solver[0], color="C0", lw=2.5, label="LocalDispersion")
    axes[0].plot(ky, boeuf[0], "o", color="C1", ms=4, label="Boeuf's relation")
    axes[0].plot(ky, quadratic[0], "--", color="k", label="Simon–Hoh quadratic")
    axes[0].plot(
        ky, ver["approximation"], ":", color="0.4", label=r"$k_y\sqrt{eE_0L_n/M}$"
    )
    axes[0].set_ylabel("growth rate γ [1/s]")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[1].plot(ky, solver[1], color="C0", lw=2.5, label="LocalDispersion")
    axes[1].plot(ky, boeuf[1], "o", color="C1", ms=4, label="Boeuf's relation")
    axes[1].plot(ky, quadratic[1], "--", color="k", label="Simon–Hoh quadratic")
    axes[1].set_ylabel("phase speed along E×B [m/s]")
    axes[1].set_xlabel("wavenumber $k_y$ [1/m]")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.grid(True, alpha=0.3, which="both")

    conv = result["convergence"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    order = np.argsort(conv["epsilon"])
    eps = conv["epsilon"][order]
    ax.plot(eps, conv["growth_error"][order], "o", color="C0", label="growth rate")
    ax.plot(
        eps, conv["frequency_error"][order], "s", color="C1", label="real frequency"
    )
    ax.plot(eps, eps, "--", color="0.5", label="ε")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        r"electron-inertia parameter $\varepsilon = k_y^2\rho_e^2\,\omega_0/\omega_*$"
    )
    ax.set_ylabel("relative difference from the quadratic")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    for key, title in (("lengths", "L_n"), ("fields", "B")):
        fig, axes = plt.subplots(
            2, 1, sharex=True, figsize=(6.5, 6.5), constrained_layout=True
        )
        for i, (value, data) in enumerate(result[key].items()):
            label = (
                f"$L_n$ = {1e3 * value:g} mm"
                if title == "L_n"
                else f"B = {1e3 * value:g} mT"
            )
            growth, speed = growing(data["growth"], data["frequency"], ky)
            axes[0].plot(ky, growth, color=f"C{i}", lw=2, label=label)
            axes[1].plot(ky, speed, color=f"C{i}", lw=2, label=label)
            if "quadratic" in data:
                growth, speed = growing(
                    data["quadratic"].imag, data["quadratic"].real, ky
                )
                axes[0].plot(ky, growth, "--", color=f"C{i}", lw=1)
                axes[1].plot(ky, speed, "--", color=f"C{i}", lw=1)
        axes[0].set_ylabel("growth rate γ [1/s]")
        axes[0].set_yscale("log")
        axes[0].legend()
        axes[1].set_ylabel("phase speed along E×B [m/s]")
        axes[1].set_xlabel("wavenumber $k_y$ [1/m]")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        for ax in axes:
            ax.grid(True, alpha=0.3, which="both")

    sign = result["sign"]
    # E . grad n / (|E| n) = sign(E_0) / L_n
    drive = np.sign(E0) * sign["inverse_lengths"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    # each side of E . grad n = 0 separately, so that no line crosses the gap
    for k, side in enumerate((drive < 0, drive > 0)):
        order = np.argsort(drive[side])
        ax.plot(
            drive[side][order],
            sign["growth"][side][order],
            "o-",
            color="C0",
            label="LocalDispersion (Poisson)" if k else None,
        )
        ax.plot(
            drive[side][order],
            sign["quadratic"][side][order],
            "--",
            color="k",
            label="Simon–Hoh quadratic" if k else None,
        )
    ax.axvline(0.0, color="0.5", lw=1)
    ax.set_xlabel(r"$\mathbf{E}\cdot\nabla n\,/\,(|E|\,n)$ [1/m]")
    ax.set_ylabel(f"growth rate at $k_y$ = {sign['ky']:g} /m [1/s]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(
        2, 1, sharex=True, figsize=(6.5, 6.5), constrained_layout=True
    )
    styles = ["-", "--", "--", ":", ":"]
    for i, (label, data) in enumerate(result["variants"].items()):
        kwargs = {"color": f"C{i}", "ls": styles[i], "lw": 2.5 if i == 0 else 1.8}
        growth, speed = growing(data["growth"], data["frequency"], ky)
        axes[0].plot(ky, growth, label=label, **kwargs)
        axes[1].plot(ky, speed, label=label, **kwargs)
    axes[0].set_ylabel("growth rate γ [1/s]")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[1].set_ylabel("phase speed along E×B [m/s]")
    axes[1].set_xlabel("wavenumber $k_y$ [1/m]")
    axes[1].set_xscale("log")
    for ax in axes:
        ax.grid(True, alpha=0.3, which="both")

    small = conv["epsilon"] < 0.02
    print(
        f"Simon–Hoh limit: for ε < 0.02 the growth rate differs from the quadratic by at most "
        f"{100 * conv['growth_error'][small].max():.2f} % and the real frequency by "
        f"{100 * conv['frequency_error'][small].max():.2f} %; the smallest ε = "
        f"{conv['epsilon'].min():.1e} gives {conv['growth_error'].min():.1e}."
    )
    rho2 = TE * ELECTRON_MASS / (ELEMENTARY_CHARGE * B0**2)
    inertia = ky**2 * rho2 * E0 * LN / TE
    weak = (inertia < 0.1) & (ver["boeuf"].imag > 0)
    mismatch = np.abs(ver["growth"][weak] / ver["boeuf"].imag[weak] - 1.0).max()
    peak = int(np.nanargmax(ver["growth"]))
    print(
        f"LocalDispersion against Boeuf's relation (quasineutral): growth rates within "
        f"{100 * mismatch:.2f} % for ε < 0.1 (k_y < {ky[weak].max():.0f} /m); at the fastest "
        f"growth (k_y = {ky[peak]:.0f} /m, ε = {inertia[peak]:.2f}) "
        f"{ver['growth'][peak]:.2e} /s against {ver['boeuf'][peak].imag:.2e} /s."
    )
    for length, data in result["lengths"].items():
        j = int(np.nanargmax(data["growth"]))
        print(
            f"L_n = {1e3 * length:g} mm (Poisson): fastest growth {data['growth'][j]:.2e} /s at "
            f"k_y = {ky[j]:.0f} /m (wavelength {2e3 * np.pi / ky[j]:.1f} mm), phase speed "
            f"{data['frequency'][j] / ky[j]:.0f} m/s; e B |L_n| / 2M = "
            f"{ELEMENTARY_CHARGE * B0 * abs(length) / (2 * ION_MASS):.0f} m/s."
        )
    reference = result["variants"]["cold, collisionless"]
    for label, data in result["variants"].items():
        if data is reference:
            continue
        j = int(np.nanargmax(reference["growth"]))
        print(
            f"{label}: fastest growth {np.nanmax(data['growth']):.2e} /s (cold, collisionless "
            f"{reference['growth'][j]:.2e} /s); at k_y = {ky[0]:.0f} /m growth "
            f"{data['growth'][0]:.2e} /s, phase speed {data['frequency'][0] / ky[0]:.0f} m/s "
            f"(cold, collisionless {reference['frequency'][0] / ky[0]:.0f} m/s)."
        )
    stable = sign["growth"][np.sign(E0) * sign["inverse_lengths"] < 0]
    print(
        f"E·∇n < 0: largest growth rate at k_y = {sign['ky']:g} /m is {np.nanmax(stable):.1e} /s "
        f"(stable)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    args = parser.parse_args()
    plot(simulate(args.preset))
    plt.show()
