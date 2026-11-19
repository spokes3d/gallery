"""Charoy ExB benchmark.

The verification case of the PIC code before any magnetron physics: the benchmark of Charoy et
al. (Plasma Sources Sci. Technol. 28, 105010, 2019), run by seven independent codes. It is the
axial ($x$, 2.5 cm) and azimuthal ($y$, 1.28 cm, periodic) plane of a xenon discharge, with a
radial magnetic field $B(x)$ out of the plane (a Gaussian of 10 mT at $0.3 L_x$, 6 mT at the
anode and 1 mT on the cathode side) and 200 V between the anode at $x = 0$ and an emission plane
at $x_e = 2.4$ cm. There are no collisions: electron-ion pairs (10 eV and 0.5 eV) are created
from a cosine ionization profile on 0.25 to 1 cm that carries $J_M = 400$ A/m$^2$ of ion
current, and the electrons the anode collects in excess of its ions are emitted again from the
plane $x_e$ at 10 eV (current continuity). The potential is shifted at every step so that its
azimuthal mean is zero at $x_e$, which keeps the cathode sheath out of the discharge.

The electrons drift azimuthally at $E/B$ and drive instabilities in two zones: short waves
($\\lambda \\approx 0.5$ mm, $f \\approx 5$ MHz) between the anode and the field maximum, long
ones ($\\lambda \\approx 2$ mm, $f \\approx 3$ MHz) beyond it. The figures show the azimuthally
and time averaged axial profiles of the axial electric field, the ion density and the electron
temperature, the spectrum of the azimuthal field at the two probe positions, and the emitted
current over $J_M$.

**The published case is a cluster run, and this page does not show it.** It has $500 \\times
256$ cells, $4\\times10^6$ steps of 5 ps and about 550 markers per cell and species at steady
state, around $10^8$ markers. The seven published codes needed 2.5 to 21 days on 32 to 448 CPU
cores (about 60 000 CPU hours). The page shows the `quick` preset: 0.3 $\\mu$s on a coarser
grid with a quarter of the azimuthal length and a fraction of the markers. That breaks the
stability limits of the published case ($\\Delta t\\,\\omega_{pe} < 0.2$ and $\\Delta x <
\\lambda_D$; the table says by how much) and ends long before the published steady state
(10 $\\mu$s), so it checks the set-up and the diagnostics and shows the start-up. `extended`
runs five times longer (about 47 minutes on a busy laptop); `published` is the benchmark.

Compared: the peak ion density, the peak electron temperature, the instability wavelengths and
frequencies and the drift speed, against the values the paper states in its text (Sections 2
and 4.2). The paper's supplementary profiles were not used. At 0.3 $\\mu$s the comparison is
not meaningful for the instabilities (the window is shorter than the periods the paper reports);
the set-up checks (potential drop, zero mean potential at the plane, quasi-neutrality) are in
the tests of picnic.

Reference: T. Charoy et al., Plasma Sources Sci. Technol. 28, 105010 (2019).
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from picnic import Simulation
from picnic.models.vlasov_poisson import VlasovPoisson
from picnic.parameters import (
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    InjectorParams,
    MagneticFieldParams,
    OutputParams,
    ParticleBCs,
    ParticleBoundaryCondition,
    SolverOptions,
    SolverParams,
    SourceParams,
    SymbolicSpaceFunction,
    VlasovPoissonConfig,
)
from picnic.particles.element import Element
from picnic.particles.injector import ChargeBalanceSource
from scipy import constants

#: Values the Charoy et al. (2019) paper states in its text (Section 2 and 4.2).
PUBLISHED = {
    "steady_state_time": 10e-6,
    "peak_ion_density": 5e17,
    "peak_electron_temperature": 50.0,
    "minimum_debye_length": 70e-6,
    "zone_1": {"x": 0.003, "wavelength": 0.5e-3, "frequency": 5e6},
    "zone_2": {"x": 0.015, "wavelength": 2e-3, "frequency": 3e6},
}


@dataclass
class CharoyCase:
    """The benchmark of Charoy et al. (2019) in SI units; the defaults are the published case.

    ``cells`` is the number of cells along x and y (the grid has one more node in x, and
    ``cells[1]`` nodes in y, which is periodic). ``particles_per_cell`` is the initial number
    of markers of each species per cell: 150 is the nominal case 1 (about 550 at steady
    state); 75 and 300 are the convergence cases 2 and 3.
    """

    length_x: float = 0.025
    length_y: float = 0.0128
    cells: tuple[int, int] = (500, 256)
    dt: float = 5e-12
    t_end: float = 20e-6
    voltage: float = 200.0
    emission_x: float = 0.024
    # Radial magnetic field: two Gaussian halves meeting at the maximum.
    b_anode: float = 6e-3
    b_cathode: float = 1e-3
    b_max: float = 10e-3
    b_max_x: float = 0.3  # fraction of length_x
    b_sigma: float = 0.25  # fraction of length_x
    # The cosine ionization profile and the ion current it carries.
    ionization_x: tuple[float, float] = (0.0025, 0.01)
    max_ion_current: float = 400.0
    number_density: float = 5e16
    electron_temperature_ev: float = 10.0
    ion_temperature_ev: float = 0.5
    emission_temperature_ev: float = 10.0
    particles_per_cell: float = 150.0
    # Push the ions only every ``ion_subcycle`` steps (the published codes did this
    # with 11 in one case without a visible effect).
    ion_subcycle: int = 1
    seed: int = 11

    @property
    def node_spacing(self) -> tuple[float, float]:
        """Grid spacing along x and y."""
        return self.length_x / self.cells[0], self.length_y / self.cells[1]

    @property
    def num_gridpoints(self) -> tuple[int, int]:
        """Grid nodes: one more than the cells along x, the cells along (periodic) y."""
        return self.cells[0] + 1, self.cells[1]

    @property
    def macroparticle_weight(self) -> float:
        """Physical particles per metre of depth carried by one marker."""
        dx, dy = self.node_spacing
        return self.number_density * dx * dy / self.particles_per_cell

    @property
    def ionization_rate(self) -> float:
        """Pairs created per second per metre of depth: ``L_y int S(x) dx``."""
        low, high = self.ionization_x
        return self.length_y * 2.0 / np.pi * (high - low) * self.peak_ionization_rate

    @property
    def peak_ionization_rate(self) -> float:
        """``S_0`` [m^-3 s^-1] that makes the ion current density ``max_ion_current``."""
        low, high = self.ionization_x
        return np.pi * self.max_ion_current / (2.0 * constants.e * (high - low))

    @property
    def magnetic_field(self) -> MagneticFieldParams:
        """``B_z(x)``: Gaussians of width ``b_sigma L_x`` that meet at the maximum."""
        centre = self.b_max_x * self.length_x
        sigma = self.b_sigma * self.length_x
        halves = []
        for edge, value in ((0.0, self.b_anode), (self.length_x, self.b_cathode)):
            drop = float(np.exp(-0.5 * ((edge - centre) / sigma) ** 2))
            amplitude = (self.b_max - value) / (1.0 - drop)
            halves.append(
                f"{amplitude!r} * exp(-(x - {centre!r})**2 / {2.0 * sigma**2!r}) "
                f"+ {self.b_max - amplitude!r}"
            )
        return MagneticFieldParams(
            bz=f"Piecewise(({halves[0]}, x < {centre!r}), ({halves[1]}, True))",
            unit="T",
        )

    def field_at(self, x: float) -> float:
        """``B_z`` of the case at ``x`` [T]."""
        centre = self.b_max_x * self.length_x
        sigma = self.b_sigma * self.length_x
        edge, value = (
            (0.0, self.b_anode) if x < centre else (self.length_x, self.b_cathode)
        )
        drop = np.exp(-0.5 * ((edge - centre) / sigma) ** 2)
        amplitude = (self.b_max - value) / (1.0 - drop)
        return amplitude * np.exp(-((x - centre) ** 2) / (2.0 * sigma**2)) + (
            self.b_max - amplitude
        )

    def sources(self) -> list[SourceParams]:
        """Pair creation (same positions for both species) and the emission plane."""
        low, high = self.ionization_x
        pairs = [
            SourceParams(
                species=name,
                type="volume",
                rate=self.ionization_rate,
                lower_bound=(low, 0.0),
                upper_bound=(high, self.length_y),
                profile={"shape": "cosine"},
                pair_seed=self.seed,
                seed=self.seed + 1 + index,
            )
            for index, name in enumerate(("electrons", "ions"))
        ]
        emission = SourceParams(
            species="electrons",
            type="charge_balance",
            position=self.emission_x,
            target="x_min",
            temperature=self.emission_temperature_ev,
            seed=self.seed + 5,
        )
        return [*pairs, emission]

    def stability_report(self) -> dict[str, float | bool]:
        """The explicit-PIC limits at the density and temperature the published run reaches.

        At the peak density of about 5e17 m^-3 and 50 eV the electron plasma frequency
        and Debye length set ``dt omega_pe < 0.2`` and ``dx < lambda_D``.
        """
        n = PUBLISHED["peak_ion_density"]
        omega = float(
            np.sqrt(n * constants.e**2 / (constants.epsilon_0 * constants.m_e))
        )
        debye_peak = float(
            np.sqrt(
                constants.epsilon_0
                * PUBLISHED["peak_electron_temperature"]
                / (n * constants.e)
            )
        )
        dx = max(self.node_spacing)
        return {
            "omega_pe_dt": omega * self.dt,
            "dx_over_debye": dx / debye_peak,
            "dt_ok": omega * self.dt < 0.2,
            "dx_ok": dx < debye_peak,
        }


#: The published case: a cluster run (see the docstring).
CASE = CharoyCase()

#: A quarter of the azimuthal length, twice the cell size, a fifth of the markers and 6.4 times
#: the time step, for 0.3 microseconds, a thirtieth of the published run, which is not enough
#: to reach its steady state. It violates ``dt omega_pe < 0.2`` and ``dx < lambda_D``.
REDUCED = replace(
    CASE,
    length_y=0.0032,
    cells=(250, 32),
    dt=3.2e-11,
    t_end=0.3e-6,
    particles_per_cell=20.0,
    ion_subcycle=4,
)

#: ``REDUCED`` run five times longer with 8 markers per cell: the ions cross the gap in 1.4
#: microseconds, so this reaches the start of the quasi-steady state, but the marker noise is
#: large (the benchmark asks for more than 250 per cell).
EXTENDED = replace(REDUCED, t_end=1.5e-6, particles_per_cell=8.0)

#: Seconds, for the test suite: a tiny grid that checks the study runs end to end.
SMOKE = replace(
    REDUCED,
    length_y=0.0016,
    cells=(100, 16),
    dt=1e-10,
    t_end=4e-8,
    particles_per_cell=6.0,
)

#: The case, and the steps between two recorded samples, of every preset.
PRESETS = {
    "smoke": (SMOKE, 5),
    "quick": (REDUCED, 20),
    "extended": (EXTENDED, 20),
    "published": (CASE, 1000),
}


def charoy_config(case: CharoyCase) -> VlasovPoissonConfig:
    """Return the configuration of ``case``."""
    return VlasovPoissonConfig(
        grid=GridParams(
            ndim=2,
            nvel=3,
            num_gridpoints=case.num_gridpoints,
            lower_bound=(0.0, 0.0),
            upper_bound=(case.length_x, case.length_y),
            coordinates="cartesian",
            domain_decomposition=(False, False),
        ),
        solver=SolverParams(
            pde="Poisson2D",
            options=SolverOptions(type="patchwork", reference_plane=case.emission_x),
        ),
        bcs_field=FieldBCs(
            x_lower=FieldBoundaryCondition(
                "dirichlet", SymbolicSpaceFunction(str(case.voltage), unit="V")
            ),
            x_upper=FieldBoundaryCondition(
                "dirichlet", SymbolicSpaceFunction("0.0", unit="V")
            ),
            y_lower=FieldBoundaryCondition("periodic"),
            y_upper=FieldBoundaryCondition("periodic"),
        ),
        bcs_particles=ParticleBCs(
            x_lower=ParticleBoundaryCondition("kill"),
            x_upper=ParticleBoundaryCondition("kill"),
            y_lower=ParticleBoundaryCondition("periodic"),
            y_upper=ParticleBoundaryCondition("periodic"),
        ),
        injector=InjectorParams(
            lower_bound=(0.0, 0.0), upper_bound=(case.length_x, case.length_y)
        ),
        species={
            "electrons": Element(
                "e-",
                name="electrons",
                name_latex="e^-",
                number_density=case.number_density / u.m**3,
                temperature=case.electron_temperature_ev * u.eV,
                macroparticle_weight=case.macroparticle_weight,
                injection_method="manual",
            ),
            "ions": Element(
                "Xe 1+",
                name="ions",
                name_latex="Xe^+",
                number_density=case.number_density / u.m**3,
                temperature=case.ion_temperature_ev * u.eV,
                macroparticle_weight=case.macroparticle_weight,
                injection_method="manual",
                subcycle=case.ion_subcycle,
            ),
        },
        dt=case.dt,
        sources=case.sources(),
        magnetic_field=case.magnetic_field,
    )


# Diagnostics ------------------------------------------------------------------#


def species_density(model: VlasovPoisson, name: str) -> np.ndarray:
    """Number density of ``name`` on the grid, shape ``num_gridpoints``."""
    species = model.species_registry.get(name)
    species.update_charge_density()
    return np.asarray(
        species.number_density_field.field_data.local_with_halos
    ).squeeze()


def electron_temperature_profile(
    model: VlasovPoisson, case: CharoyCase, bins: int
) -> np.ndarray:
    """Electron temperature [eV] in ``bins`` slabs of x: the velocity variance over three.

    ``T = m (var v_x + var v_y + var v_z) / 3`` per slab, so the azimuthal drift is not
    counted as heat; NaN in a slab with fewer than two markers. The velocities are the
    pusher's (staggered by half a step from the positions: no correction is applied).
    """
    electrons = model.species_registry.get("electrons")
    x = np.asarray(electrons.markers.get_active_positions())[0]
    v = np.asarray(electrons.markers.get_active_velocities())
    index = np.clip((x / case.length_x * bins).astype(int), 0, bins - 1)
    count = np.bincount(index, minlength=bins).astype(float)
    energy = np.zeros(bins)
    for component in range(v.shape[0]):
        mean = np.bincount(index, weights=v[component], minlength=bins)
        mean2 = np.bincount(index, weights=v[component] ** 2, minlength=bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            energy += mean2 / count - (mean / count) ** 2
    with np.errstate(invalid="ignore"):
        return np.where(count > 1, constants.m_e * energy / (3.0 * constants.e), np.nan)


class Recorder:
    """Collects the benchmark diagnostics while the simulation runs (a callback).

    Every ``every`` steps it stores the azimuthal electric field at the probe positions
    (for the spectra) and the emitted current over ``J_M``; from ``average_from`` on
    it also accumulates the azimuthal means of the axial field, the ion and electron
    densities and the electron temperature.
    """

    def __init__(
        self,
        case: CharoyCase,
        average_from: float,
        probes: Sequence[float] = (0.003, 0.015),
        temperature_bins: int = 100,
    ) -> None:
        self.case = case
        self.average_from = average_from
        self.probes = tuple(probes)
        self.temperature_bins = temperature_bins
        self.times: list[float] = []
        self.probe_fields: list[list[np.ndarray]] = [[] for _ in self.probes]
        self.emitted_current: list[float] = []
        self.sums: dict[str, np.ndarray] = {}
        self.num_averaged = 0
        self._last_emitted = 0.0
        self._last_time = 0.0
        nx = case.num_gridpoints[0]
        self.x = np.linspace(0.0, case.length_x, nx)
        self.probe_index = [round(p / case.length_x * (nx - 1)) for p in probes]

    def result(self) -> dict[str, Any]:
        """The recorded series and time-averaged profiles, as plain arrays."""
        out = {
            "x": self.x,
            "times": np.array(self.times),
            "emitted_current": np.array(self.emitted_current),
            "probes": self.probes,
            "probe_fields": [np.array(f) for f in self.probe_fields],
            "num_averaged": self.num_averaged,
            "t_bins": (
                (np.arange(self.temperature_bins) + 0.5)
                * self.case.length_x
                / self.temperature_bins
            ),
        }
        for name, total in self.sums.items():
            out[name] = total / max(self.num_averaged, 1)
        return out

    def __call__(self, simulation: Simulation) -> None:
        model = simulation.model
        case = self.case
        e_field = np.asarray(model.electric_field.field_data.local_with_halos)
        e_field = e_field.reshape(*case.num_gridpoints, -1)
        for k, index in enumerate(self.probe_index):
            self.probe_fields[k].append(e_field[index, :, 1].copy())
        self.times.append(model.time)
        source = next(
            s for _, s in model.particle_sources if isinstance(s, ChargeBalanceSource)
        )
        emitted = source.emitted_weight
        interval = model.time - self._last_time
        self.emitted_current.append(
            constants.e
            * (emitted - self._last_emitted)
            / (case.length_y * interval)
            / case.max_ion_current
            if interval > 0
            else 0.0
        )
        self._last_emitted, self._last_time = emitted, model.time
        if model.time < self.average_from:
            return
        values = {
            "e_x": e_field[:, :, 0].mean(axis=1),
            "n_i": species_density(model, "ions").mean(axis=1),
            "n_e": species_density(model, "electrons").mean(axis=1),
            "t_e": electron_temperature_profile(model, case, self.temperature_bins),
            "potential": np.asarray(
                model.electrostatic_potential.field_data.local_with_halos
            )
            .reshape(*case.num_gridpoints)
            .mean(axis=1),
        }
        for name, value in values.items():
            self.sums.setdefault(name, np.zeros_like(value))
            self.sums[name] = self.sums[name] + np.nan_to_num(value)
        self.num_averaged += 1


def dominant_mode(
    field: np.ndarray, times: np.ndarray, length_y: float, min_periods: float = 2.0
) -> dict[str, float]:
    """The strongest azimuthal wave in ``field[time, y]``: wavelength, frequency, phase speed.

    A Hann-windowed 2D FFT over y and time; the strongest ``omega > 0`` bin with
    ``k != 0`` (a real signal repeats it at ``-omega, -k``) among the frequencies of at
    least ``min_periods`` periods in the time span (``min_frequency`` in the result:
    below it the window cannot tell a wave from a drift of the mean). The phase
    speed is positive towards +y (``exp(i(k y - omega t))`` has its peak at ``omega > 0``
    and ``k < 0`` in the FFT's convention).
    """
    nt, ny = field.shape
    dt = float(np.mean(np.diff(times)))
    span = float(times[-1] - times[0])
    window = np.hanning(nt)[:, None] * np.hanning(ny)[None, :]
    data = (field - field.mean()) * window
    spectrum = np.abs(np.fft.fft2(data)) ** 2
    omega = np.fft.fftfreq(nt, dt)  # Hz
    wave = np.fft.fftfreq(ny, length_y / ny)  # cycles per metre
    min_frequency = min_periods / span
    keep = (omega[:, None] >= min_frequency) & (wave[None, :] != 0)
    peak = np.unravel_index(np.argmax(np.where(keep, spectrum, 0.0)), spectrum.shape)
    f, cycles = float(omega[peak[0]]), float(wave[peak[1]])
    return {
        "frequency": f,
        "wavelength": 1.0 / abs(cycles) if cycles else float("nan"),
        "phase_speed": -f / cycles if cycles else float("nan"),
        "span": span,
        "min_frequency": min_frequency,
    }


def analyze(result: dict[str, Any], case: CharoyCase, window: float) -> dict[str, Any]:
    """The comparison with the published numbers.

    ``window`` is the time span at the end of the run that the spectra use [s].
    """
    report: dict[str, Any] = {}
    n_i, t_e, e_x = result["n_i"], result["t_e"], result["e_x"]
    x, x_t = result["x"], result["t_bins"]
    # the artificial sheath between the emission plane and the right boundary is not part
    # of the discharge: the peaks are taken before it (2 mm ahead of the plane)
    region = x < case.emission_x - 2e-3
    region_t = x_t < case.emission_x - 2e-3
    report["peak_ion_density"] = float(np.max(n_i[region]))
    report["peak_ion_density_x"] = float(x[region][np.argmax(n_i[region])])
    report["peak_electron_temperature"] = float(np.nanmax(t_e[region_t]))
    report["peak_axial_field"] = float(np.max(e_x[region]))
    report["peak_axial_field_x"] = float(x[region][np.argmax(e_x[region])])
    times = result["times"]
    current = result["emitted_current"]
    tail = times > times[-1] - window
    report["emitted_current_over_jm"] = float(np.mean(current[tail]))
    report["modes"] = []
    b_center = case.b_max_x * case.length_x
    for probe, series in zip(result["probes"], result["probe_fields"], strict=True):
        keep = times > times[-1] - window
        mode = dominant_mode(series[keep], times[keep], case.length_y)
        # the drift speed E/B at the probe
        sample = int(np.argmin(np.abs(x - probe)))
        mode["x"] = float(probe)
        mode["drift_speed"] = float(-e_x[sample] / case.field_at(probe))
        mode["zone"] = 1 if probe < b_center else 2
        report["modes"].append(mode)
    return report


# Running ------------------------------------------------------------------------#


def simulate(preset: str = "quick", *, average_fraction: float = 0.4) -> dict:
    """Run the case of ``preset`` and return the diagnostics and the report.

    The profiles are averaged over the last ``average_fraction`` of the run and the
    spectra use the same window.
    """
    case, record_every = PRESETS[preset]
    model = VlasovPoisson(charoy_config(case))
    sim = Simulation(
        model, t_end=case.t_end, output=OutputParams(path=None, every=10**9)
    )
    recorder = Recorder(case, average_from=case.t_end * (1.0 - average_fraction))
    sim.add_callback(recorder, every=record_every)
    start = time.time()
    sim.run()
    result = recorder.result()
    result["wall_time"] = time.time() - start
    result["case"] = asdict(case)
    result["num_markers"] = {
        name: int(model.species_registry.get(name).num_mks)
        for name in ("electrons", "ions")
    }
    result["report"] = analyze(result, case, window=case.t_end * average_fraction)
    result["stability"] = case.stability_report()
    return result


def table(result: dict) -> str:
    """The comparison with the published values as a Markdown table."""
    report, stability = result["report"], result["stability"]
    case = CharoyCase(**result["case"])
    published_stability = CASE.stability_report()
    rows = [
        "| | this run | published |",
        "|---|---|---|",
        f"| duration | {case.t_end * 1e6:.2f} µs | 20 µs (steady from ~10 µs) |",
        f"| grid, markers per cell | {case.cells[0]} × {case.cells[1]}, "
        f"{case.particles_per_cell:g} initial | 500 × 256, 150 initial (≈550 steady) |",
        f"| markers at the end | {result['num_markers']['electrons']:.2e} electrons | "
        "≈ 7×10⁷ per species |",
        f"| Δt ω_pe (limit 0.2, at 5e17 m⁻³) | {stability['omega_pe_dt']:.2f} "
        f"({'ok' if stability['dt_ok'] else '**violated**'}) | "
        f"{published_stability['omega_pe_dt']:.2f} |",
        f"| Δx / λ_D (limit 1, at 5e17 m⁻³, 50 eV) | {stability['dx_over_debye']:.2f} "
        f"({'ok' if stability['dx_ok'] else '**violated**'}) | "
        f"{published_stability['dx_over_debye']:.2f} |",
        f"| peak n_i | {report['peak_ion_density']:.2e} m⁻³ at "
        f"{report['peak_ion_density_x'] * 1e3:.1f} mm | ≈ {PUBLISHED['peak_ion_density']:.0e} m⁻³ |",
        f"| peak T_e | {report['peak_electron_temperature']:.0f} eV | "
        f"≈ {PUBLISHED['peak_electron_temperature']:.0f} eV |",
        f"| emitted current / J_M | {report['emitted_current_over_jm']:.2f} | 1 (current continuity) |",
    ]
    for mode in report["modes"]:
        published = PUBLISHED[f"zone_{mode['zone']}"]
        floor = mode["min_frequency"] / 1e6
        note = (
            ""
            if published["frequency"] >= mode["min_frequency"]
            else f" (not resolved: this window resolves only above {floor:.0f} MHz)"
        )
        rows.append(
            f"| zone {mode['zone']} (x = {mode['x'] * 1e3:.0f} mm): λ, f of the strongest wave | "
            f"{mode['wavelength'] * 1e3:.2f} mm, {mode['frequency'] / 1e6:.0f} MHz | "
            f"≈ {published['wavelength'] * 1e3:.1f} mm, ≈ {published['frequency'] / 1e6:.0f} MHz"
            f"{note} |"
        )
        rows.append(
            f"| zone {mode['zone']}: phase speed along y, E×B drift along y "
            f"(window {mode['span'] * 1e6:.2f} µs) | "
            f"{mode['phase_speed'] / 1e3:.0f} km/s, {mode['drift_speed'] / 1e3:.0f} km/s | |"
        )
    return "\n".join(rows)


def plot(result: dict) -> None:
    """Draw the three figures of the study."""
    case = CharoyCase(**result["case"])
    print(table(result))
    peak = case.b_max_x * case.length_x * 1e3
    x = result["x"] * 1e3

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True, constrained_layout=True)
    series = (
        (result["e_x"], x, "$E_x$ [V/m]", None),
        (result["n_i"], x, "$n_i$ [m$^{-3}$]", PUBLISHED["peak_ion_density"]),
        (
            result["t_e"],
            result["t_bins"] * 1e3,
            "$T_e$ [eV]",
            PUBLISHED["peak_electron_temperature"],
        ),
    )
    for ax, (values, grid, label, published) in zip(axes, series, strict=True):
        ax.plot(grid, values)
        ax.axvline(peak, ls="--", c="gray")
        if published is not None:
            ax.axhline(published, ls=":", c="gray")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("x [mm]")
    fig.suptitle("Averaged axial profiles (reduced run)")

    probes = result["probes"]
    fig, axes = plt.subplots(
        len(probes),
        1,
        figsize=(7, 2.8 * len(probes) + 0.8),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    times = result["times"]
    span = result["report"]["modes"][0]["span"]
    window = times > times[-1] - span - 1e-12
    for ax, probe, series in zip(
        axes[:, 0], probes, result["probe_fields"], strict=True
    ):
        data = series[window]
        dt = float(np.mean(np.diff(times[window])))
        spectrum = np.abs(np.fft.rfft(data - data.mean(), axis=0)) ** 2
        freq = np.fft.rfftfreq(data.shape[0], dt) / 1e6
        zone = 1 if probe * 1e3 < peak else 2
        ax.semilogy(freq[1:], spectrum.mean(axis=1)[1:])
        ax.axvline(PUBLISHED[f"zone_{zone}"]["frequency"] / 1e6, ls="--", c="gray")
        ax.set_title(f"x = {probe * 1e3:.0f} mm, zone {zone}")
        ax.set_ylabel("$|E_y|^2$ (azimuthal mean)")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("f [MHz]")
    fig.suptitle("Spectrum of the azimuthal field (dashed: published frequency)")

    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.plot(times * 1e6, result["emitted_current"])
    ax.set_xlabel("t [$\\mu$s]")
    ax.set_ylabel("$J_{ec1} / J_M$")
    ax.set_title("Emitted electron current over the ion current")
    ax.grid(True, alpha=0.3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    args = parser.parse_args()
    plot(simulate(args.preset))
    plt.show()
