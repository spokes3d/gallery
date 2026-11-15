# SPOKES3D gallery

Studies of magnetron discharges and the codes behind them, from the SPOKES3D project: verification of the codes, method demonstrations and examples.

Site: <https://spokes3d.github.io/gallery/>

| Study | Kind | Codes |
|---|---|---|
| [Hall current field](https://spokes3d.github.io/gallery/studies/hall-current-field/) | example | none |
| [Rotating-spoke test electrons](https://spokes3d.github.io/gallery/studies/rotating-spoke-test-electrons/) | example | byob, mpifield, mpiparticles, plasmacoll |
| [Rotating-spoke test ions](https://spokes3d.github.io/gallery/studies/rotating-spoke-test-ions/) | example | byob, mpifield, mpiparticles, plasmacoll |
| [Langmuir probe in spokes](https://spokes3d.github.io/gallery/studies/spoke-langmuir-probe/) | example | byob, plasmacoll |
| [PIC cost against density](https://spokes3d.github.io/gallery/studies/pic-cost-hipims/) | example | none |
| [Magnetron magnetic field](https://spokes3d.github.io/gallery/studies/magnetron-field/) | example | none |
| [Magnetron electron orbits](https://spokes3d.github.io/gallery/studies/magnetron-electron-orbits/) | verification | mpiparticles, plasmacoll |
| [Gradient-drift instability](https://spokes3d.github.io/gallery/studies/gradient-drift-dispersion/) | verification | wellbalanced, plasmacoll |
| [Magnetron plasma parameters](https://spokes3d.github.io/gallery/studies/magnetron-plasma-parameters/) | example | plasmacoll, plasma-plots |
| [Cathode sheath](https://spokes3d.github.io/gallery/studies/cathode-sheath/) | example | byob, plasmacoll |

## Running a study

```bash
uv sync
uv run python studies/<study>/study.py --preset smoke
uv run gallery build --cached-only   # the pages of the site, from the cached results
```

This repository is generated from a private gallery with `gallery export` ([gallery-template](https://github.com/spokes3d/gallery-template)): changes come in as pull requests from the sync and are not made here by hand.
