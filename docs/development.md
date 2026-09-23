# Development and validation

Run all commands from the repository root. Follow the main
[installation instructions](../README.md#install-from-a-fresh-github-download)
first, but install the test extra:

```sh
.venv/bin/python -m pip install -e '.[test]'
```

On Windows, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

## Automated checks

macOS/Linux:

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
QT_QPA_PLATFORM=offscreen .venv/bin/python -m lapd_explorer --smoke-test
```

Windows PowerShell:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m lapd_explorer --smoke-test
Remove-Item Env:QT_QPA_PLATFORM
```

The tests cover numerical integration/detrending, named-axis averaging, vector
magnitude, smoothing, spectra, motion grouping and shot provenance, manual
mapping, portable-HDF5 round trips, real `bapsflib` synthetic-file reads, GUI
slices, Langmuir analysis, spectral analysis, and MP4 export.

## Optional real-acquisition regression

The September 2026 sample files are not distributed with the repository. If the
x-line file is available, include it in the test suite as follows.

macOS/Linux:

```sh
LAPD_SAMPLE_FILE='/path/to/04_bdot_port25_xline 2026-09-02 11.39.39.hdf5' \
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

Windows PowerShell:

```powershell
$env:LAPD_SAMPLE_FILE = 'C:\path\to\04_bdot_port25_xline 2026-09-02 11.39.39.hdf5'
$env:QT_QPA_PLATFORM = 'offscreen'
.\.venv\Scripts\python.exe -m pytest -q
Remove-Item Env:LAPD_SAMPLE_FILE
Remove-Item Env:QT_QPA_PLATFORM
```

## Verification scripts and screenshots

`scripts/verify_sample.py` exercises the x-line import dialog, preprocessing,
plotting, and movie export. It also regenerates `docs/sample-xline.png` from the
real GUI in Light mode. Its short verification movie is written to the system
temporary directory.

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/verify_sample.py '/path/to/xline.hdf5'
```

`scripts/verify_plane.py` checks the complete 51 × 51 plane against direct
`bapsflib` reads, exercises vector and slice rendering, regenerates
`docs/sample-plane.png` in Light mode, and writes temporary SVG/MP4 outputs.

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/verify_plane.py '/path/to/plane.hdf5'
```

For Windows PowerShell, set `$env:QT_QPA_PLATFORM = 'offscreen'` and replace
`.venv/bin/python` with `.\.venv\Scripts\python.exe` in either command.

Both scripts overwrite their tracked documentation screenshot. Review the PNG
before committing it.

## Standalone application build

The checked-in PyInstaller spec currently targets the macOS `.app` bundle. Build
it on macOS with:

```sh
.venv/bin/python -m pip install pyinstaller
.venv/bin/python -m PyInstaller --clean 'LAPD HDF5 Explorer.spec'
```

The output is written below `dist/`. PyInstaller bundles are platform-specific;
the existing spec should not be presented as a Windows or Linux installer without
adding and testing platform-specific packaging first.

## Code map

- `lapd_explorer/model.py`: named-axis dataset, preprocessing, scalar/vector
  quantities, and basic PSD.
- `lapd_explorer/io.py`: BaPSF adapter, motion/manual mapping, raw import, and
  portable HDF5.
- `lapd_explorer/smoothing.py`: time-domain smoothing/filter implementations.
- `lapd_explorer/widgets.py`: background worker and import/setup dialogs.
- `lapd_explorer/plotting.py`: shared plot renderer and frame updates.
- `lapd_explorer/app.py`: main desktop workflow, animation, and exports.
- `lapd_explorer/langmuir_analysis_core.py`: Qt-independent Langmuir fitting.
- `lapd_explorer/langmuir_gui.py`: Langmuir workflow and result display.
- `lapd_explorer/spectral.py`: Qt-independent spectral estimators and results.
- `lapd_explorer/spectral_gui.py`: spectral workflow and animations.
