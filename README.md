# LAPD Explorer

A Python desktop application for BaPSF / Large Plasma Device HDF5 time-series data. Built with PySide6, Matplotlib, NumPy, SciPy and bapsflib.

![LAPD Explorer displaying the supplied x-line acquisition](docs/sample-xline.png)

## Run

The development environment in this directory is already installed:

```sh
.venv/bin/python -m lapd_explorer
```

For a fresh installation (Python 3.11 or newer):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
lapd-explorer
```

The application starts with a synthetic two-component plane wave. **Open HDF5…** loads an acquisition; **Demo** restores the demonstration. Source files are opened read-only. Processing always starts from the imported original, so pressing Apply twice does not integrate twice.

## Features

- Digitizer, ADC, configuration and channel discovery through `bapsflib.lapd.File`.
- One to three channels, with explicit horizontal, vertical and optional third vector-component assignment.
- Motion coordinates matched to global shot numbers through bapsflib, including **bmotion**. Choose target positions when available or measured positions.
- Explicit manual axis mapping; point, x/y/z line, or xy/xz/yz plane. Up to two spatial axes, plus independent case, shot and time axes.
- Per-trace mean removal, linear detrending against time, cumulative trapezoidal integration, gain and averaging over the named shot axis.
- Scalar values, absolute values, two/three-component magnitude, and plane quiver plots.
- Independent mesh/arrow color maps: viridis, plasma, inferno, magma, cividis, turbo, RdBu, coolwarm, Spectral, seismic and grayscale; solid-white arrows are also available.
- Click-to-select spatial slices, numeric slice indices, case/repeat selection, and a linked full time trace.
- Line position–time preview, Welch power spectrum, and point trace statistics.
- Time slider, numeric frame selection, playback, selectable s/ms/µs/ns units and significant figures.
- PNG/SVG/PDF stills, H.264 MP4 with bundled FFmpeg, and portable HDF5 exports preserving named coordinates, units, processing history and acquisition metadata.
- Background file reads, processing and exports. Data/movie exports replace the destination only after successful completion; canceled movies leave it unchanged.

## Importing the supplied September 2026 sample

For `04_bdot_port25_xline 2026-09-02 11.39.39.hdf5`:

1. Click **Open HDF5…** and select the file.
2. On **BaPSF / bapsflib**, select board **3**, channels **6, 7, 8** with Cmd-click on macOS or Ctrl-click on Windows/Linux.
3. Select motion control **bmotion / 3 - <Hermes> p25_C16_Nx91_Lx40cm**.
4. Keep **Motion coordinates** and **Target if available**. Selecting a channel automatically guesses **1 case** and **5 shots per case**; both values remain editable.
5. Click **Load acquisition**.

Validated result: `(x=91, shot=5, time=6144)`, x from −20 to +20 cm, global shots 1–455, sample interval 10 ns. The three channels correspond to Bx, By and Bz in the run notes, but the imported amplitudes are **digitizer volts**. Probe/amplifier calibration is required to obtain magnetic field units. Integration alone produces V·s.

The measured positions contain small x/y deviations. Target positions recover the intended one-dimensional acquisition grid. Both target and measured per-record coordinates are retained in exported metadata. The coordinate source used is recorded per channel. Measured-coordinate grouping can be controlled by the decimal-rounding setting; this changes grouping precision and must be chosen relative to the physical scan spacing.

The time origin defaults to **0 at the first digitizer sample**. Enter a different origin in seconds when a plasma-relative trigger offset is known. No timing offset is inferred from free-text notes.

Choose **Average stored shots** and **Remove mean**, then **Apply to original data** for an averaged baseline-corrected view. The sample is a line, so use **Magnitude** to combine components; vector arrows are available for planes. For a shorter movie, increase the frame stride (for example, 24) and select first/last frame indices.

## Verified 2D sample

For `14_bdot_port25_xy_51x51_delta7mm 2026-09-07 10.35.11.hdf5`, select board **3**, channels **6–8**, motion **bmotion / 0 - <Hermes> p25_C16_51x51_deta7mm**, and **Target if available**. The importer fills **1 case** and **5 shots per case**.

The complete file was verified as `(y=51, x=51, shot=5, time=6144)`: 13,005 global shots, 0.7 cm spacing, x/y extents −17.5 to +17.5 cm, and 10 ns sampling. Selected mapped signals and coordinates were compared with direct bapsflib reads. The full GUI import, averaged baseline correction, three-component magnitude, plane arrows, click-selected x/y slices, still export and a four-frame MP4 were exercised.

![Verified plane visualization with linked slices](docs/sample-plane.png)

The run notes label all three channels “Bx”; the application retains board/channel identifiers and leaves physical component assignment to the user. The verification checks numerical channel combination and arrow rendering, without assuming the notes establish the correct physical vector basis.

Reproduce with:

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/verify_plane.py '/path/to/plane.hdf5'
```

## Dimension mapping

### Motion

Motion grouping requires a complete rectangular grid with the same record count at every position. When a digitizer channel is selected, the importer reads one time sample per record, counts the spatial positions, assumes one case, and fills **Number of shots per case** from the repeated records at each position. The user can replace that guess with any valid combination of cases and shots. Within each position, the acquisition order is mapped according to `case,shot` (shot fastest) or `shot,case` (case fastest). Cases cannot be inferred reliably from position and global shot number alone.

Missing grid points, unequal repeats, three-dimensional scans and mismatched vector shot/time axes produce an error. Select a balanced record range or use manual mapping after preparing the appropriate subset. Single case/repeat dimensions are removed; a missing shot axis never creates statistical information.

### Manual

Add axes in **slowest-to-fastest acquisition order**, excluding time. Size products must equal the number of records. Start/end specify evenly spaced coordinates; non-singleton axes must increase. Examples:

| Stored layout | Manual axes, in order |
| --- | --- |
| A single trace `(nt,)` | No axes |
| Point with repeats `(nshot, nt)` | `shot` |
| Line without stored repeats `(nx, nt)` | `x` |
| Line with repeats `(nx, nshot, nt)` | `x, shot` |
| Plane with cases and repeats `(ny, nx, ncase, nshot, nt)` | `y, x, case, shot` |
| Complete plane scanned again for each case | `case, y, x, shot` |

For **Raw HDF5**, select numeric datasets and supply a sample interval and signal units. The final source axis is interpreted as time; all leading axes are flattened before applying the manual mapping. Raw values have no automatic ADC calibration or global shot-number association. A raw `(nx, nt)` dataset must be mapped as x, rather than interpreted as repeated point measurements.

## Processing and interpretation

Operations run in this order: baseline → integration → optional time smoothing → gain → shot average. Integration uses actual time coordinates with initial value zero. Detrending removes an independently fitted linear baseline from each trace. No case or position averaging occurs. Nonfinite values propagate through means/integration; detrending and PSD reject nonfinite traces. No uncertainty estimate is invented for hardware-averaged or single-shot data.

Enable **Smooth in time** in **02 PREPROCESSING**, then open **Settings…** to choose moving average, Savitzky–Golay, Gaussian, or zero-phase Butterworth low-pass. Only the selected method's parameters are shown. Smoothing works on point, line and plane data with any case/shot axes, without mixing traces. It is off by default; reset or loading data disables it and restores defaults. Settings take effect with **Apply to original data** and are recorded in exported processing history.

Moving/Savitzky–Golay window lengths and Gaussian sigma are in samples; on irregular grids these operate in sample-index space. Savitzky–Golay requires an odd window larger than the polynomial order and no longer than the trace. Butterworth cutoff is in Hz, with sampling frequency inferred from uniformly spaced time coordinates; invalid cutoffs and traces too short for standard padding are rejected. Moving/Gaussian boundary modes are selectable (constant means zero padding); Savitzky–Golay uses polynomial edges. Missing values may propagate or be interpolated along actual time, extending the nearest finite value at endpoints. Entirely missing traces remain NaN. Interpolation occurs after integration and cannot recover samples lost during earlier preprocessing. Smoothing can change the integrated trace's initial zero value.

Magnitude is computed **after** preprocessing each component. Averaging components before magnitude is different from averaging per-shot magnitudes. Vector arrows show the first two assigned components in the displayed plane; the optional third contributes only to the background magnitude. Select distinct channels with matching physical units/calibration. **Mesh color map** controls the scalar/magnitude background; **Arrow color map** independently colors arrows by the in-plane magnitude of their two assigned components. Colored arrows have a separate scale bar. **Fixed scale across time** locks both color ranges. These choices are preserved in stills and MP4 exports.

Frame stride changes playback/export sampling, not the scientific arrays. All processing, cursor traces and PSD use full time resolution. The line position–time preview subsamples time to approximately 1,000 columns to keep interaction responsive. Playback FPS is a target; achievable rate depends on plot size and hardware.

## Export

- **Save image…** captures the full current visualization, including slices/time trace.
- **Export MP4…** renders the inclusive first/last frame range with the chosen stride and FPS. Case, repeat, slices, processing and component selection remain fixed. Cancel discards the partial movie.
- **Save data…** writes every currently processed channel, case and spatial point, including its complete time range. It does not crop to the displayed cursor or export-frame range. The resulting HDF5 file can be reopened directly in this application.
- **Run info** shows acquisition details, dimensions, units and processing history.

## Development and validation

```sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
QT_QPA_PLATFORM=offscreen .venv/bin/python -m lapd_explorer --smoke-test
```

To include the supplied sample in the regression suite:

```sh
LAPD_SAMPLE_FILE='/path/to/04_bdot_port25_xline 2026-09-02 11.39.39.hdf5' \
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

`scripts/verify_sample.py /path/to/sample.hdf5` exercises the actual import dialog, preprocessing, plotting and movie export. It refreshes `docs/sample-xline.png` and writes a short verification movie to the system temporary directory.

The tests cover numerical integration/detrending, named-axis averaging, vector magnitude, spectra, motion grouping and shot provenance, manual mapping, HDF5 round-trips, real bapsflib synthetic-file reading, GUI slices, and MP4 export. The supplied sample is an optional integration test; the source acquisition is not bundled.

Selected data are held in memory. Use the first/stop record controls for large acquisitions. Irregular point clouds and volume scans are outside this application's current scope. Raw import assumes a final time axis. Calibration, arbitrary per-shot parameter tables, per-channel time-offset correction and out-of-core processing are extension points.

## Code map

- `model.py`: validated named-axis dataset, preprocessing, scalar/vector quantities and PSD.
- `io.py`: bapsflib acquisition adapter, motion/manual mapping, raw import and portable HDF5.
- `widgets.py`: background worker and import dialog.
- `plotting.py`: shared plotting renderer with frame updates.
- `app.py`: desktop workflow, animation and exports.

API behavior was checked against the installed bapsflib 2026.4.0 and its [official LAPD documentation](https://bapsflib.readthedocs.io/en/latest/using_lapd/main.html).
