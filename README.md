# LAPD Explorer

<img src="lapd_explorer/assets/BaPSF_Logo+Name_Color_RGB.png" alt="Basic Plasma Science Facility" width="300">

LAPD Explorer is a cross-platform Python desktop application for inspecting,
processing, visualizing, and exporting BaPSF / Large Plasma Device (LAPD) HDF5
time-series data. It uses PySide6, Matplotlib, NumPy, SciPy, and `bapsflib`.

![LAPD Explorer showing a processed x-line acquisition in Light mode](docs/sample-xline.png)

*A real three-channel x-line acquisition after stored-shot averaging and mean
removal. The application starts in Light mode; Dark mode remains available from
the Appearance menu.*

## Highlights

- Opens BaPSF acquisitions through `bapsflib.lapd.File`, as well as numeric raw
  HDF5 datasets and portable HDF5 files previously exported by the application.
- Maps point, line, and plane acquisitions with named spatial, case, shot, and
  time axes.
- Displays scalar, absolute, two/three-component magnitude, and plane-vector
  views with linked slices and time traces.
- Applies per-trace baseline removal, detrending, integration, temporal
  smoothing, gain, and stored-shot averaging without modifying the source file.
- Provides Welch spectra, cross-spectral analysis, covariance, and Langmuir
  probe analysis.
- Exports PNG, SVG, or PDF figures; H.264 MP4 movies; and portable HDF5 data.

## Requirements

- A 64-bit installation of Python 3.11 or newer.
- macOS, Linux, or Windows with a graphical desktop session.
- Enough memory for the selected acquisition range. Selected data are held in
  memory; large files can be limited with the first/stop record controls.
- Git, if cloning the repository. Git is not needed when using GitHub's
  **Code → Download ZIP** option.

Python installs the application dependencies, Qt, and the FFmpeg executable used
for movie export. A separate Qt or FFmpeg installation is normally unnecessary.

## Install from a fresh GitHub download

### 1. Get the source

Clone the repository:

```text
git clone https://github.com/svincena/bapsf_hdf5_explorer.git
cd bapsf_hdf5_explorer
```

Alternatively, download the ZIP from GitHub, extract it, and open a terminal in
the extracted `bapsf_hdf5_explorer` folder. Every command below must be run from
that folder—the one containing `pyproject.toml`.

### 2. Create an isolated environment and install

The commands deliberately use the virtual environment's Python directly, so
activation is optional and Windows PowerShell execution-policy settings do not
get in the way.

#### macOS

First confirm that `python3 --version` reports 3.11 or newer. If it does not,
install a current Python from [python.org](https://www.python.org/downloads/macos/)
or another trusted Python distribution. Then run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e .
```

#### Linux

Install Python, its virtual-environment support, and Git using your distribution's
package manager if they are not already available. Confirm that
`python3 --version` reports 3.11 or newer, then run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e .
```

On Debian/Ubuntu, a missing `venv` module is usually supplied by the
`python3-venv` package. LAPD Explorer must be launched from a graphical session;
see [Troubleshooting](#troubleshooting) if Qt reports a missing Linux display
library.

#### Windows (PowerShell or Command Prompt)

Install a current 64-bit Python from
[python.org](https://www.python.org/downloads/windows/) if needed. The Python
installer's `py` launcher makes it easy to select Python 3. Verify that
`py -3 --version` reports 3.11 or newer, then run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e .
```

If the `py` command is unavailable but `python` reports a suitable version,
replace `py -3` in the first command with `python`.

### 3. Start the application

On macOS or Linux:

```sh
.venv/bin/python -m lapd_explorer
```

On Windows:

```powershell
.\.venv\Scripts\python.exe -m lapd_explorer
```

The installed `lapd-explorer` command is also available inside the virtual
environment, but `python -m lapd_explorer` is the most dependable form on every
platform. The application opens with a synthetic two-component plane wave, so
you can explore the controls without an HDF5 file.

### Optional: activate the environment

Activation lets you type `python` and `lapd-explorer` without their full paths:

| Shell | Command |
| --- | --- |
| macOS/Linux (`bash` or `zsh`) | `source .venv/bin/activate` |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |

Run `deactivate` when finished. Activation is a convenience, not an installation
requirement.

## First run

1. Start with the built-in **Demo** data or select **Open HDF5…**.
2. For a BaPSF file, select one to three digitizer channels and, when available,
   a motion control. For a generic file, choose **Raw HDF5**, select numeric
   datasets, and provide the sample interval and signal units.
3. Confirm the inferred dimension mapping. Case and shot counts remain editable.
4. Select **Load acquisition**. Source files are always opened read-only.
5. Choose the displayed quantity and components, set any preprocessing options,
   and select **Apply to original data**.
6. Move through time with the frame control or **Play**. Click a line or plane to
   move the linked spatial cursor.
7. Use **Save image…**, **Export MP4…**, or **Save data…** for output.

The file dialogs remember separate folders for opening data, saving data,
saving images, and exporting movies, including across app restarts. A folder
is remembered after a successful read or write; canceling or a failed operation
keeps the previous location. If a remembered folder is unavailable, the dialog
starts in your home folder.
Real-valued fields accept ordinary decimals or scientific notation such as
`1.4e5`. Discrete indices, sample counts, and axis sizes remain integer-only.

Processing always starts from the imported original, so applying integration or
gain twice does not apply the operation twice. **Reset processing** returns to
the imported values. Changing Light/Dark appearance preserves the data,
processing, component assignments, colormaps, and slice settings.

## Loading BaPSF acquisitions

The importer discovers digitizers, ADC configurations, channels, and motion
controls through `bapsflib`. Multiple channels must have compatible record and
time axes. Use Command-click on macOS or Ctrl-click on Windows/Linux to select
several channels.

For SIS 3302 and SIS 3305 channels, nonblank user-entered **Data type**
descriptions become the imported channel names used in selectors and plots.
The import and preview lists show the description alongside the board/channel
identity. Missing or whitespace-only descriptions retain the board/channel
name; the `C1`, `C2`, `C3` prefixes distinguish selected inputs even when their
descriptions match. Names and descriptions are preserved in saved explorer data.

### Read fewer temporal samples

Before loading, use **Read fewer temporal samples** for either BaPSF or raw
HDF5 data:

- Choose **Sample limits** for zero-based, inclusive original sample indices,
  or **Time limits (s)** for inclusive times relative to the entered time origin.
  Leave the final bound blank for the end of the recording. **All samples**
  retains the complete time range.
- Set **Keep every Nth sample** to an integer factor; one keeps all samples.
  Above one, a **Downsampling** choice appears. The imported time spacing is
  multiplied by N: a 100 MHz acquisition with N=10 becomes 10 MHz, with a
  5 MHz Nyquist limit. Crop and downsampling may be combined.
- **Polyphase FIR resampling** is the default. A low-pass Kaiser FIR (20N+1
  taps, beta=5) suppresses higher frequencies before reducing the sample rate.
  Bounded overlapping disk reads avoid artificial chunk boundaries; only the
  chosen interval edges use zero padding. Substantial filtering batches use
  up to four CPU cores across records, while HDF5 access stays serialized.
  Factors above 50,000 require a smaller factor or another method to bound
  filter memory. The first output is aligned to the first selected sample.
- **Simple decimation** reads every Nth original sample directly from disk,
  without anti-alias filtering. Higher-frequency signals can alias. The last
  retained sample may precede the inclusive end limit.
- **Block averaging** averages consecutive complete blocks of N samples and
  uses their center times. An incomplete trailing block is discarded. Averaging
  has weaker anti-alias suppression than FIR. All methods require at least
  two output samples, and their settings are recorded in saved metadata.
- During import, counts show shots (acquisitions) processed and remaining for
  the current channel, followed by grouping/mapping counts. A shot counts as
  processed once its entire selected trace is read and resampled. The progress
  bar also advances within long traces and across channels; it measures work
  completed, not estimated time remaining. After loading, the main status bar
  shows the total imported acquisitions across locations, cases, and repeats,
  counting each acquisition once regardless of channel count. This original
  count stays visible when processing averages shots.
- **Cancel** stops reading, resampling, and mapping at the next bounded chunk,
  discards partial results, and closes the importer automatically after workers
  release their files and buffers. A native read already in progress finishes
  its current chunk; the rest of the acquisition is not read.
- Choose **Preview one trace / choose limits…** to open a zoom/pan toolbar and
  draggable interval. The initial trace comes from the middle of the selected
  record range. Choose another selected channel, original record index, global
  shot number, or nearest motion coordinate. At a coordinate, an occurrence
  index chooses among repeated records in acquisition order. Blank coordinates
  are ignored; global shots and positions require BaPSF metadata.
- Refine the first/last sample indices or start/end times manually in the
  preview. The entries stay synchronized with the plot. **Use these limits**
  copies the final original indices into the importer, where they may still be
  edited. Zooming reloads only the visible interval; the display contains at
  most 20,000 original points from one trace; import filtering is applied only
  when loading. Display thinning can hide short features,
  so zoom in to inspect them.

After previewing, the importer shows the original and effective rates, Nyquist
limit, and retained samples per trace. Signal-array memory scales approximately
with that retained fraction. Reads use bounded record batches; motion mapping
and later processing can still require additional arrays. Original sample
bounds, thinning factor, and effective rate are saved in the imported metadata.
These controls apply to acquisitions and raw HDF5 imports; opening a saved
processed explorer dataset continues to restore that saved dataset directly.

### Motion mapping

Motion grouping requires a complete rectangular grid with the same number of
records at every position. When a channel is selected, the importer checks the
row count of one channel dataset and uses shot-number and motion metadata to
fill **Number of shots per case**, assuming one case. Automatic estimation reads
no signal samples; an explicitly requested preview reads only one trace.
Without motion mapping, the selected row count is the
shot estimate for a single point. Replace that guess when the acquisition
contains multiple cases.

Within each position, acquisition order is interpreted as either `case, shot`
(shot fastest) or `shot, case` (case fastest). Cases cannot be inferred reliably
from position and global shot number alone. Single case/shot dimensions are
removed; the importer never invents statistical repeats.

Target coordinates usually recover the intended scan grid. Measured coordinates
retain real positioning deviations. Both target and measured per-record
coordinates are preserved in export metadata, along with the source used for
each channel. When measured positions are grouped, choose decimal rounding that
is fine enough for the physical scan spacing.

Missing grid points, unequal repeats, three-dimensional scans, and mismatched
vector shot/time axes are rejected. Select a balanced record range or prepare a
subset and use manual mapping.

### Manual and raw-HDF5 mapping

Add manual axes in slowest-to-fastest acquisition order, excluding time. The
axis-size product must equal the number of records. Start/end values describe
evenly spaced coordinates, and every non-singleton axis must increase.

| Stored layout | Manual axes, in order |
| --- | --- |
| Single trace `(nt,)` | No axes |
| Point with repeats `(nshot, nt)` | `shot` |
| Line without repeats `(nx, nt)` | `x` |
| Line with repeats `(nx, nshot, nt)` | `x, shot` |
| Plane with cases/repeats `(ny, nx, ncase, nshot, nt)` | `y, x, case, shot` |
| Complete plane scanned again for each case | `case, y, x, shot` |

For **Raw HDF5**, the final source axis is time and all leading axes are flattened
before manual mapping. Raw values receive no automatic ADC calibration or global
shot-number association. A raw `(nx, nt)` dataset must therefore be mapped as
`x`, rather than interpreted as repeated point measurements.

## Verified example acquisitions

The source acquisitions are not included in this repository. These settings are
recorded so collaborators with the same September 2026 data can reproduce the
validated imports.

### X-line sample

For `04_bdot_port25_xline 2026-09-02 11.39.39.hdf5`:

1. Select board **3**, channels **6, 7, 8**.
2. Select motion control **bmotion / 3 - <Hermes> p25_C16_Nx91_Lx40cm**.
3. Keep **Motion coordinates** and **Target if available**. Selecting a channel
   should infer **1 case** and **5 shots per case**.
4. Select **Load acquisition**.

The validated result is `(x=91, shot=5, time=6144)`, with x from -20 to +20 cm,
global shots 1–455, and a 10 ns sample interval. The channels correspond to Bx,
By, and Bz in the run notes, but their imported amplitudes are digitizer volts.
Probe/amplifier calibration is required for magnetic-field units; integration
alone produces V·s.

The time origin defaults to the first digitizer sample. Enter another origin in
seconds only when a known plasma-relative trigger offset is available. The
application does not infer timing offsets from free-text notes.

### Plane sample

For `14_bdot_port25_xy_51x51_delta7mm 2026-09-07 10.35.11.hdf5`, select board
**3**, channels **6–8**, motion
**bmotion / 0 - <Hermes> p25_C16_51x51_deta7mm**, and **Target if available**.
The importer should infer **1 case** and **5 shots per case**.

![LAPD Explorer showing the verified plane acquisition in Light mode](docs/sample-plane.png)

*The verified 51 × 51 plane in Light mode, with independent mesh/arrow colors,
linked x/y slices, and their all-time vertical scales.*

The complete result was verified as `(y=51, x=51, shot=5, time=6144)`: 13,005
global shots, 0.7 cm spacing, x/y extents of -17.5 to +17.5 cm, and 10 ns
sampling. Mapped signals and coordinates were compared with direct `bapsflib`
reads. The full GUI import, averaged baseline correction, three-component
magnitude, plane arrows, linked slices, still export, and MP4 export were also
exercised.

The run notes label all three channels “Bx”. The application preserves the
board/channel identifiers and leaves physical component assignment to the user;
the validation does not assume that the notes establish the correct vector
basis.

## Processing and interpretation

Operations run in this order:

```text
baseline → integration → time smoothing → gain → stored-shot average → spatial averaging
```

- Mean removal and linear detrending are applied independently to each trace.
- Integration is cumulative trapezoidal integration over the actual time
  coordinates, with initial value zero.
- Shot averaging never averages cases or spatial positions.
- Magnitude is calculated after preprocessing each component. Averaging
  components before magnitude is not equivalent to averaging per-shot
  magnitudes.
- Nonfinite values propagate through means and integration. Detrending and PSD
  reject nonfinite traces. No uncertainty is invented for hardware-averaged or
  single-shot data.

### Temporal smoothing

Enable **Smooth in time**, then open **Settings…** to choose moving average,
Savitzky–Golay, Gaussian, or zero-phase Butterworth filtering. Butterworth
supports low-pass, high-pass, and band-pass filters. Cutoffs are in Hz and must
be below the Nyquist frequency; a band-pass also requires lower < upper.

Moving/Savitzky–Golay windows and Gaussian sigma are in samples. They operate in
sample-index space on irregular grids. Butterworth sampling frequency is inferred
from uniformly spaced time coordinates. Missing values may propagate or be
interpolated along time; entirely missing traces remain NaN. All selected filter
settings are written to exported processing history.

### Spatial averaging (2D planes)

Enable **Spatial averaging (2D)** in **Preprocessing**, open its **Settings…**,
then click **Apply to original data**. Choose from:

- **Box average**: equal weights in a rectangular neighborhood, with independent
  odd window sizes along each spatial axis.
- **Gaussian average**: distance-weighted smoothing, with independent sigma
  values along each axis and support extending approximately four sigma.
- **Disk average**: equal weights inside a circular neighborhood in grid-index
  space, controlled by its radius.
- **Median filter**: a robust neighborhood median for suppressing isolated
  spikes, with independent odd window sizes. This is not an arithmetic mean.

Sizes are in grid points, not centimeters. Unequal axis spacing makes a disk
elliptical in physical space; on irregular grids physical smoothing widths vary.
Edges can be reflected (default), extended using the nearest edge, or wrapped
periodically. **Propagate missing values** (default) marks neighborhoods containing
NaN or infinity as missing. **Omit missing neighbors** uses finite neighbors only,
renormalizes averaging weights, and can fill gaps; empty neighborhoods stay NaN.

Filtering runs independently for every channel, time, case, and remaining shot,
after shot averaging and before absolute values or vector magnitudes. It preserves
grid coordinates and array shape. Maps, vector arrows, linked slices, cursor
traces, saved images, movies, and exported data all use the processed channels.
Settings and axis order are recorded in exported processing history. The controls
are disabled for point and line data; filtering is off by default. **Reset
processing** restores the loaded data, and repeated Apply operations always start
from those originals.

### Plots and playback

Vector arrows use the first two assigned components in the displayed plane. A
third component contributes only to the magnitude background. Use channels with
matching physical units and calibration. Mesh and arrow colormaps are
independent; colored arrows receive their own scale bar. **Fixed scale across
time** locks both color ranges.

In Vector mode, expand **Arrow style** to change shaft thickness, length
multiplier, grid stride, head width/length, opacity, pivot (middle/tail/tip),
and outline thickness. Shaft and outline thicknesses use points (1/72 inch),
so exported images and movies retain the intended physical size. Head sizes
are multiples of shaft thickness. A stride of **Auto** keeps the current
automatic density; 1 draws every grid point and larger values draw every Nth
point along both axes. These are display settings and do not change the data
or the magnitudes represented by arrow colors. **Reset arrow style** restores
the defaults.

At length multiplier 1, the reference maximum spans 80% of the sampled grid
spacing. **Fixed scale across time** uses the maximum in-plane magnitude over
the selected case and shot for both arrow length and color; otherwise each
frame supplies its reference. Spectral vector plots inherit the main arrow
style when opened and expose the same controls in **Display**. Fixed spectral
animation scales use the reference maximum over the animation. Arrow style
changes redraw cached fields without recalculating spectra.

All color map selectors offer gradient previews and the same 36 palettes, each
with a reversed (`_r`) version: perceptually uniform and other sequential maps
for magnitudes/power, diverging maps for signed fields, cyclic maps for phase,
grayscale maps for printing, and familiar rainbow maps. Hover over a palette
for guidance. These choices are available for the main mesh, vector arrows,
spectral fields and location maps, spectrograms, and Langmuir derived quantities.
Langmuir results have their own **Color map** selector. Image and movie exports
use the selected palette. Color limits and scaling remain independent choices;
use symmetric limits for centered signed data or a full-cycle range for phase.

For 2D planes, **Mesh colormap range** defaults to **Automatic**, preserving
the current scaling method. Choose **Manual** and enter Min/Max in displayed
signal units (decimal or scientific notation) to keep those color bounds fixed
for all times, including movie exports. Manual mesh bounds override the fixed
scale checkbox for the mesh; arrow colors still follow that checkbox. Invalid
bounds retain the previous valid range.

Frame stride changes playback/export sampling only. Processing, cursor traces,
and PSD use the full time resolution. The line position–time preview may
subsample its display to remain responsive.

For planes, each slice has its own vertical-axis control:

- **Auto — all times** uses the finite range across every position and time in
  that slice for the current quantity, case, and shot.
- **Manual** accepts explicit minimum/maximum values in displayed signal units.
- **Interactive** preserves toolbar zoom/pan as time advances; **Home** restores
  the full-time slice range.

Still images and movies preserve the selected slice limits.

## Analysis tools

### Spectral analysis

Assign one or two channels with the existing component selectors, set unused
components to **None**, and open **Spectral Analysis…**. The tool uses the
currently processed dataset and preserves spatial geometry, cases, and stored
shots. It provides Welch auto/cross-power, coherency, coherence, cross-phase,
lag-domain covariance, scalar/vector animations, and single-trace auto/cross-power
spectrograms. Use **Trace** to select spatial coordinates, axis/flat indices, or a
global shot, or click the **Locations** plot. The persistent **Previous/Next**
controls cycle through traces, locations, shots, or cases. **Make spectrogram**
opens a time-frequency view with separate estimation and display options; saved
NPZ files preserve complex cross-power and all trace/settings metadata.

Spectral animations include a **Playback** seek slider beneath the spatial
field plot. It tracks Play/Step and lets you scrub frequency or phase frames
using cached results. Seeking pauses playback; Play resumes from that frame.

See [Spectral Analysis](docs/spectral.md) for the complete workflow,
normalization, sign conventions, shot averaging, and amplitude interpretation.

### Langmuir probe analysis

Import sweep-voltage and sweep-current digitizer channels in volts, then select
**Langmuir…**. This analysis uses the original imported channels independently
of main-browser integration, smoothing, or shot averaging.

1. Assign `V_sweep` and `I_sweep`. Voltage attenuation is a multiplier:
   `V_probe = V_digitizer × V_attenuation`. Current is calculated as
   `(I_digitizer − offset_volts) × I_attenuation / resistance_ohms`, with optional
   sign inversion. Enter collection area in mm².
2. Use **Configure current offset…** to select a zero-current interval or enter a
   known digitizer-voltage offset. Without correction, explicitly confirm that
   current zero was independently calibrated.
3. Select a rising sweep containing enough ion, retarding, and electron branches
   for fitting; exclude the rapid return. Graphical and numeric interval controls
   stay synchronized to recorded samples.
4. Adjust binning, smoothing, fit method, fit quality, and physical-range settings
   as needed.
5. Use **Process This Shot** for a representative result or **Process All Shots**
   for every location/case/repeat. Rejected fits become NaN and are summarized.

Results include Te, ne, plasma potential, and floating potential. Settings are
session-local. Leaving the dialog preserves its state for the current dataset;
loading another dataset discards it. Main-browser data and the original HDF5 file
remain unchanged.

## Export behavior

- **Save image…** captures the complete current visualization, including linked
  slices and time trace.
- **Export MP4…** renders the inclusive first/last frame range at the selected
  stride and FPS. A canceled or failed export does not replace an existing file.
- **Save data…** writes every currently processed channel, case, spatial point,
  and time sample. It does not crop to the cursor or movie frame range. The
  portable HDF5 result can be reopened directly in LAPD Explorer.
- **Run info** displays acquisition details, dimensions, units, metadata, and
  processing history.

## Updating or reinstalling

For a Git clone, pull changes and refresh dependencies with:

```sh
git pull
.venv/bin/python -m pip install -e .
```

On Windows, replace `.venv/bin/python` with
`.\.venv\Scripts\python.exe`. If an installation becomes inconsistent, delete
only the repository's `.venv` directory, recreate it with the platform-specific
commands above, and reinstall.

## Troubleshooting

### `python3`, `py`, or Python 3.11+ is not found

Install a current 64-bit Python, close and reopen the terminal, then recheck the
version. Do not use `sudo pip`; keep the installation inside `.venv`.

### `lapd-explorer` is not recognized

Use the full `python -m lapd_explorer` launch command shown above. It does not
depend on the environment's script directory being on `PATH`.

### PowerShell refuses to run `Activate.ps1`

Activation is optional. Use `.\.venv\Scripts\python.exe` directly; no execution
policy change is needed.

### Qt reports that a Linux platform plugin cannot be initialized

Launch from a graphical desktop, not a text-only SSH session. A minimal Linux
installation may also need its distribution's XCB/EGL runtime libraries. On
Debian/Ubuntu, the commonly missing packages are `libxcb-cursor0`,
`libxkbcommon-x11-0`, and `libegl1`; package names differ on other distributions.

### Dependency installation tries to compile a large package

Confirm that the environment uses a supported 64-bit CPython and that `pip` is
current. Current Python versions normally receive binary wheels for PySide6,
NumPy, SciPy, Matplotlib, Astropy, and h5py.

### A file opens as Raw HDF5 instead of BaPSF data

The file may not contain the LAPD metadata expected by `bapsflib`. Select numeric
datasets and use manual mapping, or verify that the intended acquisition file was
chosen.

## Development

See [Development and validation](docs/development.md) for test installation,
headless commands on every platform, real-sample verification, screenshot
regeneration, and the code map.

API behavior has been checked against `bapsflib` 2026.4.0 and its
[official LAPD documentation](https://bapsflib.readthedocs.io/en/latest/using_lapd/main.html).

## Current scope

Irregular point clouds and volume scans are not supported. Raw import assumes a
final time axis. Calibration, arbitrary per-shot parameter tables, per-channel
time-offset correction, and out-of-core processing remain extension points.

## Credits

This tool was developed at the Basic Plasma Science Facility, which is a
Collaborative Research Facility funded by the U.S. Department of Energy.

LAPD Explorer was vibe coded by Stephen Vincena with the aid of OpenAI Codex
using Astra and Sol models.
