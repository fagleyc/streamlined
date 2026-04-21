# Installation Guide

## Prerequisites

- **Python 3.9 or newer** (3.10 or 3.11 recommended)
- **pip** (included with Python)
- **Git** (for cloning the repository)
- Windows, macOS, or Linux

Verify your Python version:
```bash
python --version
```

## Quick Install (end users)

```bash
# 1. Clone the repository
git clone https://github.com/fagleyc/streamlined.git
cd streamlined

# 2. Create and activate a virtual environment
#    Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

#    Windows (cmd):
python -m venv .venv
.venv\Scripts\activate.bat

#    macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# 3. Install runtime dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the GUI
python streamlined.py
```

## Developer Install (with build tools)

If you plan to build a standalone executable with PyInstaller, install the
dev requirements instead:

```bash
pip install -r requirements-dev.txt
```

## Runtime Dependencies

| Package       | Purpose                                    |
| ------------- | ------------------------------------------ |
| numpy         | Numerical arrays and linear algebra        |
| scipy         | Interpolation, MAT file I/O, curve fitting |
| pandas        | DataFrame operations, CSV and Excel export |
| PyQt6         | GUI framework                              |
| pyqtgraph     | Fast interactive plotting (preferred)      |
| matplotlib    | Fallback plotting backend                  |
| nptdms        | TDMS file reader                           |
| h5py          | HDF5 export                                |
| openpyxl      | Excel writer backend for pandas            |

## Optional: Build a Standalone Executable

With `requirements-dev.txt` installed:

```bash
pyinstaller streamlined.spec
```

The executable is written to `dist/`.

## LabVIEW Integration

The `labview_balance_cal.py` script at the project root can be called from
LabVIEW via the Python Node or System Exec. It only needs the runtime
dependencies listed above (no additional packages required).

From the command line:
```bash
python labview_balance_cal.py CalFiles\200009_26MAR2012.vol --cal_type Linear --output json
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'PyQt6'`**
The virtual environment is not activated. Re-activate it and re-run pip install.

**`Could not find a version that satisfies the requirement PyQt6`**
Your Python is too old (need 3.9+) or is 32-bit. Install 64-bit Python 3.10+.

**The plot window is blank or crashes on start**
pyqtgraph may conflict with certain Qt versions. Try:
```bash
pip install --force-reinstall pyqtgraph
```

**Excel export fails**
Ensure `openpyxl` is installed (it is included in `requirements.txt`).
