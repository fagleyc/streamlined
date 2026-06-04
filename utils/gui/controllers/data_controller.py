"""
Data Controller
===============

Controller for managing data loading, processing, and model updates.
Connects the GUI to the windtunnel processing backend.
"""

import os
import re
import traceback
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QRunnable, QThreadPool

# Import the windtunnel backend
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from utils.windtunnel import DAQ
    from utils.windtunnel.calibration import read_vol_file, read_pcf_file, calc_coeffs
    from utils.windtunnel.data_io import (
        parse_tdms_filename, group_files_by_configuration,
        extract_alpha_beta_from_filename, read_tdms_file, FileInfo
    )
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False

from ..models.data_model import DataModel
from ..models.case import TestCase
from ..models.settings import AppSettings


@dataclass
class FileInfoSimple:
    """Simple file info for when backend is not available."""
    filepath: Path
    configuration: str
    air_state: str
    alpha: float
    beta: float


def parse_filename_simple(filepath: str) -> FileInfoSimple:
    """Parse TDMS filename without backend dependency.

    Expected format: [AirState]_[Configuration]_Alpha_[value]_Beta_[value].tdms
    Example: AirOff_F16check_no_beta_Alpha_-2.0_Beta_0.0.tdms
    """
    filepath = Path(filepath)
    filename = filepath.stem

    # Extract alpha - look for _Alpha_ followed by a number
    alpha_match = re.search(r'_Alpha_(-?\d+\.?\d*)', filename, re.IGNORECASE)
    alpha = float(alpha_match.group(1)) if alpha_match else 0.0

    # Extract beta - look for _Beta_ followed by a number
    beta_match = re.search(r'_Beta_(-?\d+\.?\d*)', filename, re.IGNORECASE)
    beta = float(beta_match.group(1)) if beta_match else 0.0

    # Extract air state - look for AirOn or AirOff at the start
    if filename.lower().startswith('airon'):
        air_state = 'AirOn'
    elif filename.lower().startswith('airoff'):
        air_state = 'AirOff'
    else:
        air_state = 'Unknown'

    # Extract configuration - everything between AirState and _Alpha_
    # Pattern: (AirOn|AirOff)_<configuration>_Alpha_...
    config_match = re.match(
        r'^(?:AirOn|AirOff)_(.+?)_Alpha_',
        filename,
        re.IGNORECASE
    )

    if config_match:
        configuration = config_match.group(1)
    else:
        # Fallback: try to extract without the air state prefix
        alt_match = re.match(r'^(.+?)_Alpha_', filename, re.IGNORECASE)
        if alt_match:
            configuration = alt_match.group(1)
            # Remove AirOn/AirOff if present
            configuration = re.sub(r'^(AirOn|AirOff)_?', '', configuration, flags=re.IGNORECASE)
        else:
            configuration = 'Unknown'

    return FileInfoSimple(
        filepath=filepath,
        configuration=configuration,
        air_state=air_state,
        alpha=alpha,
        beta=beta
    )


def group_files_simple(files: list) -> Dict[str, Dict[str, list]]:
    """Group files by configuration without backend dependency."""
    grouped = {}

    for f in files:
        info = parse_filename_simple(str(f))

        if info.configuration not in grouped:
            grouped[info.configuration] = {'AirOn': [], 'AirOff': [], 'Unknown': []}

        grouped[info.configuration][info.air_state].append(info)

    # Sort files within each group by alpha, then beta
    for config in grouped:
        for air_state in grouped[config]:
            grouped[config][air_state].sort(key=lambda x: (x.alpha, x.beta))

    return grouped


class ProcessingWorker(QRunnable):
    """Worker for background data processing."""

    class Signals(QObject):
        started = pyqtSignal(str)
        progress = pyqtSignal(int, int)
        finished = pyqtSignal(str)
        error = pyqtSignal(str, str)
        case_ready = pyqtSignal(object)  # TestCase

    def __init__(self, directories: List[str], balance_cal, pressure_cal,
                 balance_cal_file: str, pressure_cal_file: str,
                 geometry: dict, settings: dict, recursive: bool = False):
        super().__init__()
        self.signals = self.Signals()
        # Support both single directory (str) and multiple directories (list)
        if isinstance(directories, str):
            self.directories = [directories]
        else:
            self.directories = list(directories)
        self.balance_cal = balance_cal
        self.pressure_cal = pressure_cal
        self.balance_cal_file = balance_cal_file
        self.pressure_cal_file = pressure_cal_file
        self.geometry = geometry
        self.settings = settings
        self.recursive = recursive
        self._cancelled = False

    def run(self):
        """Process data files in background."""
        try:
            self.signals.started.emit("Loading data files...")

            # First pass: count total configurations across all directories
            total_expected = 0
            dir_configs = []  # List of (directory, grouped_files) tuples

            for directory in self.directories:
                data_dir = Path(directory)
                tdms_files = list(data_dir.rglob("*.tdms") if self.recursive else data_dir.glob("*.tdms"))

                if not tdms_files:
                    continue

                # Group files by configuration
                if BACKEND_AVAILABLE:
                    grouped = group_files_by_configuration(tdms_files)
                else:
                    grouped = group_files_simple(tdms_files)

                total_expected += len(grouped)
                dir_configs.append((directory, data_dir.name, grouped))

            if total_expected == 0:
                self.signals.error.emit("No Data Found",
                                        f"No TDMS files found in selected directories")
                return

            self.signals.progress.emit(0, total_expected)

            # Second pass: process each configuration
            processed_count = 0
            dir_names = []
            failed_configs = []

            for directory, dir_name, grouped in dir_configs:
                if self._cancelled:
                    self.signals.finished.emit("Processing cancelled")
                    return

                dir_names.append(dir_name)

                for config_name, air_states in grouped.items():
                    if self._cancelled:
                        self.signals.finished.emit("Processing cancelled")
                        return

                    try:
                        case = self._process_configuration(
                            config_name, air_states, processed_count, directory
                        )
                        if case:
                            self.signals.case_ready.emit(case)
                            processed_count += 1
                    except Exception as e:
                        # Log error but continue processing other configurations
                        traceback.print_exc()
                        failed_configs.append(
                            f"{config_name}: {type(e).__name__}: {e}")

                    self.signals.progress.emit(processed_count, total_expected)

            # Final progress update
            self.signals.progress.emit(total_expected, total_expected)

            # If everything failed, raise as an error so the user sees it
            if processed_count == 0 and failed_configs:
                self.signals.error.emit(
                    "Processing Failed",
                    "All configurations failed to process. First few errors:\n\n"
                    + "\n".join(failed_configs[:5])
                    + ("\n\n..." if len(failed_configs) > 5 else ""))
                return

            # Summary message
            if len(dir_names) == 1:
                msg = (f"Loaded {dir_names[0]}: "
                       f"{processed_count} configuration(s)")
            else:
                msg = (f"Loaded {len(dir_names)} directories: "
                       f"{processed_count} configuration(s)")
            if failed_configs:
                msg += f"  ({len(failed_configs)} failed - see console)"
            self.signals.finished.emit(msg)

        except Exception as e:
            self.signals.error.emit(
                "Processing Error",
                f"{type(e).__name__}: {e}")
            traceback.print_exc()

    def _process_configuration(self, config_name: str,
                                air_states: Dict[str, list],
                                index: int, directory: str = None) -> Optional[TestCase]:
        """Process all files for a single configuration into one case."""
        air_on_files = air_states.get('AirOn', [])
        air_off_files = air_states.get('AirOff', [])

        # Use AirOn files primarily, or AirOff if no AirOn
        primary_files = air_on_files if air_on_files else air_off_files

        if not primary_files:
            return None

        # Get the directory name for the run name
        if directory is None:
            directory = self.directories[0] if self.directories else ""
        run_name = Path(directory).name

        # Create case name from run + configuration
        case_name = f"{run_name}_{config_name}" if config_name != 'Unknown' else run_name

        # Collect all unique alpha/beta combinations
        alpha_beta_pairs = []
        for file_info in primary_files:
            alpha_beta_pairs.append((file_info.alpha, file_info.beta))

        # Get unique alphas and betas
        alphas = sorted(set(ab[0] for ab in alpha_beta_pairs))
        betas = sorted(set(ab[1] for ab in alpha_beta_pairs))

        n_alpha = len(alphas)
        n_beta = len(betas)

        if BACKEND_AVAILABLE and self.balance_cal:
            return self._process_with_backend(
                case_name, config_name, primary_files, air_off_files,
                alphas, betas, n_alpha, n_beta, index, directory
            )
        else:
            return self._create_case_from_files(
                case_name, config_name, primary_files,
                alphas, betas, n_alpha, n_beta, index
            )

    def _process_with_backend(self, case_name: str, config_name: str,
                               air_on_files: list, air_off_files: list,
                               alphas: list, betas: list,
                               n_alpha: int, n_beta: int, index: int,
                               directory: str = None) -> TestCase:
        """Process files using the windtunnel backend."""
        try:
            from utils.windtunnel import DAQ

            # Create DAQ instance
            daq = DAQ()

            # Set facility defaults
            facility = self.settings.get('facility', 'SWT')
            if facility == 'SWT':
                daq.set_swt_defaults()
            elif facility == 'LSWT':
                daq.set_lswt_defaults()

            # Override balance config from user setting
            balance_config = self.settings.get('balance_config', 'Force')
            daq.fac.balance_config = balance_config

            # Load balance calibration if available
            if self.balance_cal:
                daq.cal = self.balance_cal
            elif self.balance_cal_file:
                # Try to load from file
                daq.cal_balance(self.balance_cal_file, self.settings.get('cal_type', 'Linear'))

            # Load pressure calibration if available
            if self.pressure_cal:
                daq.pressure_cal = self.pressure_cal
            elif self.pressure_cal_file:
                daq.cal_instruments(self.pressure_cal_file)

            # Set geometry
            mac = self.geometry.get('mac', 1.0)
            ref_area = self.geometry.get('ref_area', 1.0)
            span = self.geometry.get('span', 1.0)
            mrc = self.geometry.get('mrc', [0.0, 0.0, 0.0])
            units = self.geometry.get('units', 'IPS')

            daq.set_geometry(MAC=mac, S=ref_area, MRC=mrc, units=units,
                             span=span)

            # Load only this configuration's files (not the whole directory)
            if directory is None:
                directory = self.directories[0] if self.directories else ""
            data_dir = str(air_on_files[0].filepath.parent) if air_on_files else directory

            # Sort AirOn files by (alpha, beta) for consistent ordering
            on_sorted = sorted(air_on_files, key=lambda f: (f.alpha, f.beta))
            off_sorted = sorted(air_off_files, key=lambda f: (f.alpha, f.beta))

            for i, on_info in enumerate(on_sorted):
                raw_entry = {'AirOn': {}, 'AirOff': {}}

                raw_on, _ = read_tdms_file(str(on_info.filepath))
                raw_entry['AirOn'] = raw_on.data
                raw_entry['AirOn']['Time'] = raw_on.time

                # Find matching AirOff file by alpha/beta
                matched = False
                if off_sorted:
                    for off_info in off_sorted:
                        if (np.isclose(on_info.alpha, off_info.alpha, atol=0.5) and
                                np.isclose(on_info.beta, off_info.beta, atol=0.5)):
                            raw_off, _ = read_tdms_file(str(off_info.filepath))
                            raw_entry['AirOff'] = raw_off.data
                            raw_entry['AirOff']['Time'] = raw_off.time
                            matched = True
                            break

                    if not matched:
                        # Fallback: use first AirOff as tare
                        raw_off, _ = read_tdms_file(str(off_sorted[0].filepath))
                        raw_entry['AirOff'] = raw_off.data
                        raw_entry['AirOff']['Time'] = raw_off.time

                daq.raw.append(raw_entry)

            # Reduce data
            if daq.cal:
                daq.reduce_datasets()
                daq.reduce_steady_state()

                # Create TestCase from steady-state results
                case = TestCase(
                    id=str(uuid.uuid4())[:8],  # Use unique ID
                    name=case_name,
                    filepath=data_dir
                )

                ss = daq.ss
                case.alphas = ss.alphas
                case.betas = ss.betas
                case.Cl = ss.Cl
                case.Cd = ss.Cd
                case.Cs = ss.Cs
                case.CRoll = ss.CRoll
                case.CPitch = ss.CPitch
                case.CYaw = ss.CYaw

                # Standard deviations
                case.Cl_std = ss.Cl_std
                case.Cd_std = ss.Cd_std
                case.Cs_std = ss.Cs_std
                case.CRoll_std = ss.CRoll_std
                case.CPitch_std = ss.CPitch_std
                case.CYaw_std = ss.CYaw_std

                # Store DAQ reference for later use
                case.daq = daq

                # Transfer tunnel conditions and forces/moments from reduced data
                if daq.red and len(daq.red) > 0:
                    # Collect tunnel conditions from all reduced points
                    Q_list = []
                    U_list = []
                    Re_list = []
                    Mach_list = []
                    rho_list = []
                    T_list = []
                    Ptot_list = []

                    # Collect WRF forces and moments
                    lift_list = []
                    drag_list = []
                    side_list = []
                    roll_list = []
                    pitch_list = []
                    yaw_list = []

                    # Collect balance element forces
                    eN1_list = []
                    eN2_list = []
                    eY1_list = []
                    eY2_list = []
                    eAx_list = []
                    eRoll_list = []

                    for rd in daq.red:
                        if hasattr(rd, 'tunnel') and rd.tunnel is not None:
                            tc = rd.tunnel
                            if len(tc.Q) > 0:
                                Q_list.append(np.mean(tc.Q))
                            if len(tc.U_inf) > 0:
                                U_list.append(np.mean(tc.U_inf))
                            if len(tc.Re) > 0:
                                Re_list.append(np.mean(tc.Re))
                            if len(tc.Mach) > 0:
                                Mach_list.append(np.mean(tc.Mach))
                            if len(tc.rho) > 0:
                                rho_list.append(np.mean(tc.rho))
                            if len(tc.T) > 0:
                                T_list.append(np.mean(tc.T))
                            if hasattr(tc, 'P_tot') and len(tc.P_tot) > 0:
                                # Convert from Pa to psi (internal unit)
                                Ptot_list.append(np.mean(tc.P_tot) / 6894.75729)

                        # Extract WRF forces/moments if available (wrf_aero is the aerodynamic forces)
                        if hasattr(rd, 'wrf_aero') and rd.wrf_aero is not None:
                            wrf = rd.wrf_aero
                            if hasattr(wrf, 'Lift') and len(wrf.Lift) > 0:
                                lift_list.append(np.mean(wrf.Lift))
                            if hasattr(wrf, 'Drag') and len(wrf.Drag) > 0:
                                drag_list.append(np.mean(wrf.Drag))
                            if hasattr(wrf, 'Side') and len(wrf.Side) > 0:
                                side_list.append(np.mean(wrf.Side))
                            if hasattr(wrf, 'Roll') and len(wrf.Roll) > 0:
                                roll_list.append(np.mean(wrf.Roll))
                            if hasattr(wrf, 'Pitch') and len(wrf.Pitch) > 0:
                                pitch_list.append(np.mean(wrf.Pitch))
                            if hasattr(wrf, 'Yaw') and len(wrf.Yaw) > 0:
                                yaw_list.append(np.mean(wrf.Yaw))

                        # Extract balance element forces (air-on minus air-off)
                        if hasattr(rd, 'brf_on') and rd.brf_on is not None:
                            elems_on = rd.brf_on.elements
                            elems_off = (rd.brf_off.elements
                                         if hasattr(rd, 'brf_off')
                                         and rd.brf_off is not None
                                         and rd.brf_off.elements is not None
                                         and len(rd.brf_off.elements) > 0
                                         else None)
                            if (elems_on is not None and len(elems_on) > 0
                                    and elems_on.ndim == 2
                                    and elems_on.shape[1] >= 6):
                                if (elems_off is not None
                                        and elems_off.ndim == 2
                                        and elems_off.shape[1] >= 6):
                                    elems = (elems_on
                                             - np.mean(elems_off, axis=0))
                                else:
                                    elems = elems_on
                                eN1_list.append(np.mean(elems[:, 0]))
                                eN2_list.append(np.mean(elems[:, 1]))
                                eY1_list.append(np.mean(elems[:, 2]))
                                eY2_list.append(np.mean(elems[:, 3]))
                                eAx_list.append(np.mean(elems[:, 4]))
                                eRoll_list.append(np.mean(elems[:, 5]))

                    # Store tunnel conditions as arrays on case
                    if Q_list:
                        case.dynamic_pressures = np.array(Q_list)
                    if U_list:
                        case.velocities = np.array(U_list)
                    if Re_list:
                        case.reynolds = np.array(Re_list)
                    if Mach_list:
                        case.machs = np.array(Mach_list)
                    if rho_list:
                        case.densities = np.array(rho_list)
                    if T_list:
                        case.temperatures = np.array(T_list)
                    if Ptot_list:
                        case.total_pressures = np.array(Ptot_list)

                    # Store WRF forces and moments (in IPS units: lbf, lb-in)
                    if lift_list:
                        case.lift_forces = np.array(lift_list)
                    if drag_list:
                        case.drag_forces = np.array(drag_list)
                    if side_list:
                        case.side_forces = np.array(side_list)
                    if roll_list:
                        case.roll_moments = np.array(roll_list)
                    if pitch_list:
                        case.pitch_moments = np.array(pitch_list)
                    if yaw_list:
                        case.yaw_moments = np.array(yaw_list)

                    # Store balance element forces (in IPS units: lbf)
                    if eN1_list:
                        case.elem_N1 = np.array(eN1_list)
                    if eN2_list:
                        case.elem_N2 = np.array(eN2_list)
                    if eY1_list:
                        case.elem_Y1 = np.array(eY1_list)
                    if eY2_list:
                        case.elem_Y2 = np.array(eY2_list)
                    if eAx_list:
                        case.elem_Ax = np.array(eAx_list)
                    if eRoll_list:
                        case.elem_Roll = np.array(eRoll_list)

                    # Set mean values
                    if Q_list:
                        case.pressure = float(np.mean(Q_list))
                    if U_list:
                        case.velocity = float(np.mean(U_list))
                    if Re_list:
                        case.reynolds_number = float(np.mean(Re_list))
                    if Mach_list:
                        case.mach_number = float(np.mean(Mach_list))
                    if rho_list:
                        case.density = float(np.mean(rho_list))
                    if T_list:
                        case.temperature = float(np.mean(T_list))

                return case
            else:
                # No calibration - fall back to demo mode
                return self._create_case_from_files(
                    case_name, config_name, air_on_files,
                    alphas, betas, n_alpha, n_beta, index
                )

        except Exception as e:
            traceback.print_exc()
            # Fall back to simplified processing
            return self._create_case_from_files(
                case_name, config_name, air_on_files,
                alphas, betas, n_alpha, n_beta, index
            )

    def _create_case_from_files(self, case_name: str, config_name: str,
                                 files: list, alphas: list, betas: list,
                                 n_alpha: int, n_beta: int, index: int) -> TestCase:
        """Create a case from file information (demo mode or simple processing)."""
        case = TestCase(
            id=str(uuid.uuid4())[:8],  # Use unique ID
            name=case_name,
            filepath=str(files[0].filepath.parent) if files else ""
        )

        # Create 2D arrays for alpha/beta grid
        if n_beta == 1:
            # Single beta sweep - 1D arrays
            case.alphas = np.array(alphas).reshape(-1, 1)
            case.betas = np.full_like(case.alphas, betas[0])
        else:
            # Multiple beta sweeps - 2D grid
            case.alphas = np.zeros((n_alpha, n_beta))
            case.betas = np.zeros((n_alpha, n_beta))
            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    case.alphas[i, j] = alpha
                    case.betas[i, j] = beta

        # Generate representative aerodynamic coefficients
        # (In demo mode, use typical values based on alpha)
        alpha_arr = np.array(alphas)

        # CL: Linear with alpha, CL = CL0 + CLa * alpha
        CL0 = 0.1
        CLa = 0.11  # per degree
        Cl_1d = CL0 + CLa * alpha_arr

        # CD: Parabolic drag polar, CD = CD0 + k * CL^2
        CD0 = 0.02
        k = 0.04
        Cd_1d = CD0 + k * Cl_1d**2

        # Cm: Pitch moment, Cm = Cm0 + Cma * alpha
        Cm0 = 0.05
        Cma = -0.01  # per degree (stable)
        CPitch_1d = Cm0 + Cma * alpha_arr

        # Side force and lateral moments (small for symmetric model at beta=0)
        Cs_1d = np.zeros_like(alpha_arr)
        CRoll_1d = np.zeros_like(alpha_arr)
        CYaw_1d = np.zeros_like(alpha_arr)

        # Reshape to match alpha/beta grid
        if n_beta == 1:
            case.Cl = Cl_1d.reshape(-1, 1)
            case.Cd = Cd_1d.reshape(-1, 1)
            case.Cs = Cs_1d.reshape(-1, 1)
            case.CPitch = CPitch_1d.reshape(-1, 1)
            case.CRoll = CRoll_1d.reshape(-1, 1)
            case.CYaw = CYaw_1d.reshape(-1, 1)
        else:
            case.Cl = np.tile(Cl_1d.reshape(-1, 1), (1, n_beta))
            case.Cd = np.tile(Cd_1d.reshape(-1, 1), (1, n_beta))
            case.Cs = np.tile(Cs_1d.reshape(-1, 1), (1, n_beta))
            case.CPitch = np.tile(CPitch_1d.reshape(-1, 1), (1, n_beta))
            case.CRoll = np.tile(CRoll_1d.reshape(-1, 1), (1, n_beta))
            case.CYaw = np.tile(CYaw_1d.reshape(-1, 1), (1, n_beta))

            # Add beta effects
            for j, beta in enumerate(betas):
                # Side force proportional to beta
                case.Cs[:, j] = 0.01 * beta
                # Yaw moment proportional to beta (stable)
                case.CYaw[:, j] = -0.002 * beta
                # Roll moment proportional to beta
                case.CRoll[:, j] = -0.001 * beta

        # Add small random noise for realism
        noise = 0.002
        case.Cl += np.random.normal(0, noise, case.Cl.shape)
        case.Cd += np.random.normal(0, noise/2, case.Cd.shape)
        case.CPitch += np.random.normal(0, noise/2, case.CPitch.shape)

        return case

    def cancel(self):
        """Cancel processing."""
        self._cancelled = True


class DataController(QObject):
    """
    Controller for managing data operations.

    Connects the GUI views to the windtunnel processing backend.
    Handles file loading, calibration, and data processing.
    """

    # Signals
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)
    config_loaded = pyqtSignal(str)  # Config file path

    def __init__(self, model: DataModel, settings: AppSettings):
        super().__init__()
        self.model = model
        self.settings = settings

        self._balance_cal = None
        self._balance_cal_file = None
        self._pressure_cal = None
        self._pressure_cal_file = None
        self._current_worker = None
        self._last_directories: List[str] = []

        self._thread_pool = QThreadPool()
        self._thread_pool.setMaxThreadCount(1)  # Process sequentially

    def load_balance_calibration(self, filepath: str) -> bool:
        """Load balance calibration file (.vol)."""
        if not filepath or not Path(filepath).exists():
            self.error_occurred.emit(
                "Calibration File Not Found",
                f"Balance calibration file does not exist:\n{filepath}\n\n"
                "Choose a valid .vol file.")
            return False

        try:
            if BACKEND_AVAILABLE:
                from utils.windtunnel.calibration import read_vol_file, calc_coeffs
                self._balance_cal = read_vol_file(filepath)
                if not self._balance_cal:
                    self.error_occurred.emit(
                        "Invalid Calibration File",
                        f"Could not parse balance calibration:\n"
                        f"{Path(filepath).name}\n\n"
                        "The file may be corrupt or not in the expected "
                        "(.vol) format.")
                    return False
                cal_type = self.model.cal_type
                self._balance_cal = calc_coeffs(self._balance_cal, cal_type)
            else:
                # Dummy for testing
                self._balance_cal = {'filepath': filepath, 'loaded': True}

            self._balance_cal_file = filepath
            self.model.balance_cal_file = filepath

            # Register as a named calibration
            cal_name = Path(filepath).stem
            cal_entry = {
                'bal_file': filepath,
                'cal_type': self.model.cal_type,
                'balance_config': self.model.balance_config,
                'cal_object': self._balance_cal,
            }
            self.model.add_calibration(cal_name, cal_entry)

            self.status_changed.emit(f"Loaded balance calibration: {Path(filepath).name}")
            return True

        except Exception as e:
            self.error_occurred.emit(
                "Calibration Load Error",
                f"Failed to load balance calibration "
                f"'{Path(filepath).name}':\n\n"
                f"{type(e).__name__}: {e}")
            traceback.print_exc()
            return False

    def add_balance_calibration(self, filepath: str, cal_type: str,
                                balance_config: str,
                                name: str = '') -> bool:
        """Load a balance cal file with specific settings (for per-case use)."""
        if not filepath or not Path(filepath).exists():
            self.error_occurred.emit(
                "Calibration File Not Found",
                f"Balance calibration file does not exist:\n{filepath}")
            return False

        try:
            if BACKEND_AVAILABLE:
                from utils.windtunnel.calibration import read_vol_file, calc_coeffs
                cal_obj = read_vol_file(filepath)
                if not cal_obj:
                    self.error_occurred.emit(
                        "Invalid Calibration File",
                        f"Could not parse '{Path(filepath).name}'.\n\n"
                        "The file may be corrupt or not a .vol file.")
                    return False
                cal_obj = calc_coeffs(cal_obj, cal_type)
            else:
                cal_obj = None

            cal_name = name or Path(filepath).stem
            cal_entry = {
                'bal_file': filepath,
                'cal_type': cal_type,
                'balance_config': balance_config,
                'cal_object': cal_obj,
            }
            self.model.add_calibration(cal_name, cal_entry)
            self.status_changed.emit(
                f"Added calibration '{cal_name}' ({cal_type}, {balance_config})")
            return True

        except Exception as e:
            self.error_occurred.emit(
                "Calibration Load Error",
                f"Failed to load calibration "
                f"'{Path(filepath).name}':\n\n"
                f"{type(e).__name__}: {e}")
            traceback.print_exc()
            return False

    def load_pressure_calibration(self, filepath: str) -> bool:
        """Load pressure calibration file (.pcf)."""
        if not filepath or not Path(filepath).exists():
            self.error_occurred.emit(
                "Calibration File Not Found",
                f"Pressure calibration file does not exist:\n{filepath}\n\n"
                "Choose a valid .PCF file.")
            return False

        try:
            if BACKEND_AVAILABLE:
                from utils.windtunnel.calibration import read_pcf_file
                self._pressure_cal = read_pcf_file(filepath)
                if not self._pressure_cal:
                    self.error_occurred.emit(
                        "Invalid Calibration File",
                        f"Could not parse pressure calibration:\n"
                        f"{Path(filepath).name}\n\n"
                        "The file may be corrupt or not in the expected "
                        "(.PCF) format.")
                    return False
            else:
                # Dummy for testing
                self._pressure_cal = {'filepath': filepath, 'loaded': True}

            self._pressure_cal_file = filepath
            self.model.pressure_cal_file = filepath
            self.status_changed.emit(f"Loaded pressure calibration: {Path(filepath).name}")
            return True

        except Exception as e:
            self.error_occurred.emit(
                "Calibration Load Error",
                f"Failed to load pressure calibration "
                f"'{Path(filepath).name}':\n\n"
                f"{type(e).__name__}: {e}")
            traceback.print_exc()
            return False

    def load_data_directories(self, directories: List[str], recursive: bool = False, clear_existing: bool = True):
        """Load and process data from multiple directories."""
        # ----- Pre-flight checks (show user-friendly dialogs) -----
        if not directories:
            self.error_occurred.emit(
                "No Data Directories",
                "No data directories were selected. Use File > Load Data Directory to choose a folder containing TDMS files.")
            return

        valid_dirs = [d for d in directories if d and Path(d).exists()]
        if not valid_dirs:
            self.error_occurred.emit(
                "Data Directory Not Found",
                "None of the selected data directories exist:\n\n"
                + "\n".join(directories) + "\n\nVerify the path and try again.")
            return

        # Check at least one TDMS file exists in the directories
        any_tdms = False
        for d in valid_dirs:
            dp = Path(d)
            files = (list(dp.rglob("*.tdms")) if recursive
                     else list(dp.glob("*.tdms")))
            if files:
                any_tdms = True
                break
        if not any_tdms:
            search = "recursively" if recursive else "in the top level"
            self.error_occurred.emit(
                "No TDMS Files Found",
                f"No .tdms files were found {search} of:\n\n"
                + "\n".join(valid_dirs)
                + "\n\nIf the files are in subfolders, enable 'Recursive' on the Load Data dialog.")
            return

        # Balance calibration is required to compute forces
        if not self._balance_cal and not self._balance_cal_file:
            self.error_occurred.emit(
                "No Balance Calibration",
                "Load a balance calibration (.vol) file before processing data.\n\n"
                "Use File > Load Balance Calibration or the 'Browse' button in the Calibration panel.")
            return

        # Pressure calibration is required to compute tunnel conditions
        if not self._pressure_cal and not self._pressure_cal_file:
            self.error_occurred.emit(
                "No Pressure Calibration",
                "Load a pressure calibration (.PCF) file before processing data.\n\n"
                "Use File > Load Pressure Calibration or the 'Browse' button in the Calibration panel.")
            return

        # Sanity check on geometry
        default_geo = self.model.get_geometry(self.model.default_geometry)
        mac = default_geo.get('mac', 0.0)
        ref_area = default_geo.get('ref_area', 0.0)
        if mac <= 0 or ref_area <= 0:
            self.error_occurred.emit(
                "Invalid Geometry",
                f"Reference geometry has invalid values:\n"
                f"  MAC = {mac}\n  Reference Area = {ref_area}\n\n"
                "Set positive values via Edit > Model Geometry before processing.")
            return

        # ----- Pre-flight passed; proceed with processing -----

        # Cancel any existing processing
        if self._current_worker:
            self._current_worker.cancel()

        if clear_existing:
            # Clear existing cases before loading
            self.model.clear_all()

        # Use only the valid directories from here on
        directories = valid_dirs

        # Store directories for potential reprocessing
        self._last_directories = list(directories)

        # Create worker with default geometry (individual cases may override)
        default_geo = self.model.get_geometry(self.model.default_geometry)
        geometry = {
            'mac': default_geo.get('mac', 1.0),
            'ref_area': default_geo.get('ref_area', 1.0),
            'span': default_geo.get('span', 1.0),
            'mrc': default_geo.get('mrc', [0.0, 0.0, 0.0]),
            'units': default_geo.get('units', 'IPS'),
            'output_units': self.model.output_units,
        }

        settings = {
            'cal_type': self.model.cal_type,
            'facility': self.model.facility,
            'output_units': self.model.output_units,
            'balance_config': self.model.balance_config,
        }

        worker = ProcessingWorker(
            directories=directories,
            balance_cal=self._balance_cal,
            pressure_cal=self._pressure_cal,
            balance_cal_file=self._balance_cal_file,
            pressure_cal_file=self._pressure_cal_file,
            geometry=geometry,
            settings=settings,
            recursive=recursive
        )

        # Connect signals
        worker.signals.started.connect(self.model.processing_started.emit)
        worker.signals.progress.connect(self.model.processing_progress.emit)
        worker.signals.finished.connect(self.model.processing_finished.emit)
        worker.signals.error.connect(self.model.error_occurred.emit)
        worker.signals.case_ready.connect(self._on_case_ready)

        self._current_worker = worker
        self._thread_pool.start(worker)

    def load_data_directory(self, directory: str):
        """Load and process data from a single directory (legacy support)."""
        self.load_data_directories([directory])

    # ------------------------------------------------------------------
    # Calculator rule serialization + application
    # ------------------------------------------------------------------

    def _serialize_calc_rules(self) -> list:
        try:
            from utils.windtunnel.calculator import rules_to_dicts
            return rules_to_dicts(getattr(self.model, 'calc_rules', []) or [])
        except Exception:
            return []

    def _deserialize_calc_rules(self, raw_list) -> None:
        try:
            from utils.windtunnel.calculator import rules_from_dicts
            self.model.calc_rules = rules_from_dicts(raw_list or [])
        except Exception:
            self.model.calc_rules = []

    def _apply_calc_rules_to_case(self, case: TestCase) -> None:
        """
        Run every active calculator rule on `case` and populate
        both case.custom_vars (means) and case.custom_vars_std (stds)
        with arrays reshaped to the case's alpha/beta grid.  Safe to
        call repeatedly.
        """
        if not case.has_data:
            return
        rules = getattr(self.model, 'calc_rules', None) or []
        if not rules:
            case.custom_vars = {}
            case.custom_vars_std = {}
            return
        try:
            from utils.windtunnel.calculator import (
                apply_rules_to_case, geometry_scalars)
            # Make geometric parameters (MAC, span, ref_area, MRC_x/y/z)
            # available as named scalars in user expressions.
            geo = self.model.get_geometry_for_case(case.id)
            extra = geometry_scalars(geo)
            means, stds = apply_rules_to_case(
                case, rules, extra_scalars=extra)
            case.custom_vars = means
            case.custom_vars_std = stds
        except Exception:
            traceback.print_exc()
            case.custom_vars = {}
            case.custom_vars_std = {}

    def reapply_calc_rules_to_all_cases(self) -> int:
        """Apply current rules to every case; return number reprocessed."""
        n = 0
        for case in list(self.model.cases):
            if not case.has_data:
                continue
            self._apply_calc_rules_to_case(case)
            self.model.case_updated.emit(case.id)
            n += 1
        if n:
            self.model.cases_changed.emit()
        return n

    def _apply_blockage_to_case(self, case: TestCase) -> None:
        """
        Compute blockage-corrected arrays for a case based on the
        current model.blockage_config.  If the method is 'none' or
        the config is missing, the corrected arrays are cleared so the
        uncorrected fields remain primary.

        Safe to call multiple times (idempotent).
        """
        if not case.has_data:
            return
        cfg_dict = getattr(self.model, 'blockage_config', None) or {}
        method = cfg_dict.get('method', 'none')
        if method == 'none':
            case.alphas_corr = np.array([])
            case.Cl_corr = np.array([])
            case.Cd_corr = np.array([])
            case.blockage_epsilon = np.array([])
            case.blockage_method = 'none'
            return

        try:
            from utils.windtunnel.blockage import (
                BlockageConfig, apply_blockage_correction)
        except Exception:
            return

        cfg = BlockageConfig(
            method=method,
            test_section_area_in2=cfg_dict.get(
                'test_section_area_in2', 0.0),
            test_section_width_in=cfg_dict.get(
                'test_section_width_in', 0.0),
            test_section_height_in=cfg_dict.get(
                'test_section_height_in', 0.0),
            lambda_=cfg_dict.get('lambda_', 1.0),
            k=cfg_dict.get('k', 0.333),
            delta=cfg_dict.get('delta', 0.141),
            sigma=cfg_dict.get('sigma', 0.011),
            frontal_area_alpha_low_in2=cfg_dict.get(
                'frontal_area_alpha_low_in2', 0.0),
            frontal_area_alpha_low_deg=cfg_dict.get(
                'frontal_area_alpha_low_deg', 0.0),
            frontal_area_alpha_high_in2=cfg_dict.get(
                'frontal_area_alpha_high_in2', 0.0),
            frontal_area_alpha_high_deg=cfg_dict.get(
                'frontal_area_alpha_high_deg', 0.0),
            reference_area_in2=cfg_dict.get(
                'reference_area_in2',
                self.model.get_geometry(
                    self.model.default_geometry).get('ref_area', 1.0)),
        )

        # Flatten arrays for the correction (works for 1D and 2D)
        alphas_flat = np.asarray(case.alphas, dtype=float).ravel()
        Cl_flat = np.asarray(case.Cl, dtype=float).ravel()
        Cd_flat = np.asarray(case.Cd, dtype=float).ravel()
        try:
            result = apply_blockage_correction(
                alphas_flat, Cl_flat, Cd_flat, cfg)
        except Exception:
            return

        # Reshape back to original shape
        orig_shape = case.alphas.shape
        case.alphas_corr = result.alpha_corrected_deg.reshape(orig_shape)
        case.Cl_corr = result.Cl_corrected.reshape(orig_shape)
        case.Cd_corr = result.Cd_corrected.reshape(orig_shape)
        case.blockage_epsilon = result.epsilon.reshape(orig_shape)
        case.blockage_method = result.method

    def _on_case_ready(self, case: TestCase):
        """Handle a processed case."""
        # Apply blockage corrections (if enabled) before adding to model
        self._apply_blockage_to_case(case)
        # Evaluate any active custom calculator rules
        self._apply_calc_rules_to_case(case)
        self.model.add_case(case)

    def process_data(self):
        """Reprocess all loaded data with current settings."""
        # Use the last loaded directories if available
        if self._last_directories:
            valid_dirs = [d for d in self._last_directories if Path(d).exists()]
            if valid_dirs:
                self.load_data_directories(valid_dirs)
                return

        # Fall back to the single directory from settings
        last_dir = self.settings.last_data_directory
        if not last_dir or not Path(last_dir).exists():
            self.error_occurred.emit("No Data", "No data directory loaded. Please load a data directory first.")
            return

        # Reprocess by reloading the directory with updated settings
        self.load_data_directories([last_dir])

    def reprocess_case(self, case_id: str):
        """Re-reduce a single case with its assigned geometry and calibration."""
        case = self.model.cases.get(case_id)
        if not case:
            self.error_occurred.emit(
                "Case Not Found",
                f"Could not find case with id '{case_id}'. "
                "It may have been removed.")
            return
        if not case.daq:
            self.error_occurred.emit(
                "Cannot Reprocess",
                f"Case '{case.name}' has no raw DAQ data attached.\n\n"
                "Reload the data directory (File > Load Data Directory) "
                "to attach raw data so cases can be re-reduced.")
            return

        try:
            daq = case.daq
            geo = self.model.get_geometry_for_case(case_id)

            # Clear existing geometry and re-set with the new one
            daq.geo.clear()
            daq.set_geometry(
                MAC=geo.get('mac', 1.0),
                S=geo.get('ref_area', 1.0),
                MRC=geo.get('mrc', [0.0, 0.0, 0.0]),
                units=geo.get('units', 'IPS'),
                span=geo.get('span', 1.0)
            )

            # Apply per-case calibration if assigned (includes its own
            # cal_type and balance_config)
            cal_name = self.model.get_calibration_name_for_case(case_id)
            if cal_name:
                cal_entry = self.model.get_calibration(cal_name)
                cal_obj = cal_entry.get('cal_object')
                if cal_obj is not None:
                    daq.cal = cal_obj
                # Use per-calibration balance_config if set
                bal_cfg = cal_entry.get('balance_config')
                if bal_cfg:
                    daq.fac.balance_config = bal_cfg
                else:
                    daq.fac.balance_config = self.model.balance_config
            else:
                daq.fac.balance_config = self.model.balance_config
            daq.reduce_datasets()
            daq.reduce_steady_state()

            ss = daq.ss
            case.alphas = ss.alphas
            case.betas = ss.betas
            case.Cl = ss.Cl
            case.Cd = ss.Cd
            case.Cs = ss.Cs
            case.CRoll = ss.CRoll
            case.CPitch = ss.CPitch
            case.CYaw = ss.CYaw
            case.Cl_std = ss.Cl_std
            case.Cd_std = ss.Cd_std
            case.Cs_std = ss.Cs_std
            case.CRoll_std = ss.CRoll_std
            case.CPitch_std = ss.CPitch_std
            case.CYaw_std = ss.CYaw_std

            # Re-apply blockage correction (if any) with refreshed data
            self._apply_blockage_to_case(case)
            # Re-evaluate any custom calculator rules
            self._apply_calc_rules_to_case(case)

            self.model.case_updated.emit(case_id)
            self.model.cases_changed.emit()
            geo_label = case.geometry_name
            cal_label = case.calibration_name
            msg = f"Reprocessed '{case.name}' with geometry '{geo_label}'"
            if cal_label:
                msg += f", calibration '{cal_label}'"
            self.status_changed.emit(msg)

        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(
                "Reprocess Error",
                f"Failed to reprocess case '{case.name}':\n{str(e)}")

    def remove_case(self, case_id: str):
        """Remove a test case by ID."""
        self.model.remove_case(case_id)
        self.status_changed.emit(f"Case removed")

    def append_data_directory(self, directory: str):
        """Load and append data from a directory without clearing existing cases."""
        # Use load_data_directories with clear_existing=False
        self.load_data_directories([directory], clear_existing=False)

    def export_data(self, filepath: str, format: str = 'csv'):
        """Export processed data."""
        try:
            import pandas as pd

            if format == 'excel':
                # Export each case to a separate sheet
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    used_names = {}
                    for case in self.model.cases:
                        if not case.has_data:
                            continue

                        alphas = case.alphas.flatten()
                        betas = case.betas.flatten()
                        sort_order = np.lexsort((alphas, np.round(betas)))

                        case_data = []
                        for i in sort_order:
                            row = {
                                'Alpha': case.alphas.flatten()[i],
                                'Beta': case.betas.flatten()[i],
                                'CL': case.Cl.flatten()[i],
                                'CD': case.Cd.flatten()[i],
                                'CY': case.Cs.flatten()[i],
                                'Cl': case.CRoll.flatten()[i],
                                'Cm': case.CPitch.flatten()[i],
                                'Cn': case.CYaw.flatten()[i],
                            }

                            machs_flat = case.machs.flatten() if case.machs is not None else np.array([])
                            if i < len(machs_flat):
                                row['Mach'] = machs_flat[i]
                            reynolds_flat = case.reynolds.flatten() if case.reynolds is not None else np.array([])
                            if i < len(reynolds_flat):
                                row['Re'] = reynolds_flat[i]

                            case_data.append(row)

                        df = pd.DataFrame(case_data)
                        # Sheet name must be <= 31 characters and valid
                        sheet_name = case.name[:31].replace('/', '_').replace('\\', '_')
                        sheet_name = sheet_name.replace('[', '(').replace(']', ')')
                        sheet_name = sheet_name.replace('*', '').replace('?', '')
                        sheet_name = sheet_name.replace(':', '-')

                        # Deduplicate: append counter if name already used
                        if sheet_name in used_names:
                            used_names[sheet_name] += 1
                            suffix = f"_{used_names[sheet_name]}"
                            sheet_name = sheet_name[:31 - len(suffix)] + suffix
                        else:
                            used_names[sheet_name] = 0

                        # Build header summary
                        header = self._build_case_header(
                            case, self.model.output_units)
                        startrow = len(header) + 1  # +1 blank row
                        df.to_excel(writer, sheet_name=sheet_name,
                                    index=False, startrow=startrow)

                        # Write header info into the sheet
                        ws = writer.sheets[sheet_name]
                        for r, (key, val) in enumerate(header, start=1):
                            ws.cell(row=r, column=1, value=key)
                            ws.cell(row=r, column=2, value=val)

            else:  # CSV format - all cases in one file
                all_data = []
                for case in self.model.cases:
                    if not case.has_data:
                        continue

                    alphas = case.alphas.flatten()
                    betas = case.betas.flatten()
                    sort_order = np.lexsort((alphas, np.round(betas)))

                    for i in sort_order:
                        row = {
                            'Case': case.name,
                            'Alpha': case.alphas.flatten()[i],
                            'Beta': case.betas.flatten()[i],
                            'CL': case.Cl.flatten()[i],
                            'CD': case.Cd.flatten()[i],
                            'CY': case.Cs.flatten()[i],
                            'Cl': case.CRoll.flatten()[i],
                            'Cm': case.CPitch.flatten()[i],
                            'Cn': case.CYaw.flatten()[i],
                        }

                        machs_flat = case.machs.flatten() if case.machs is not None else np.array([])
                        if i < len(machs_flat):
                            row['Mach'] = machs_flat[i]
                        reynolds_flat = case.reynolds.flatten() if case.reynolds is not None else np.array([])
                        if i < len(reynolds_flat):
                            row['Re'] = reynolds_flat[i]

                        all_data.append(row)

                df = pd.DataFrame(all_data)
                df.to_csv(filepath, index=False)

            self.status_changed.emit(f"Exported data to {Path(filepath).name}")

        except Exception as e:
            self.error_occurred.emit("Export Error", f"Failed to export data:\n{str(e)}")

    @staticmethod
    def _build_case_header(case, output_units: str = 'IPS') -> list:
        """Build header key-value pairs summarizing a case for Excel export."""
        header = [("Case Name", case.name)]

        # Set up converter and unit labels
        converter = None
        labels = None
        try:
            from utils.windtunnel.units import (
                UnitSystem, UnitConverter, UNIT_LABELS)
            converter = UnitConverter(UnitSystem[output_units])
            labels = UNIT_LABELS[UnitSystem[output_units]]
        except Exception:
            pass

        if case.has_data:
            alphas = case.alphas.flatten()
            header.append(("Alpha Range",
                           f"{np.min(alphas):.1f} to {np.max(alphas):.1f} deg"))
            betas = sorted(set(round(float(b), 1) for b in case.betas.flatten()))
            header.append(("Beta Values",
                           ", ".join(f"{b:.1f}" for b in betas) + " deg"))
            header.append(("Data Points", str(case.n_points)))

        if case.mach_number is not None:
            header.append(("Mach", f"{case.mach_number:.4f}"))
        elif len(case.machs) > 0:
            header.append(("Mach", f"{np.mean(case.machs):.4f}"))

        if case.reynolds_number is not None:
            header.append(("Reynolds Number", f"{case.reynolds_number:.2e}"))
        elif len(case.reynolds) > 0:
            header.append(("Reynolds Number", f"{np.mean(case.reynolds):.2e}"))

        q_val = None
        if case.pressure is not None:
            q_val = case.pressure
        elif len(case.dynamic_pressures) > 0:
            q_val = float(np.mean(case.dynamic_pressures))
        if q_val is not None:
            if converter:
                q_val = converter.convert_pressure(q_val)
            unit = f" [{labels.pressure}]" if labels else ""
            header.append((f"Dynamic Pressure (Q){unit}", f"{q_val:.4f}"))

        if len(getattr(case, 'total_pressures', [])) > 0:
            ptot_val = float(np.mean(case.total_pressures))
            if converter:
                ptot_val = converter.convert_pressure(ptot_val)
            unit = f" [{labels.pressure}]" if labels else ""
            header.append((f"Total Pressure (P_tot){unit}", f"{ptot_val:.4f}"))

        u_val = None
        if case.velocity is not None:
            u_val = case.velocity
        elif len(case.velocities) > 0:
            u_val = float(np.mean(case.velocities))
        if u_val is not None:
            if converter:
                u_val = converter.convert_velocity(u_val)
            unit = f" [{labels.velocity}]" if labels else ""
            header.append((f"Velocity (U_inf){unit}", f"{u_val:.2f}"))

        rho_val = None
        if case.density is not None:
            rho_val = case.density
        elif len(case.densities) > 0:
            rho_val = float(np.mean(case.densities))
        if rho_val is not None:
            if converter:
                rho_val = converter.convert_density(rho_val)
            unit = f" [{labels.density}]" if labels else ""
            header.append((f"Density (rho){unit}", f"{rho_val:.6f}"))

        t_val = None
        if case.temperature is not None:
            t_val = case.temperature
        elif len(case.temperatures) > 0:
            t_val = float(np.mean(case.temperatures))
        if t_val is not None:
            if converter:
                t_val = converter.convert_temperature(t_val)
            unit = f" [{labels.temperature}]" if labels else ""
            header.append((f"Temperature{unit}", f"{t_val:.1f}"))

        for key, val in case.metadata.items():
            header.append((str(key), str(val)))

        return header

    def get_backend_status(self) -> str:
        """Get status of backend availability."""
        if BACKEND_AVAILABLE:
            return "Backend: windtunnel package loaded"
        else:
            return "Backend: Demo mode (windtunnel package not found)"

    @staticmethod
    def _make_relative_path(file_path: Optional[str], config_dir: Path) -> Optional[str]:
        """Convert an absolute path to a relative path based on config file directory."""
        if not file_path:
            return None
        try:
            rel = os.path.relpath(file_path, config_dir)
            return rel.replace('\\', '/')
        except ValueError:
            # Different drive on Windows - keep absolute
            return file_path.replace('\\', '/')

    @staticmethod
    def _resolve_cal_path(stored_path: Optional[str], config_dir: Path) -> Optional[str]:
        """Resolve a calibration path from config, trying relative then absolute."""
        if not stored_path:
            return None
        # Try as relative to config directory first
        candidate = config_dir / stored_path
        if candidate.exists():
            return str(candidate.resolve())
        # Try as absolute path
        abs_path = Path(stored_path)
        if abs_path.exists():
            return str(abs_path)
        return None

    def save_configuration(self, filepath: str) -> bool:
        """
        Save current configuration to a JSON file.

        Calibration file paths are stored relative to the config file
        directory for portability across machines.

        Parameters
        ----------
        filepath : str
            Path to save configuration file

        Returns
        -------
        bool
            True if successful
        """
        import json

        try:
            config_dir = Path(filepath).parent

            config = {
                'version': '1.1',
                'calibration': {
                    'balance_file': self._make_relative_path(self._balance_cal_file, config_dir),
                    'pressure_file': self._make_relative_path(self._pressure_cal_file, config_dir),
                    'cal_type': self.model.cal_type,
                },
                'geometries': self.model.geometries,
                'default_geometry': self.model.default_geometry,
                'case_geometry_map': self.model.case_geometry_map,
                'calibrations': {
                    name: {
                        'bal_file': self._make_relative_path(
                            entry.get('bal_file'), config_dir),
                        'cal_type': entry.get('cal_type', 'Cubic'),
                        'balance_config': entry.get('balance_config', 'Force'),
                    }
                    for name, entry in self.model.calibrations.items()
                },
                'default_calibration': self.model.default_calibration,
                'case_calibration_map': self.model.case_calibration_map,
                'output_units': self.model.output_units,
                'facility': {
                    'name': self.model.facility,
                    'balance_config': self.model.balance_config,
                    'pdiff_channel': self.model.pdiff_channel,
                    'p0_channel': self.model.p0_channel,
                },
                'blockage': getattr(
                    self.model, 'blockage_config', {'method': 'none'}),
                'calc_rules': self._serialize_calc_rules(),
                'directories': {
                    'last_data': self.settings.last_data_directory,
                    'last_calibration': self.settings.last_calibration_directory,
                }
            }

            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)

            self.status_changed.emit(f"Configuration saved: {Path(filepath).name}")
            return True

        except Exception as e:
            self.error_occurred.emit(
                "Save Configuration Failed",
                f"Could not save configuration to:\n{filepath}\n\n"
                f"{type(e).__name__}: {e}\n\n"
                "Check that the destination is writable and not "
                "open in another program.")
            traceback.print_exc()
            return False

    def load_configuration(self, filepath: str) -> bool:
        """
        Load configuration from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to configuration file

        Returns
        -------
        bool
            True if successful
        """
        import json

        if not filepath or not Path(filepath).exists():
            self.error_occurred.emit(
                "Configuration Not Found",
                f"Configuration file does not exist:\n{filepath}")
            return False

        try:
            with open(filepath, 'r') as f:
                try:
                    config = json.load(f)
                except json.JSONDecodeError as je:
                    self.error_occurred.emit(
                        "Invalid Configuration File",
                        f"'{Path(filepath).name}' is not valid JSON:\n\n"
                        f"{je}\n\n"
                        "The file may be corrupt or from a different "
                        "application.")
                    return False

            # Load calibration files (resolve relative to config directory)
            config_dir = Path(filepath).parent
            cal_config = config.get('calibration', {})

            balance_file = self._resolve_cal_path(cal_config.get('balance_file'), config_dir)
            if balance_file:
                self.load_balance_calibration(balance_file)

            pressure_file = self._resolve_cal_path(cal_config.get('pressure_file'), config_dir)
            if pressure_file:
                self.load_pressure_calibration(pressure_file)

            cal_type = cal_config.get('cal_type', 'Linear')
            self.model.cal_type = cal_type

            # Load geometry — support both old (single) and new (multi) formats
            if 'geometries' in config:
                # New multi-geometry format (v1.1+)
                self.model.geometries = config['geometries']
                self.model.default_geometry = config.get('default_geometry', 'Default')
                self.model.case_geometry_map = config.get('case_geometry_map', {})
                output_units = config.get('output_units', 'IPS')
            else:
                # Old single-geometry format (v1.0) — convert to multi
                geo_config = config.get('geometry', {})
                self.model.geometries = {
                    'Default': {
                        'mac': geo_config.get('mac', 1.0),
                        'ref_area': geo_config.get('ref_area', 1.0),
                        'span': geo_config.get('span', 1.0),
                        'mrc': geo_config.get('mrc', [0.0, 0.0, 0.0]),
                        'units': geo_config.get('units', 'IPS'),
                    }
                }
                self.model.default_geometry = 'Default'
                self.model.case_geometry_map = {}
                output_units = geo_config.get('output_units', 'IPS')
            self.model.set_output_units(output_units)

            # Load multi-calibration definitions
            if 'calibrations' in config:
                saved_cals = config['calibrations']
                self.model.calibrations = {}
                for cal_name, cal_info in saved_cals.items():
                    bal_path = self._resolve_cal_path(
                        cal_info.get('bal_file'), config_dir)
                    cal_obj = None
                    if bal_path:
                        try:
                            from utils.windtunnel.calibration import (
                                read_vol_file, calc_coeffs)
                            cal_obj = read_vol_file(bal_path)
                            ct = cal_info.get('cal_type', 'Cubic')
                            cal_obj = calc_coeffs(cal_obj, ct)
                        except Exception:
                            pass
                    self.model.calibrations[cal_name] = {
                        'bal_file': bal_path or '',
                        'cal_type': cal_info.get('cal_type', 'Cubic'),
                        'balance_config': cal_info.get('balance_config', 'Force'),
                        'cal_object': cal_obj,
                    }
                self.model.default_calibration = config.get(
                    'default_calibration', '')
                if (not self.model.default_calibration
                        and self.model.calibrations):
                    self.model.default_calibration = next(
                        iter(self.model.calibrations))
                self.model.case_calibration_map = config.get(
                    'case_calibration_map', {})

            # Load facility settings
            fac_config = config.get('facility', {})
            self.model.facility = fac_config.get('name', 'SWT')
            self.model.balance_config = fac_config.get('balance_config', 'Force')
            self.model.pdiff_channel = fac_config.get('pdiff_channel', '220')
            self.model.p0_channel = fac_config.get('p0_channel', '690')

            # Tunnel blockage / wall-effect configuration
            self.model.blockage_config = config.get(
                'blockage', {'method': 'none'})

            # Calculator rules
            self._deserialize_calc_rules(config.get('calc_rules', []))

            # Load directories
            dir_config = config.get('directories', {})
            if dir_config.get('last_data'):
                self.settings.last_data_directory = dir_config['last_data']
            if dir_config.get('last_calibration'):
                self.settings.last_calibration_directory = dir_config['last_calibration']

            self.config_loaded.emit(filepath)
            self.status_changed.emit(f"Configuration loaded: {Path(filepath).name}")
            return True

        except Exception as e:
            self.error_occurred.emit(
                "Load Configuration Failed",
                f"Could not load configuration from:\n{filepath}\n\n"
                f"{type(e).__name__}: {e}\n\n"
                "The file may be corrupt, from an incompatible version, "
                "or reference paths that no longer exist.")
            traceback.print_exc()
            return False

    def get_current_config(self) -> dict:
        """Get current configuration as a dictionary."""
        return {
            'balance_file': self._balance_cal_file,
            'pressure_file': self._pressure_cal_file,
            'cal_type': self.model.cal_type,
            'mac': self.model.mac,
            'ref_area': self.model.ref_area,
            'mrc': self.model.mrc,
            'units': self.model.units,
            'output_units': self.model.output_units,
            'facility': self.model.facility,
        }
