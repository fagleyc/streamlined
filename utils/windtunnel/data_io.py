"""
Data I/O Module
===============

Functions for reading wind tunnel data files (TDMS format)
and exporting processed data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from scipy.interpolate import interp1d

try:
    from nptdms import TdmsFile
    TDMS_AVAILABLE = True
except ImportError:
    TDMS_AVAILABLE = False


@dataclass
class RawData:
    """Container for raw wind tunnel data from a single file."""
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    data: Dict[str, np.ndarray] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    filename: str = ""


def read_tdms_file(filepath: str) -> Tuple[RawData, Dict[str, Any]]:
    """
    Read a TDMS file and return data resampled to a common time base.

    Parameters
    ----------
    filepath : str
        Path to the TDMS file

    Returns
    -------
    tuple
        (RawData, properties) where RawData contains all channels
        resampled to a common time base

    Notes
    -----
    TDMS (Technical Data Management Streaming) files are a binary format
    commonly used with LabVIEW and other NI data acquisition systems.
    This function requires the nptdms library.
    """
    if not TDMS_AVAILABLE:
        raise ImportError(
            "nptdms library is required to read TDMS files. "
            "Install with: pip install nptdms"
        )

    filepath = Path(filepath)
    if not filepath.suffix.lower() == '.tdms':
        filepath = filepath.with_suffix('.tdms')

    raw = RawData(filename=str(filepath))
    properties = {}

    with TdmsFile.open(str(filepath)) as tdms_file:
        # Collect all channels and their time vectors
        channels = {}
        time_vectors = {}

        for group in tdms_file.groups():
            group_name = group.name
            if group_name == 'Time':
                continue

            for channel in group.channels():
                channel_name = channel.name.replace(' ', '_')
                data = channel[:]

                # Get time vector from channel properties
                props = channel.properties
                if 'wf_increment' in props and 'wf_samples' in props:
                    dt = props['wf_increment']
                    n_samples = props['wf_samples']
                    time = np.arange(n_samples) * dt
                else:
                    # Try to get from data length
                    time = np.arange(len(data))

                channels[channel_name] = {
                    'data': np.array(data),
                    'time': time,
                    'group': group_name.replace(' ', '_')
                }
                time_vectors[channel_name] = time

        # Find the smallest time step for resampling
        if time_vectors:
            dt_values = []
            for name, time in time_vectors.items():
                if len(time) > 1:
                    dt_values.append(time[1] - time[0])

            if dt_values:
                min_dt = min(dt_values)

                # Find the reference channel (smallest dt)
                ref_channel = None
                ref_time = None
                for name, ch in channels.items():
                    if len(ch['time']) > 1:
                        dt = ch['time'][1] - ch['time'][0]
                        if np.isclose(dt, min_dt):
                            ref_channel = name
                            ref_time = ch['time']
                            break

                if ref_time is not None:
                    raw.time = ref_time

                    # Resample all channels to the reference time
                    for name, ch in channels.items():
                        ch_dt = ch['time'][1] - ch['time'][0] if len(ch['time']) > 1 else min_dt

                        if not np.isclose(ch_dt, min_dt):
                            # Need to interpolate
                            interp_func = interp1d(
                                ch['time'], ch['data'],
                                kind='cubic',
                                bounds_error=False,
                                fill_value='extrapolate'
                            )
                            raw.data[name] = interp_func(ref_time)
                        else:
                            # Same time base, just copy
                            raw.data[name] = ch['data'][:len(ref_time)]

        # Extract properties (Alpha, Beta, etc.)
        property_types = ['Stiffness', 'Damping', 'Mass', 'L1', 'L2', 'L3', 'L4',
                          'Alpha', 'Beta', 'Plunge', 'Roll']

        for group in tdms_file.groups():
            for prop_name, prop_value in group.properties.items():
                for ptype in property_types:
                    if ptype in prop_name:
                        properties[ptype] = prop_value

        # Ensure Alpha and Beta are in raw.data
        # They might be channels (time-series) or properties (single values)
        n_samples = len(raw.time) if len(raw.time) > 0 else 1

        # Check if Alpha is a channel, if not try to get from properties
        if 'Alpha' not in raw.data:
            if 'Alpha' in properties:
                raw.data['Alpha'] = np.full(n_samples, float(properties['Alpha']))
            else:
                # Try to extract from filename
                alpha, _ = extract_alpha_beta_from_filename(str(filepath))
                raw.data['Alpha'] = np.full(n_samples, alpha)

        # Check if Beta is a channel, if not try to get from properties
        if 'Beta' not in raw.data:
            if 'Beta' in properties:
                raw.data['Beta'] = np.full(n_samples, float(properties['Beta']))
            else:
                # Try to extract from filename
                _, beta = extract_alpha_beta_from_filename(str(filepath))
                raw.data['Beta'] = np.full(n_samples, beta)

    return raw, properties


def read_tdms_simple(filepath: str) -> Dict[str, np.ndarray]:
    """
    Simple TDMS reader that returns a dictionary of arrays.

    Parameters
    ----------
    filepath : str
        Path to the TDMS file

    Returns
    -------
    dict
        Dictionary mapping channel names to numpy arrays
    """
    raw, _ = read_tdms_file(filepath)

    result = {'Time': raw.time}
    result.update(raw.data)

    return result


def export_to_csv(data: Dict[str, np.ndarray], filepath: str,
                  index_col: str = 'Alpha') -> None:
    """
    Export processed data to CSV format.

    Parameters
    ----------
    data : dict
        Dictionary of arrays to export
    filepath : str
        Output file path
    index_col : str
        Column to use as index
    """
    df = pd.DataFrame(data)
    if index_col in df.columns:
        df = df.set_index(index_col)
    df.to_csv(filepath)


def export_to_excel(data: Dict[str, np.ndarray], filepath: str,
                    sheet_name: str = 'Data') -> None:
    """
    Export processed data to Excel format.

    Parameters
    ----------
    data : dict
        Dictionary of arrays to export
    filepath : str
        Output file path
    sheet_name : str
        Name of the Excel sheet
    """
    df = pd.DataFrame(data)
    df.to_excel(filepath, sheet_name=sheet_name, index=False)


def find_data_files(directory: str, pattern: str = '*.tdms',
                    recursive: bool = True) -> list:
    """
    Find data files matching a pattern in a directory.

    Parameters
    ----------
    directory : str
        Directory to search
    pattern : str
        Glob pattern to match files
    recursive : bool
        Whether to search recursively

    Returns
    -------
    list
        List of Path objects for matching files
    """
    directory = Path(directory)

    if recursive:
        files = list(directory.rglob(pattern))
    else:
        files = list(directory.glob(pattern))

    return sorted(files, key=lambda x: x.stat().st_mtime)


def classify_files_by_condition(files: list) -> Dict[str, list]:
    """
    Classify data files by test condition (AirOn/AirOff).

    Parameters
    ----------
    files : list
        List of file paths

    Returns
    -------
    dict
        Dictionary with 'AirOn' and 'AirOff' keys
    """
    classified = {'AirOn': [], 'AirOff': []}

    for f in files:
        fname = str(f).lower()
        if 'airon' in fname:
            classified['AirOn'].append(f)
        elif 'airoff' in fname:
            classified['AirOff'].append(f)

    return classified


def extract_alpha_beta_from_filename(filepath: str) -> Tuple[float, float]:
    """
    Extract alpha and beta values from a filename.

    Expected format: ..._Alpha_X.X_Beta_Y.Y.tdms

    Parameters
    ----------
    filepath : str
        File path to parse

    Returns
    -------
    tuple
        (alpha, beta) values
    """
    import re

    filename = Path(filepath).stem

    # Try to extract Alpha value
    alpha_match = re.search(r'Alpha[_\s]*(-?\d+\.?\d*)', filename, re.IGNORECASE)
    alpha = float(alpha_match.group(1)) if alpha_match else 0.0

    # Try to extract Beta value
    beta_match = re.search(r'Beta[_\s]*(-?\d+\.?\d*)', filename, re.IGNORECASE)
    beta = float(beta_match.group(1)) if beta_match else 0.0

    return alpha, beta


def extract_configuration_from_filename(filepath: str) -> str:
    """
    Extract configuration name from a filename.

    Expected format: [AirState]_[Configuration]_Alpha_X.X_Beta_Y.Y.tdms
    Example: AirOff_F16check_no_beta_Alpha_-2.0_Beta_0.0.tdms -> F16check_no_beta

    Parameters
    ----------
    filepath : str
        File path to parse

    Returns
    -------
    str
        Configuration name (e.g., 'F16check_no_beta')
    """
    import re

    filename = Path(filepath).stem

    # Extract configuration - everything between AirState and _Alpha_
    # Pattern: (AirOn|AirOff)_<configuration>_Alpha_...
    config_match = re.match(
        r'^(?:AirOn|AirOff)_(.+?)_Alpha_',
        filename,
        re.IGNORECASE
    )

    if config_match:
        return config_match.group(1)

    # Fallback: try to extract without the air state prefix
    alt_match = re.match(r'^(.+?)_Alpha_', filename, re.IGNORECASE)
    if alt_match:
        configuration = alt_match.group(1)
        # Remove AirOn/AirOff if present
        configuration = re.sub(r'^(AirOn|AirOff)_?', '', configuration, flags=re.IGNORECASE)
        return configuration if configuration else 'Unknown'

    return 'Unknown'


def extract_air_state_from_filename(filepath: str) -> str:
    """
    Extract air state (AirOn/AirOff) from a filename.

    Parameters
    ----------
    filepath : str
        File path to parse

    Returns
    -------
    str
        'AirOn', 'AirOff', or 'Unknown'
    """
    filename = Path(filepath).stem.lower()

    if 'airon' in filename:
        return 'AirOn'
    elif 'airoff' in filename:
        return 'AirOff'
    else:
        return 'Unknown'


@dataclass
class FileInfo:
    """Information extracted from a TDMS filename."""
    filepath: Path
    configuration: str
    air_state: str
    alpha: float
    beta: float


def parse_tdms_filename(filepath: str) -> FileInfo:
    """
    Parse all information from a TDMS filename.

    Parameters
    ----------
    filepath : str
        Path to TDMS file

    Returns
    -------
    FileInfo
        Parsed file information
    """
    filepath = Path(filepath)
    alpha, beta = extract_alpha_beta_from_filename(str(filepath))
    config = extract_configuration_from_filename(str(filepath))
    air_state = extract_air_state_from_filename(str(filepath))

    return FileInfo(
        filepath=filepath,
        configuration=config,
        air_state=air_state,
        alpha=alpha,
        beta=beta
    )


def group_files_by_configuration(files: list) -> Dict[str, Dict[str, list]]:
    """
    Group TDMS files by configuration and air state.

    Parameters
    ----------
    files : list
        List of file paths

    Returns
    -------
    dict
        Nested dictionary: {config_name: {'AirOn': [files], 'AirOff': [files]}}
    """
    grouped = {}

    for f in files:
        info = parse_tdms_filename(str(f))

        if info.configuration not in grouped:
            grouped[info.configuration] = {'AirOn': [], 'AirOff': [], 'Unknown': []}

        grouped[info.configuration][info.air_state].append(info)

    # Sort files within each group by alpha, then beta
    for config in grouped:
        for air_state in grouped[config]:
            grouped[config][air_state].sort(key=lambda x: (x.alpha, x.beta))

    return grouped
