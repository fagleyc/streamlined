"""
Data I/O Module
===============

Functions for reading wind tunnel data files (TDMS, HDF5 and MATLAB .mat
formats) and exporting processed data.
"""

import json
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from scipy.interpolate import interp1d

try:
    from nptdms import TdmsFile
    TDMS_AVAILABLE = True
except ImportError:
    TDMS_AVAILABLE = False

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    from scipy import io as scipy_io
    MAT_AVAILABLE = True
except ImportError:
    MAT_AVAILABLE = False


# Property names recognized in file/group attributes (run parameters)
PROPERTY_TYPES = ['Stiffness', 'Damping', 'Mass', 'L1', 'L2', 'L3', 'L4',
                  'Alpha', 'Beta', 'Plunge', 'Roll']

# Known balance groups written by Freestream/Conductor run files.
# Internal (sting) balance: bridge volts (N1, N2, Y1, Y2, Axial, Roll)
# needing .vol calibration via calc_brf_forces.
# External (ATE) balance: resolved wind-axis loads (Lift, Pitch, Drag,
# Side, Yaw, Roll) in N / N*m — no calibration/reduction needed.
BALANCE_GROUP_INTERNAL = 'StrainBook_0'
BALANCE_GROUP_EXTERNAL = 'ATE_Balance'

# Channels whose per-channel unit attribute decides the resolved-load
# unit system of an external-balance file (Freestream writes 'N').
_EXTERNAL_UNIT_PROBE_CHANNELS = ('Lift', 'Drag', 'Side')
# ...and the moment channels, which are what separate the OGI's "Lb and
# Lbft" setting from the chain's native lb / in-lb.
_EXTERNAL_MOMENT_PROBE_CHANNELS = ('Pitch', 'Yaw', 'Roll')


def _probe_unit(channel_units: Dict[str, Any],
                channels: Tuple[str, ...]) -> str:
    """First non-empty unit string among ``channels``, lowercased."""
    for ch in channels:
        unit = channel_units.get(ch)
        if isinstance(unit, bytes):
            unit = unit.decode('utf-8', errors='replace')
        if isinstance(unit, str) and unit.strip():
            return unit.strip().lower()
    return ''


def _finalize_load_units(raw: 'RawData',
                         channel_units: Dict[str, Any]) -> None:
    """
    Record a ``load_units`` marker for external-balance files.

    The ATE_Balance channels carry per-channel ``unit`` attributes; the
    downstream chain works in lb / in-lb
    (deprecated/scripts/calc_coeffs.m 'External'), so the unit system is
    surfaced in ``raw.properties['load_units']`` for the reducers to
    convert on.

    The OGI's engineering-unit setting is operator-selectable — "Kg and
    Kgm, N and Nm, Lb and Lbft" — and never appears on the wire, so the
    recorded unit attributes are the only witness. All four systems get
    distinct markers, in particular ``'lbft'`` for pounds-with-FEET,
    whose moments need a x12 conversion the old "not-SI means already
    native" rule would have missed. Files without unit metadata get no
    marker (treated as already lb / in-lb, the legacy behaviour).
    """
    if raw.properties.get('balance_type') != 'external':
        return
    if 'load_units' in raw.properties:
        return
    force_u = _probe_unit(channel_units, _EXTERNAL_UNIT_PROBE_CHANNELS)
    if not force_u:
        return
    moment_u = _probe_unit(channel_units, _EXTERNAL_MOMENT_PROBE_CHANNELS)

    if force_u.startswith('n'):
        marker = 'N'
    elif force_u.startswith('kg'):
        marker = 'kg'
    elif 'ft' in moment_u:
        marker = 'lbft'
    else:
        marker = 'lb'
    raw.properties['load_units'] = marker


@dataclass
class RawData:
    """Container for raw wind tunnel data from a single file."""
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    data: Dict[str, np.ndarray] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    filename: str = ""

    @property
    def balance_type(self) -> str:
        """
        'internal' (bridge volts needing calibration) or 'external'
        (resolved wind-axis loads, no bridge-to-force reduction needed).

        Carried in ``properties`` like all other file metadata; files
        without a marker (legacy TDMS/HDF5) default to 'internal'.
        """
        return str(self.properties.get('balance_type', 'internal')).strip().lower()

    @property
    def balance_group(self) -> str:
        """In-file balance group name (defaults to the legacy StrainBook)."""
        return str(self.properties.get('balance_group', BALANCE_GROUP_INTERNAL))


def _finalize_balance_markers(raw: RawData, present_groups) -> None:
    """
    Normalize/derive the self-describing balance markers in
    ``raw.properties``.

    Marker values already merged into raw.properties (from root attrs,
    /meta, or meta.devices.ate) win; otherwise the balance flavor is
    inferred from which balance group is present in the file, and
    legacy files with neither cue default to the historical
    StrainBook_0 / internal behavior.
    """
    group = raw.properties.get('balance_group')
    btype = raw.properties.get('balance_type')

    group = None if group is None else str(group).strip()
    btype = None if btype is None else str(btype).strip().lower()
    if btype not in ('internal', 'external'):
        btype = None

    if btype is None:
        if group is not None:
            btype = 'external' if group == BALANCE_GROUP_EXTERNAL else 'internal'
        elif BALANCE_GROUP_EXTERNAL in present_groups:
            btype = 'external'
        else:
            btype = 'internal'

    if group is None:
        group = (BALANCE_GROUP_EXTERNAL if btype == 'external'
                 else BALANCE_GROUP_INTERNAL)

    raw.properties['balance_group'] = group
    raw.properties['balance_type'] = btype


def _resample_channels_to_fastest(channels: Dict[str, Dict[str, Any]],
                                  raw: RawData) -> None:
    """
    Fill raw.time / raw.data from a collected channel dict, resampling
    everything onto the fastest (smallest-dt) channel's time base.

    ``channels`` maps channel name -> {'data': ndarray, 'time': ndarray,
    'group': str}, exactly as built by the TDMS/HDF5/MAT readers. Channels
    already on the fastest time base are copied (truncated to its length);
    slower channels are cubic-interpolated onto it.

    Outside a slow channel's own time span the interpolant is CLAMPED to
    its first/last sample rather than extrapolated. Cubic extrapolation
    diverges cubically, and a slow group whose block ends before the
    fastest one's routinely leaves a tail to fill: an ATE run streaming
    35 load samples against the DaqBook's 200 came back with a Lift
    channel ranging to -1463 N from data that never left 204..207 N,
    dragging the point mean from 205 N to 59 N. Holding the end value is
    the honest answer for a steady dwell — it cannot invent a number the
    instrument never read.
    """
    if not channels:
        return

    dt_values = []
    for name, ch in channels.items():
        if len(ch['time']) > 1:
            dt_values.append(ch['time'][1] - ch['time'][0])

    if not dt_values:
        return

    min_dt = min(dt_values)

    # Find the reference channel (smallest dt)
    ref_time = None
    for name, ch in channels.items():
        if len(ch['time']) > 1:
            dt = ch['time'][1] - ch['time'][0]
            if np.isclose(dt, min_dt):
                ref_time = ch['time']
                break

    if ref_time is None:
        return

    raw.time = ref_time

    # Resample all channels to the reference time
    for name, ch in channels.items():
        n = len(ch['data'])
        # Slow instruments (e.g. the ~4 Hz Heise indicator) can yield 0
        # or 1 samples in a short acquisition — interp1d needs >= 2
        # points (cubic needs >= 4), so degrade gracefully instead of
        # crashing: constant-fill a single sample, NaN-fill an empty
        # channel, and drop to linear when cubic lacks points.
        if n == 0:
            raw.data[name] = np.full(len(ref_time), np.nan)
            continue
        if n == 1:
            raw.data[name] = np.full(len(ref_time), float(ch['data'][0]))
            continue

        ch_dt = ch['time'][1] - ch['time'][0] if len(ch['time']) > 1 else min_dt

        if not np.isclose(ch_dt, min_dt) or n < len(ref_time):
            data = np.asarray(ch['data'], dtype=float)
            interp_func = interp1d(
                ch['time'], data,
                kind='cubic' if n >= 4 else 'linear',
                bounds_error=False,
                # clamp, do NOT extrapolate — see the docstring
                fill_value=(float(data[0]), float(data[-1]))
            )
            raw.data[name] = interp_func(ref_time)
        else:
            # Same time base, just copy
            raw.data[name] = ch['data'][:len(ref_time)]


def _ensure_alpha_beta(raw: RawData, properties: Dict[str, Any],
                       filepath: str) -> None:
    """
    Ensure Alpha and Beta are in raw.data. They might be channels
    (time-series), properties (single values), or encoded in the filename.
    """
    n_samples = len(raw.time) if len(raw.time) > 0 else 1

    if 'Alpha' not in raw.data:
        if 'Alpha' in properties:
            raw.data['Alpha'] = np.full(n_samples, float(properties['Alpha']))
        else:
            alpha, _ = extract_alpha_beta_from_filename(filepath)
            raw.data['Alpha'] = np.full(n_samples, alpha)

    if 'Beta' not in raw.data:
        if 'Beta' in properties:
            raw.data['Beta'] = np.full(n_samples, float(properties['Beta']))
        else:
            _, beta = extract_alpha_beta_from_filename(filepath)
            raw.data['Beta'] = np.full(n_samples, beta)


# Filename speed-token unit tags -> canonical speed_unit strings.
# The token sits in the slot AFTER alpha/beta (replacing the legacy mach
# token for non-Mach sweeps): run_0001_alpha_0.0_beta_0.0_Hz_30.0.h5.
# Ordered so the more specific tags are tried before the shorter ones.
_SPEED_UNIT_TAGS = (
    ('ftps', 'ft/s'),
    ('mps', 'm/s'),
    ('rpm', 'rpm'),
    ('hz', 'hz'),
    ('mach', 'mach'),
)


def extract_speed_from_filename(filepath: str) -> Tuple[Optional[float],
                                                        Optional[str]]:
    """
    Extract the tunnel speed setting from a filename speed token.

    Mirrors :func:`extract_alpha_beta_from_filename` /
    :func:`extract_mach_from_filename`, parsing any of the
    ``{Hz|ftps|mps|RPM|mach}_<value>`` tokens that Freestream writes in
    the slot after alpha/beta. Non-Mach velocity sweeps use the Hz / ftps
    / mps / RPM tags; Mach sweeps keep the legacy ``mach`` token.

    Parameters
    ----------
    filepath : str
        File path to parse

    Returns
    -------
    tuple
        ``(value, unit)`` where ``unit`` is one of
        ``'hz'``/``'ft/s'``/``'m/s'``/``'rpm'``/``'mach'``, or
        ``(None, None)`` when no speed token is present.
    """
    import re

    filename = Path(filepath).stem
    for tag, unit in _SPEED_UNIT_TAGS:
        match = re.search(rf'{tag}[_\s]*(-?\d+\.?\d*)', filename,
                          re.IGNORECASE)
        if match:
            return float(match.group(1)), unit
    return None, None


def _ensure_speed(raw: 'RawData', properties: Dict[str, Any],
                  filepath: str) -> None:
    """
    Ensure the tunnel speed setting is exposed on ``raw`` like Alpha/Beta.

    The speed is a first-class sweep dimension: this fills
    ``raw.properties['speed_value']`` / ``['speed_unit']`` and adds a
    ``Speed`` channel (np.full to the sample count) to ``raw.data``.

    Resolution order (root-attr wins, filename is the fallback):

    1. ``speed_value`` / ``speed_unit`` already on ``raw.properties``
       (copied from the file's root attrs by the readers).
    2. The filename ``{Hz|ftps|mps|RPM|mach}_<value>`` token.
    3. Legacy Mach-only files: the canonical ``mach`` (from properties or
       the filename) -> ``speed_unit='mach'``, ``speed_value=<mach>``.

    Degrades gracefully (no Speed channel, no markers) when none of these
    yield a value, so callers such as read_tdms_file stay unaffected.
    """
    n_samples = len(raw.time) if len(raw.time) > 0 else 1

    value = raw.properties.get('speed_value')
    unit = raw.properties.get('speed_unit')
    if value is not None and unit is not None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        unit = str(unit).strip().lower()
    else:
        value, unit = extract_speed_from_filename(filepath)

    # Legacy Mach-only fallback: canonical mach becomes the speed setting.
    if value is None or unit is None:
        mach = raw.properties.get('mach', properties.get('mach'))
        if mach is None:
            mach = extract_mach_from_filename(filepath)
        if mach is not None:
            try:
                value, unit = float(mach), 'mach'
            except (TypeError, ValueError):
                value, unit = None, None

    if value is None or unit is None:
        return

    raw.properties['speed_value'] = value
    raw.properties['speed_unit'] = unit
    if 'Speed' not in raw.data:
        raw.data['Speed'] = np.full(n_samples, value)


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


def _capture_injected_balance_cal(raw: RawData, dev_attrs, decode) -> None:
    """Capture freestream's injected balance calibration (the computed
    ``cal_matrix`` + ``cal_type`` + ``cal_distances`` + ``balance_serial``
    written into the balance device's ``/meta/devices/<id>`` attrs) into
    ``raw.properties['injected_balance_cal']``, so the reduction can rebuild a
    BalanceCalibration WITHOUT the .vol (an explicitly loaded .vol still
    overrides it — see the data controller's precedence)."""
    try:
        matrix = np.asarray(dev_attrs['cal_matrix'], dtype=float)
    except Exception:                                  # noqa: BLE001
        return
    cal: Dict[str, Any] = {'matrix': matrix}
    if 'cal_type' in dev_attrs:
        cal['cal_type'] = decode(dev_attrs['cal_type'])
    if 'cal_distances' in dev_attrs:
        try:
            cal['distances'] = np.asarray(dev_attrs['cal_distances'],
                                          dtype=float)
        except Exception:                              # noqa: BLE001
            pass
    if 'balance_serial' in dev_attrs:
        cal['serial'] = decode(dev_attrs['balance_serial'])
    raw.properties['injected_balance_cal'] = cal


def read_hdf5_file(filepath: str) -> Tuple[RawData, Dict[str, Any]]:
    """
    Read a Conductor HDF5 run file and return data resampled to a
    common time base.

    Returns the same structure as :func:`read_tdms_file`, so existing
    consumers (``daq.DAQ.load_data_directory``, the GUI data controller)
    can use either format interchangeably.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file (.h5 or .hdf5)

    Returns
    -------
    tuple
        (RawData, properties) where RawData contains all channels
        resampled to a common time base

    Notes
    -----
    Conductor writes one group per device (StrainBook_0, DaqBook2005,
    Positioner, Tunnel) plus a Time group and /meta bookkeeping groups.
    The Positioner group replaces the legacy TDMS "Arc Crescent" group;
    since group names are flattened away (channels are keyed by channel
    name only, exactly as in read_tdms_file), the Alpha/Beta channels
    land in RawData.data under the same keys either way.
    Per-dataset attributes wf_increment/wf_samples define each channel's
    time base, mirroring the TDMS waveform properties. Root attributes
    (run parameters) are exposed via RawData.properties, and any that
    match the known property names (Alpha, Beta, Stiffness, ...) are
    also returned in the properties dict, as with TDMS group properties.
    This function requires the h5py library.
    """
    if not HDF5_AVAILABLE:
        raise ImportError(
            "h5py library is required to read HDF5 files. "
            "Install with: pip install h5py"
        )

    filepath = Path(filepath)
    if filepath.suffix.lower() not in ('.h5', '.hdf5'):
        filepath = filepath.with_suffix('.h5')

    raw = RawData(filename=str(filepath))
    properties = {}

    def _decode(value):
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='replace')
        return value

    with h5py.File(str(filepath), 'r') as h5_file:
        # Collect all channels and their time vectors
        channels = {}
        channel_units = {}
        channel_cal = {}
        present_groups = set()

        for group_name, group in h5_file.items():
            if not isinstance(group, h5py.Group):
                continue
            if group_name in ('Time', 'meta'):
                continue
            present_groups.add(group_name)

            for dataset_name, dataset in group.items():
                if not isinstance(dataset, h5py.Dataset):
                    continue

                channel_name = dataset_name.replace(' ', '_')
                data = dataset[:]

                # Get time vector from dataset attributes
                attrs = dataset.attrs
                if 'unit' in attrs:
                    channel_units[channel_name] = _decode(attrs['unit'])
                # Freestream injects per-channel tunnel calibration
                # coefficients (cal_slope/cal_offset/cal_unit/cal_type) so the
                # reduction converts raw volts -> engineering units WITHOUT an
                # external .pcf, and skips scaling for already-engineering
                # instruments (Heise, cal_type='identity'). See
                # extract_channel_cal / calc_tunnel_conditions.
                if 'cal_type' in attrs or 'cal_slope' in attrs:
                    channel_cal[channel_name] = {
                        'slope': float(attrs['cal_slope'])
                        if 'cal_slope' in attrs else 1.0,
                        'offset': float(attrs['cal_offset'])
                        if 'cal_offset' in attrs else 0.0,
                        'unit': _decode(attrs['cal_unit'])
                        if 'cal_unit' in attrs else '',
                        'type': _decode(attrs['cal_type'])
                        if 'cal_type' in attrs else 'linear',
                    }
                if 'wf_increment' in attrs and 'wf_samples' in attrs:
                    dt = float(attrs['wf_increment'])
                    n_samples = int(attrs['wf_samples'])
                    time = np.arange(n_samples) * dt
                else:
                    # Try to get from data length
                    time = np.arange(len(data))

                channels[channel_name] = {
                    'data': np.array(data),
                    'time': time,
                    'group': group_name.replace(' ', '_')
                }

        # Resample everything onto the fastest channel's time base
        _resample_channels_to_fastest(channels, raw)

        # Per-channel injected tunnel calibration (empty for legacy files ->
        # the reduction falls back to the built-in DaqBook default cal).
        if channel_cal:
            raw.properties['channel_cal'] = channel_cal

        # Root attributes carry the run parameters (run_number, air_state,
        # inherited run-sheet params, ...) -> file properties
        for attr_name, attr_value in h5_file.attrs.items():
            raw.properties[attr_name] = _decode(attr_value)

        # Self-describing balance markers may live on the root attrs
        # (already copied above) and/or under /meta — merge the /meta
        # copies in only where the root attrs did not provide them, then
        # normalize with a fallback on which balance group is present
        # (legacy files without markers stay StrainBook_0 / internal).
        if 'meta' in h5_file:
            meta = h5_file['meta']
            marker_sources = [meta.attrs]
            if 'devices/ate' in meta:
                marker_sources.append(meta['devices/ate'].attrs)
            for attrs in marker_sources:
                for key in ('balance_group', 'balance_type'):
                    if key not in raw.properties and key in attrs:
                        raw.properties[key] = _decode(attrs[key])
            # freestream injects the computed balance calibration matrix into
            # the balance device's /meta/devices/<id> group (cal_matrix +
            # cal_type + cal_distances), so Streamlined can reduce forces
            # WITHOUT the .vol. Capture the FIRST device that carries it.
            if 'devices' in meta:
                for dev_name, dev in meta['devices'].items():
                    if hasattr(dev, 'attrs') and 'cal_matrix' in dev.attrs:
                        _capture_injected_balance_cal(raw, dev.attrs, _decode)
                        break
        _finalize_balance_markers(raw, present_groups)
        _finalize_load_units(raw, channel_units)

        # Extract properties (Alpha, Beta, etc.) from root and group attrs
        attr_sources = [h5_file.attrs]
        for group_name, group in h5_file.items():
            if isinstance(group, h5py.Group):
                attr_sources.append(group.attrs)

        for attrs in attr_sources:
            for prop_name, prop_value in attrs.items():
                for ptype in PROPERTY_TYPES:
                    if ptype in prop_name:
                        properties[ptype] = _decode(prop_value)

        # Alpha/Beta might be channels, properties or filename-encoded
        _ensure_alpha_beta(raw, properties, str(filepath))
        # Speed (Hz/ftps/mps/RPM/mach) is a first-class sweep dimension
        _ensure_speed(raw, properties, str(filepath))

    return raw, properties


def _is_mat_struct(value: Any) -> bool:
    """True for scipy.io mat_struct objects (struct_as_record=False)."""
    return hasattr(value, '_fieldnames')


def _mat_to_python(value: Any) -> Any:
    """loadmat value -> plain Python: numpy scalars unwrapped, empty
    char arrays -> '', real arrays passed through."""
    if isinstance(value, np.ndarray):
        if value.size == 0 and value.dtype.kind in ('U', 'S'):
            return ''
        if value.ndim == 0:
            return value.item()
        return value
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mat_struct_to_dict(struct: Any) -> Dict[str, Any]:
    """Flatten a (possibly absent/empty) mat_struct into a plain dict."""
    if not _is_mat_struct(struct):
        return {}
    return {name: _mat_to_python(getattr(struct, name))
            for name in struct._fieldnames}


def read_mat_file(filepath: str) -> Tuple[RawData, Dict[str, Any]]:
    """
    Read a Conductor MATLAB .mat run file and return data resampled to
    a common time base.

    Returns the same structure as :func:`read_tdms_file` /
    :func:`read_hdf5_file`, so existing consumers can use any of the
    three formats interchangeably.

    Parameters
    ----------
    filepath : str
        Path to the MATLAB file (.mat)

    Returns
    -------
    tuple
        (RawData, properties) where RawData contains all channels
        resampled to a common time base

    Notes
    -----
    Conductor's .mat sibling files mirror the HDF5 schema: one top-level
    struct per device group (StrainBook_0, DaqBook2005, Positioner,
    Tunnel, Time) with channel arrays as fields, plus a ``meta`` struct:

    * ``meta.run``            -- root attrs (run parameters)
    * ``meta.channels.<G>.<C>`` -- wf_increment / wf_samples /
      wf_start_time / unit per channel (drives each channel's time base,
      exactly like the HDF5 dataset attributes)
    * ``meta.devices``        -- per-device attrs (cal-file POINTERS)
    * ``meta.config_json``    -- measurement-config snapshot (JSON string)
    * ``meta.name_map``       -- sanitized->original name mapping
      (``groups`` / ``channels.<G>`` / ``run`` / ``devices`` substructs);
      used here to restore the original group/channel/attr names so the
      returned keys match read_hdf5_file on the sibling .h5 file.

    Group names are flattened away (channels keyed by channel name only)
    and the Time group is skipped, mirroring the TDMS/HDF5 readers.
    This function requires the scipy library.
    """
    if not MAT_AVAILABLE:
        raise ImportError(
            "scipy library is required to read MATLAB .mat files. "
            "Install with: pip install scipy"
        )

    filepath = Path(filepath)
    if filepath.suffix.lower() != '.mat':
        filepath = filepath.with_suffix('.mat')

    raw = RawData(filename=str(filepath))
    properties = {}

    contents = scipy_io.loadmat(str(filepath), squeeze_me=True,
                                struct_as_record=False)

    # meta bookkeeping: per-channel waveform attrs + name sanitization map
    meta = contents.get('meta')
    chan_meta = None
    group_names: Dict[str, Any] = {}
    run_names: Dict[str, Any] = {}
    chan_name_maps: Dict[str, Dict[str, Any]] = {}
    if _is_mat_struct(meta):
        chan_meta = getattr(meta, 'channels', None)
        name_map = getattr(meta, 'name_map', None)
        if _is_mat_struct(name_map):
            group_names = _mat_struct_to_dict(getattr(name_map, 'groups', None))
            run_names = _mat_struct_to_dict(getattr(name_map, 'run', None))
            ch_nm = getattr(name_map, 'channels', None)
            if _is_mat_struct(ch_nm):
                for g in ch_nm._fieldnames:
                    chan_name_maps[g] = _mat_struct_to_dict(getattr(ch_nm, g))

    # Collect all channels and their time vectors
    channels = {}
    channel_units = {}
    channel_cal = {}
    present_groups = set()

    for key, value in contents.items():
        if key.startswith('__') or key == 'meta' or not _is_mat_struct(value):
            continue

        group_orig = str(group_names.get(key, key))
        if group_orig == 'Time':
            continue
        present_groups.add(group_orig)

        g_names = chan_name_maps.get(key, {})
        g_meta = getattr(chan_meta, key, None) if chan_meta is not None else None

        for field_name in value._fieldnames:
            data = np.atleast_1d(
                np.asarray(getattr(value, field_name), dtype=np.float64)
            ).ravel()
            channel_name = str(g_names.get(field_name,
                                           field_name)).replace(' ', '_')

            # Get time vector from the per-channel meta struct
            c_meta = (getattr(g_meta, field_name, None)
                      if _is_mat_struct(g_meta) else None)
            if c_meta is not None and hasattr(c_meta, 'unit'):
                channel_units[channel_name] = _mat_to_python(c_meta.unit)
            # freestream injects per-channel tunnel cal into the .mat's
            # meta.channels.<group>.<channel> struct (cal_slope/offset/unit/
            # type) — same contract as the HDF5 dataset attrs. Capture it so
            # the reduction applies the REAL device cal (not the built-in
            # DaqBook default) for .mat run files too.
            if c_meta is not None and (hasattr(c_meta, 'cal_type')
                                       or hasattr(c_meta, 'cal_slope')):
                channel_cal[channel_name] = {
                    'slope': float(_mat_to_python(c_meta.cal_slope))
                    if hasattr(c_meta, 'cal_slope') else 1.0,
                    'offset': float(_mat_to_python(c_meta.cal_offset))
                    if hasattr(c_meta, 'cal_offset') else 0.0,
                    'unit': str(_mat_to_python(c_meta.cal_unit))
                    if hasattr(c_meta, 'cal_unit') else '',
                    'type': str(_mat_to_python(c_meta.cal_type))
                    if hasattr(c_meta, 'cal_type') else 'linear',
                }
            if (c_meta is not None and hasattr(c_meta, 'wf_increment')
                    and hasattr(c_meta, 'wf_samples')):
                dt = float(_mat_to_python(c_meta.wf_increment))
                n_samples = int(_mat_to_python(c_meta.wf_samples))
                time = np.arange(n_samples) * dt
            else:
                # Try to get from data length
                time = np.arange(len(data))

            channels[channel_name] = {
                'data': data,
                'time': time,
                'group': group_orig.replace(' ', '_')
            }

    # Resample everything onto the fastest channel's time base
    _resample_channels_to_fastest(channels, raw)

    # Per-channel injected tunnel calibration (empty for legacy files ->
    # the reduction falls back to the built-in DaqBook default cal).
    if channel_cal:
        raw.properties['channel_cal'] = channel_cal

    # meta.run carries the run parameters (root attrs) -> file properties,
    # restored to their original (pre-sanitization) key names
    if _is_mat_struct(meta):
        run_struct = getattr(meta, 'run', None)
        if _is_mat_struct(run_struct):
            for key in run_struct._fieldnames:
                orig_key = str(run_names.get(key, key))
                raw.properties[orig_key] = _mat_to_python(
                    getattr(run_struct, key))

    # Self-describing balance markers: meta.run (root attrs) was merged
    # above; fall back to the ATE device record (meta.devices.ate) when
    # absent, then normalize (legacy files stay StrainBook_0 / internal).
    if _is_mat_struct(meta):
        devices = getattr(meta, 'devices', None)
        ate = getattr(devices, 'ate', None) if _is_mat_struct(devices) else None
        if _is_mat_struct(ate):
            for key in ('balance_group', 'balance_type'):
                if key not in raw.properties and key in ate._fieldnames:
                    raw.properties[key] = _mat_to_python(getattr(ate, key))
        # freestream's injected balance cal matrix from meta.devices.<id>
        if _is_mat_struct(devices):
            for dev_name in devices._fieldnames:
                dev = getattr(devices, dev_name)
                if not (_is_mat_struct(dev)
                        and 'cal_matrix' in dev._fieldnames):
                    continue
                cal = {'matrix': np.asarray(
                    _mat_to_python(dev.cal_matrix), dtype=float)}
                if 'cal_type' in dev._fieldnames:
                    cal['cal_type'] = _mat_to_python(dev.cal_type)
                if 'cal_distances' in dev._fieldnames:
                    cal['distances'] = np.asarray(
                        _mat_to_python(dev.cal_distances), dtype=float)
                if 'balance_serial' in dev._fieldnames:
                    cal['serial'] = _mat_to_python(dev.balance_serial)
                raw.properties['injected_balance_cal'] = cal
                break
    _finalize_balance_markers(raw, present_groups)
    _finalize_load_units(raw, channel_units)

    # Extract properties (Alpha, Beta, etc.) from the run parameters
    for prop_name, prop_value in raw.properties.items():
        for ptype in PROPERTY_TYPES:
            if ptype in prop_name:
                properties[ptype] = prop_value

    # Alpha/Beta might be channels, properties or filename-encoded
    _ensure_alpha_beta(raw, properties, str(filepath))
    # Speed (Hz/ftps/mps/RPM/mach) is a first-class sweep dimension
    _ensure_speed(raw, properties, str(filepath))

    return raw, properties


def copy_balance_markers(raw: RawData,
                         channel_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Copy the self-describing balance markers from ``raw.properties``
    into a flat channel dict.

    The reduction chain receives plain channel dicts, so the markers
    ride along as dict entries: ``balance_type`` drives
    :func:`~.transforms.is_external_balance_data`, ``load_units`` drives
    the unit conversion in
    :func:`~.external_balance.external_loads_to_ips`, ``span_config``
    selects the mount-dependent channel resolution in
    :func:`~.external_balance.resolve_external_wrf`, the speed
    markers (``speed_value`` / ``speed_unit`` / ``speed_setpoints``)
    carry the tunnel-speed sweep dimension through to the reducers, and
    ``channel_cal`` carries freestream's injected per-channel tunnel
    calibration (so :func:`~.coefficients.calc_tunnel_conditions` converts
    raw volts -> engineering units with no external .pcf).
    """
    for key in ('balance_type', 'load_units', 'span_config',
                'speed_value', 'speed_unit', 'speed_setpoints',
                'channel_cal'):
        if key in raw.properties:
            channel_dict[key] = raw.properties[key]
    return channel_dict


def read_run_file(filepath: str) -> Tuple[RawData, Dict[str, Any]]:
    """
    Read a wind tunnel run file, dispatching on file extension.

    Routes .h5/.hdf5 files to read_hdf5_file, .mat files to
    read_mat_file, and everything else (including extensionless paths)
    to read_tdms_file, preserving the historical TDMS-by-default
    behavior.

    Parameters
    ----------
    filepath : str
        Path to the data file

    Returns
    -------
    tuple
        (RawData, properties) as returned by the format-specific reader
    """
    suffix = Path(filepath).suffix.lower()
    if suffix in ('.h5', '.hdf5'):
        return read_hdf5_file(filepath)
    if suffix == '.mat':
        return read_mat_file(filepath)
    return read_tdms_file(filepath)


# ---------------------------------------------------------------------------
# Metadata-only ingest: the facts a run file records about itself
# ---------------------------------------------------------------------------

# Memoised run metadata, keyed by (absolute path, mtime, size). A directory
# scan asks the same file the same questions several times (classification,
# grouping, config seeding); it must be opened once, not once per question.
_RUN_METADATA_CACHE: Dict[Tuple[str, float, int], Dict[str, Any]] = {}

# Canonical air-state spellings. Anything else recorded in a file is not
# understood and falls back to the filename rather than being guessed at.
_AIR_STATES = {'airon': 'AirOn', 'airoff': 'AirOff'}


def _parse_config_json(value: Any) -> Dict[str, Any]:
    """The measurement config behind a ``config_json`` attribute.

    Returns ``{}`` when the attribute is absent, empty or not parsable
    JSON — a config that cannot be read seeds nothing.
    """
    text = _mat_to_python(value)
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_mat_run_metadata(path: Path) -> Dict[str, Any]:
    """``meta.run`` attrs + parsed ``meta.config_json`` from a .mat run file.

    ``variable_names=['meta']`` is what keeps this cheap: the device
    groups (the megabytes of channel arrays) are never unpacked.
    """
    if not MAT_AVAILABLE:
        return {}

    contents = scipy_io.loadmat(str(path), variable_names=['meta'],
                                squeeze_me=True, struct_as_record=False)
    meta = contents.get('meta')
    if not _is_mat_struct(meta):
        return {}

    # Restore the original (pre-sanitization) attr names, as read_mat_file
    # does when it merges meta.run into raw.properties.
    run_names: Dict[str, Any] = {}
    name_map = getattr(meta, 'name_map', None)
    if _is_mat_struct(name_map):
        run_names = _mat_struct_to_dict(getattr(name_map, 'run', None))

    info: Dict[str, Any] = {}
    run_struct = getattr(meta, 'run', None)
    if _is_mat_struct(run_struct):
        for key in run_struct._fieldnames:
            info[str(run_names.get(key, key))] = _mat_to_python(
                getattr(run_struct, key))

    info['config'] = _parse_config_json(getattr(meta, 'config_json', None))
    return info


def _read_hdf5_run_metadata(path: Path) -> Dict[str, Any]:
    """Root attrs of an HDF5 run file (no dataset is ever touched).

    h5py is optional; without it this branch degrades to ``{}`` and the
    caller falls back to the filename parsers.
    """
    if not HDF5_AVAILABLE:
        return {}

    info: Dict[str, Any] = {}
    with h5py.File(str(path), 'r') as f:
        for key, value in f.attrs.items():
            if isinstance(value, bytes):
                value = value.decode('utf-8', errors='replace')
            info[str(key)] = value
    info['config'] = _parse_config_json(info.pop('config_json', None))
    return info


def read_run_metadata(filepath: str) -> Dict[str, Any]:
    """
    Read a run file's METADATA only — never its channel arrays.

    Freestream records into every run file the facts Streamlined used to
    guess from the filename: air state, configuration, alpha/beta, tunnel
    speed, acquisition run number and hysteresis leg. Reading just those
    is cheap enough to run over a whole directory:

    * ``.mat``  -- ``loadmat(variable_names=['meta'])``, so only the meta
      struct is unpacked.
    * ``.h5``   -- the ROOT attributes via h5py.
    * ``.tdms`` -- ``{}``; legacy files carry no such record and stay on
      the filename path.

    Results are memoised per (path, mtime, size).

    Parameters
    ----------
    filepath : str
        Path to the run file

    Returns
    -------
    dict
        The flat ``meta.run`` attributes plus a ``'config'`` key holding
        the parsed ``meta.config_json`` measurement config (``{}`` when
        absent). A missing, unreadable or corrupt file returns ``{}``:
        this never raises, since one bad file must not abort the scan of
        a directory.
    """
    path = Path(filepath)
    try:
        stat = path.stat()
    except OSError:
        return {}                     # nonexistent path -> filename only

    key = (str(path), stat.st_mtime, stat.st_size)
    cached = _RUN_METADATA_CACHE.get(key)
    if cached is not None:
        return cached

    suffix = path.suffix.lower()
    try:
        if suffix == '.mat':
            meta = _read_mat_run_metadata(path)
        elif suffix in ('.h5', '.hdf5'):
            meta = _read_hdf5_run_metadata(path)
        else:
            meta = {}
    except Exception as exc:
        warnings.warn(f"Could not read run metadata from '{path.name}': "
                      f"{type(exc).__name__}: {exc}")
        meta = {}

    _RUN_METADATA_CACHE[key] = meta
    return meta


def _meta_str(meta: Dict[str, Any], key: str) -> Optional[str]:
    """A non-empty string metadata value, or None when absent/blank."""
    value = meta.get(key)
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='replace')
    text = str(value).strip()
    return text or None


def _meta_float(meta: Dict[str, Any], key: str) -> Optional[float]:
    """A finite float metadata value, or None when absent/unusable."""
    try:
        value = float(meta[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _meta_int(meta: Dict[str, Any], key: str) -> Optional[int]:
    """An integer metadata value, or None when absent/unusable."""
    value = _meta_float(meta, key)
    return None if value is None else int(round(value))


def _normalize_air_state(value: Any) -> Optional[str]:
    """Canonical 'AirOn'/'AirOff' for a recorded air-state value.

    Returns None for anything not recognized, so the caller falls back
    to the filename instead of inventing a third state (the grouping
    dicts are keyed on exactly these names).
    """
    if value is None:
        return None
    text = str(value).strip().lower().replace(' ', '').replace('_', '')
    return _AIR_STATES.get(text)


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


# ---------------------------------------------------------------------------
# Run-directory index: manifest.json + metadata-first file discovery
# ---------------------------------------------------------------------------

# Extensions a run file can have. manifest.json is an INDEX, not a run
# file, and must never reach a reader — no pattern here matches it.
RUN_FILE_PATTERNS = ('*.tdms', '*.h5', '*.hdf5', '*.mat')

MANIFEST_FILENAME = 'manifest.json'
MANIFEST_SCHEMA_VERSION = 1


def find_run_files(directory: str, recursive: bool = False) -> list:
    """
    Every run file in a directory, in sorted path order.

    Parameters
    ----------
    directory : str
        Directory to search
    recursive : bool
        Whether to search subdirectories

    Returns
    -------
    list
        List of Path objects, excluding ``manifest.json`` (an index, not
        a run file).
    """
    data_dir = Path(directory)
    files = sorted(f for pat in RUN_FILE_PATTERNS
                   for f in (data_dir.rglob(pat) if recursive
                             else data_dir.glob(pat)))
    return [f for f in files if f.name.lower() != MANIFEST_FILENAME]


def read_run_manifest(directory: str) -> Dict[str, Any]:
    """
    Read the ``manifest.json`` index a run directory may carry.

    Freestream writes one manifest per config directory, alongside the
    run files, listing every point in ACQUISITION order::

        {"schema_version": 1, "config_name": ..., "output_format": ...,
         "created": ..., "updated": ..., "config": {...},
         "points": [{"run_number": 1, "filename": ..., "timestamp": ...,
                     "alpha": ..., "beta": ..., "mach": ...,
                     "speed_value": ..., "speed_unit": ...,
                     "air_state": ..., "sweep_dir": ...}, ...]}

    A missing, unreadable, wrong-schema or malformed manifest is NOT an
    error: this returns ``{}`` and the caller falls back to per-file
    metadata.

    Parameters
    ----------
    directory : str
        Directory holding the run files

    Returns
    -------
    dict
        The parsed manifest, or ``{}``.
    """
    path = Path(directory) / MANIFEST_FILENAME
    if not path.is_file():
        return {}

    try:
        with open(path, 'r', encoding='utf-8') as fh:
            manifest = json.load(fh)
    except (OSError, ValueError) as exc:
        warnings.warn(f"Ignoring unreadable {MANIFEST_FILENAME} in "
                      f"'{path.parent.name}': {type(exc).__name__}: {exc}")
        return {}

    if not isinstance(manifest, dict):
        warnings.warn(f"Ignoring malformed {MANIFEST_FILENAME} in "
                      f"'{path.parent.name}': expected a JSON object")
        return {}
    if manifest.get('schema_version') != MANIFEST_SCHEMA_VERSION:
        warnings.warn(
            f"Ignoring {MANIFEST_FILENAME} in '{path.parent.name}': "
            f"schema_version {manifest.get('schema_version')!r}, "
            f"expected {MANIFEST_SCHEMA_VERSION}")
        return {}
    if not isinstance(manifest.get('points'), list):
        warnings.warn(f"Ignoring malformed {MANIFEST_FILENAME} in "
                      f"'{path.parent.name}': 'points' is not a list")
        return {}

    return manifest


def find_run_balance_cal(directory: str) -> Dict[str, Any]:
    """
    Resolve the run-local balance calibration a run directory carries.

    Freestream copies the active balance ``.vol`` into the run directory
    at run start and records the hand-off in ``manifest.json`` as a
    top-level ``"balance_cal"`` object::

        {"balance_cal": {"vol_file": "50lb 2026_07_24.vol",
                         "vol_source": "C:/.../CalFiles/...",
                         "cal_type": "Linear", "balance_config": "Moment",
                         "balance_type": "internal",
                         "balance_serial": "..."}, ...}

    Resolution order:

    1. the manifest's ``balance_cal.vol_file`` when that file exists in
       the directory (its recorded ``cal_type`` etc. ride along);
    2. else the directory's SINGLE ``*.vol`` file (``cal_type`` falls
       back to the manifest ``config`` block when one is recorded).

    Two or more ``.vol`` files with no manifest entry are ambiguous and
    resolve to nothing — never guess a calibration.

    Parameters
    ----------
    directory : str
        Directory holding the run files

    Returns
    -------
    dict
        ``{'vol_path': str, 'cal_type': str, ...}`` with the balance
        metadata recorded for the run, or ``{}`` when the directory
        carries no unambiguous run-local calibration.
    """
    data_dir = Path(directory)
    manifest = read_run_manifest(data_dir)

    entry = manifest.get('balance_cal')
    if isinstance(entry, dict):
        name = str(entry.get('vol_file') or '')
        path = data_dir / name if name else None
        if path is not None and path.is_file():
            resolved = {k: v for k, v in entry.items()
                        if k != 'vol_file' and v not in (None, '')}
            resolved['vol_path'] = str(path)
            return resolved

    vols = sorted(data_dir.glob('*.vol'))
    if len(vols) != 1:
        return {}
    config = manifest.get('config')
    cal_type = ''
    if isinstance(config, dict):
        cal_type = str(config.get('cal_type') or '')
    resolved = {'vol_path': str(vols[0])}
    if cal_type:
        resolved['cal_type'] = cal_type
    return resolved


def _file_info_from_manifest_point(path: Path, point: Dict[str, Any],
                                   manifest: Dict[str, Any]) -> 'FileInfo':
    """
    Fill a point's :class:`FileInfo` from the manifest where the FILE is
    silent.

    Precedence is file metadata -> manifest -> filename: the point's own
    record is the most specific, the manifest is next (it supplies
    everything for a legacy or unreadable file), and the filename is the
    last resort. A manifest value of ``null`` means genuinely unknown
    and leaves the field as parsed.
    """
    info = parse_run_file(str(path))
    meta = read_run_metadata(str(path))

    if not _meta_str(meta, 'config_name'):
        config_name = _meta_str(manifest, 'config_name')
        if config_name:
            info.configuration = config_name
    if not _meta_str(meta, 'air_state'):
        air_state = _normalize_air_state(point.get('air_state'))
        if air_state is not None:
            info.air_state = air_state
    if _meta_float(meta, 'alpha') is None:
        alpha = _meta_float(point, 'alpha')
        if alpha is not None:
            info.alpha = alpha
    if _meta_float(meta, 'beta') is None:
        beta = _meta_float(point, 'beta')
        if beta is not None:
            info.beta = beta
    if _meta_float(meta, 'speed_value') is None:
        speed = _meta_float(point, 'speed_value')
        if speed is not None:
            info.speed = speed
    if not _meta_str(meta, 'speed_unit'):
        speed_unit = _meta_str(point, 'speed_unit')
        if speed_unit is not None:
            info.speed_unit = speed_unit.lower()
    if _meta_int(meta, 'run_number') is None:
        run_number = _meta_int(point, 'run_number')
        if run_number is not None:
            info.run_number = run_number
    if not _meta_str(meta, 'sweep_dir'):
        sweep_dir = (_meta_str(point, 'sweep_dir') or '').lower()
        if sweep_dir in ('up', 'dn'):
            info.sweep_dir = sweep_dir

    return info


def scan_run_directory(directory: str,
                       recursive: bool = False) -> Tuple[List['FileInfo'],
                                                         List[str]]:
    """
    Index a run directory metadata-first, honoring ``manifest.json``.

    When the directory carries a valid manifest it is the AUTHORITATIVE
    index: it fixes the point count and the ACQUISITION order, and it
    supplies per-point fields for files whose own metadata is missing.
    Disagreements between the manifest and what is on disk are REPORTED
    rather than silently resolved in favor of either side — files the
    manifest does not list are appended after the listed ones, and
    listed points with no file on disk are dropped from the index.

    Without a manifest every run file is parsed with
    :func:`parse_run_file` and returned in sorted path order.

    Parameters
    ----------
    directory : str
        Directory holding the run files
    recursive : bool
        Whether to search subdirectories

    Returns
    -------
    tuple
        ``(file_infos, mismatches)`` — the indexed points and a list of
        human-readable mismatch messages (empty when consistent).
    """
    data_dir = Path(directory)
    files = find_run_files(data_dir, recursive=recursive)
    manifest = read_run_manifest(data_dir)
    if not manifest:
        return [parse_run_file(str(f)) for f in files], []

    by_name = {f.name: f for f in files}
    points = [p for p in manifest.get('points', []) if isinstance(p, dict)]

    infos: List['FileInfo'] = []
    listed = set()
    missing: List[str] = []
    for point in points:
        name = str(point.get('filename') or '')
        path = by_name.get(name)
        if path is None:
            missing.append(name or '<unnamed point>')
            continue
        listed.add(name)
        infos.append(_file_info_from_manifest_point(path, point, manifest))

    extra = [f for f in files if f.name not in listed]
    infos.extend(parse_run_file(str(f)) for f in extra)

    mismatches: List[str] = []
    if len(points) != len(files):
        mismatches.append(
            f"{MANIFEST_FILENAME} lists {len(points)} point(s) but "
            f"{len(files)} run file(s) are present")
    if missing:
        mismatches.append(
            f"{len(missing)} manifest point(s) have no run file on disk: "
            + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))
    if extra:
        mismatches.append(
            f"{len(extra)} run file(s) are not listed in "
            f"{MANIFEST_FILENAME}: "
            + ", ".join(f.name for f in extra[:5])
            + (" ..." if len(extra) > 5 else ""))

    for message in mismatches:
        warnings.warn(f"{data_dir.name}: {message}")

    return infos, mismatches


def read_run_config(directory: str) -> Dict[str, Any]:
    """
    The measurement config recorded for a run directory.

    Prefers the ``config`` block of ``manifest.json`` (one small read),
    falling back to the first run file that carries a
    ``meta.config_json``.

    Parameters
    ----------
    directory : str
        Directory holding the run files

    Returns
    -------
    dict
        The recorded config, or ``{}`` for a legacy directory that
        records none.
    """
    config = read_run_manifest(directory).get('config')
    if isinstance(config, dict) and config:
        return config

    for f in find_run_files(directory):
        config = read_run_metadata(str(f)).get('config')
        if isinstance(config, dict) and config:
            return config

    return {}


# Reference-dimension config keys, in preference order, paired with the
# model-geometry field each seeds. Freestream's config carries TWO
# overlapping sets (Sref/cref/bref and ref_area/ref_chord/ref_span) and in
# a real run one set holds 0.0 while the other holds the 1.0 placeholders,
# so the VALUE decides which is real, not the key.
_REFERENCE_CONFIG_KEYS = (
    ('ref_area', ('Sref', 'ref_area')),
    ('mac', ('cref', 'ref_chord')),
    ('span', ('bref', 'ref_span')),
)

# A reference dimension of exactly 1.0 is the unset placeholder; seeding
# it would pass a placeholder off as a measured dimension.
_REFERENCE_PLACEHOLDER = 1.0


def reference_geometry_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Model geometry and balance settings a recorded config can SEED.

    Only genuinely recorded values come back: a reference dimension is
    taken when it is finite, nonzero and not the 1.0 placeholder, and the
    MRC only when at least one component is nonzero. Anything the config
    does not really carry is OMITTED so the caller leaves the operator's
    current value alone — a silently substituted 1.0 is indistinguishable
    from a real reference dimension once it reaches the reduction.

    Parameters
    ----------
    config : dict
        A parsed ``meta.config_json`` / manifest ``config`` block

    Returns
    -------
    dict
        Any of ``'ref_area'``, ``'mac'``, ``'span'`` (floats), ``'mrc'``
        (3-list) and ``'balance_config'`` ('Force' or 'Moment').
    """
    seed: Dict[str, Any] = {}
    if not isinstance(config, dict):
        return seed

    for geo_key, config_keys in _REFERENCE_CONFIG_KEYS:
        for config_key in config_keys:
            value = _meta_float(config, config_key)
            if (value is not None and abs(value) > 1e-12
                    and value != _REFERENCE_PLACEHOLDER):
                seed[geo_key] = value
                break

    mrc = [_meta_float(config, f'MRC_{axis}') or 0.0
           for axis in ('x', 'y', 'z')]
    if any(abs(v) > 1e-12 for v in mrc):
        seed['mrc'] = mrc

    balance_config = (_meta_str(config, 'balance_config') or '').lower()
    if balance_config in ('force', 'moment'):
        seed['balance_config'] = balance_config.capitalize()

    return seed


def speed_condition_key(value: float, unit: Optional[str]) -> str:
    """
    Build the per-condition dict key for a (value, unit) speed setting.

    Air-off (speed 0 / mach 0) collapses to ``'AirOff'``; each distinct
    nonzero speed becomes its own condition key, e.g. ``'Hz_30.0'`` or
    ``'mach_0.3'``, so a velocity sweep classifies into its distinct
    speeds instead of collapsing to a single condition.
    """
    if value is None or abs(value) < 1e-6:
        return 'AirOff'
    tag = (unit or 'speed').replace('/', '').replace(' ', '')
    return f'{tag}_{value:g}'


def classify_files_by_condition(files: list) -> Dict[str, list]:
    """
    Classify data files by test condition, keyed by the speed dimension.

    The air state comes from :func:`parse_run_file`, i.e. from
    ``meta.run.air_state`` when the file records it and from the filename
    otherwise. That distinction matters: a tare taken with the fan
    RUNNING is an air-off point at a nonzero speed, which the old
    speed-only inference misclassified as air-on. Legacy TDMS runs carry
    an explicit ``AirOn``/``AirOff`` substring and Freestream runs
    without metadata encode the condition in the
    ``{Hz|ftps|mps|RPM|mach}_<value>`` token (speed 0 -> tare/air-off);
    both still work, unchanged.

    The returned dict always carries the ``'AirOn'`` / ``'AirOff'`` keys.
    In addition, each distinct nonzero speed of an air-on file gets its
    own condition key (e.g. ``'Hz_30.0'``) listing just that speed's
    files, so a velocity sweep surfaces as its distinct speeds rather
    than a single collapsed condition. Files with no cue at all are left
    unclassified.

    Parameters
    ----------
    files : list
        List of file paths

    Returns
    -------
    dict
        Dictionary with 'AirOn' / 'AirOff' keys plus one key per distinct
        nonzero speed condition.
    """
    classified: Dict[str, list] = {'AirOn': [], 'AirOff': []}

    for f in files:
        info = parse_run_file(str(f))
        if info.air_state == 'AirOff':
            classified['AirOff'].append(f)
            continue
        if info.air_state != 'AirOn':
            continue                                # no cue at all -> skip

        classified['AirOn'].append(f)

        value, unit = info.speed, info.speed_unit
        if value is None:                           # fall back to bare mach
            mach = extract_mach_from_filename(str(f))
            if mach is not None:
                value, unit = mach, 'mach'
        if value is not None and abs(value) > 1e-6:
            # A distinct nonzero speed is its own condition
            classified.setdefault(speed_condition_key(value, unit),
                                  []).append(f)

    return classified


def extract_sort_key_from_filename(filepath: str) -> Tuple[float, float,
                                                           float]:
    """
    Organization/sort key for a run file: ``(alpha, beta, speed_value)``.

    Speed is a first-class sweep dimension alongside alpha and beta, so a
    directory of runs organizes alpha -> beta -> speed. The speed value
    comes from the ``{Hz|ftps|mps|RPM|mach}_<value>`` token, falling back
    to the legacy bare ``mach`` token, then 0.0 when neither is present.
    """
    alpha, beta = extract_alpha_beta_from_filename(filepath)
    value, _ = extract_speed_from_filename(filepath)
    if value is None:
        mach = extract_mach_from_filename(filepath)
        value = mach if mach is not None else 0.0
    return (alpha, beta, value)


def extract_mach_from_filename(filepath: str) -> Optional[float]:
    """
    Extract the Mach value from a filename ``mach_<value>`` token.

    Expected format: ``..._mach_0.30.h5`` (mirrors
    :func:`extract_alpha_beta_from_filename`).

    Parameters
    ----------
    filepath : str
        File path to parse

    Returns
    -------
    float or None
        The parsed Mach number, or None when no ``mach_`` token is present.
    """
    import re

    filename = Path(filepath).stem
    mach_match = re.search(r'mach[_\s]*(-?\d+\.?\d*)', filename, re.IGNORECASE)
    return float(mach_match.group(1)) if mach_match else None


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
        # Freestream run files are named run_<NNNN>_alpha_..._mach_...;
        # the run counter is not a configuration, so map it to Unknown
        # to keep all runs of a directory grouped together.
        if re.fullmatch(r'run_?\d+', configuration, re.IGNORECASE):
            return 'Unknown'
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

    # Freestream run files drop the AirOn/AirOff token and encode the
    # condition in the SPEED token instead (speed 0 -> tare/air-off),
    # mirroring classify_files_by_condition. This covers EVERY selectable
    # speed unit — Hz/ftps/mps/RPM as well as the legacy mach token
    # (extract_speed_from_filename parses all of them), so a non-Mach
    # velocity sweep is no longer left 'Unknown' (which starved the
    # reducer of AirOn files and detected zero configurations).
    speed_value, _ = extract_speed_from_filename(filepath)
    if speed_value is not None:
        return 'AirOff' if abs(speed_value) < 1e-6 else 'AirOn'

    # Last resort: the canonical mach token directly.
    mach = extract_mach_from_filename(filepath)
    if mach is not None:
        return 'AirOff' if abs(mach) < 1e-6 else 'AirOn'

    return 'Unknown'


def extract_run_number_from_filename(filepath: str) -> Optional[int]:
    """
    Extract the acquisition run number from a ``run_<NNNN>`` filename token.

    Expected format: ``run_0006_alpha_2.0_beta_0.0_Hz_20.0.mat`` -> 6.

    Parameters
    ----------
    filepath : str
        File path to parse

    Returns
    -------
    int or None
        The parsed run number, or None when no ``run_`` token leads the
        filename.
    """
    import re

    filename = Path(filepath).stem
    match = re.match(r'^run[_\s]*(\d+)', filename, re.IGNORECASE)
    return int(match.group(1)) if match else None


def extract_sweep_dir_from_filename(filepath: str) -> str:
    """
    Extract the hysteresis leg from a trailing ``_up`` / ``_dn`` token.

    Freestream tags the RETURN leg of a hysteresis sweep so the up and
    down legs stay separable downstream.

    Parameters
    ----------
    filepath : str
        File path to parse

    Returns
    -------
    str
        'up', 'dn', or '' when the filename carries no leg token.
    """
    import re

    filename = Path(filepath).stem
    match = re.search(r'_(up|dn)$', filename, re.IGNORECASE)
    return match.group(1).lower() if match else ''


@dataclass
class FileInfo:
    """Information about a single run file.

    Read from the file's own metadata when it records any (see
    :func:`parse_run_file`), and parsed from the filename otherwise (see
    :func:`parse_tdms_filename`, the legacy TDMS path).
    """
    filepath: Path
    configuration: str
    air_state: str
    alpha: float
    beta: float
    # The tunnel-speed SETPOINT (the TARGET value — bulletproof for
    # grouping, since measured speed can jitter or collapse), from
    # meta.run.speed_value or the filename token. ``speed`` is None when
    # neither is present; ``speed_unit`` is one of
    # 'hz'/'ft/s'/'m/s'/'rpm'/'mach'.
    speed: Optional[float] = None
    speed_unit: Optional[str] = None
    # Acquisition run number (meta.run.run_number, else the run_<NNNN>
    # filename token); None when the file carries neither.
    run_number: Optional[int] = None
    # Hysteresis leg: 'up' / 'dn' on a return-sweep point, '' otherwise.
    sweep_dir: str = ""


def parse_tdms_filename(filepath: str) -> FileInfo:
    """
    Parse all information from a TDMS filename.

    The legacy filename-only path, kept for TDMS runs that carry no
    in-file metadata. New callers want :func:`parse_run_file`, which
    reads the file's own record first and falls back to this.

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
    speed, speed_unit = extract_speed_from_filename(str(filepath))

    return FileInfo(
        filepath=filepath,
        configuration=config,
        air_state=air_state,
        alpha=alpha,
        beta=beta,
        speed=speed,
        speed_unit=speed_unit,
        run_number=extract_run_number_from_filename(str(filepath)),
        sweep_dir=extract_sweep_dir_from_filename(str(filepath)),
    )


def parse_run_file(filepath: str) -> FileInfo:
    """
    Parse a run file METADATA-FIRST, falling back to the filename per FIELD.

    The file itself authoritatively records what the filename only
    encodes, so each field is taken from the ``meta.run`` attribute that
    :func:`read_run_metadata` returns, and from the matching
    ``extract_*_from_filename`` parser when that key is absent:

    * configuration <- ``config_name``
    * air_state     <- ``air_state``
    * alpha / beta  <- ``alpha`` / ``beta``
    * speed / unit  <- ``speed_value`` / ``speed_unit``
    * run_number    <- ``run_number``
    * sweep_dir     <- ``sweep_dir``

    Legacy TDMS runs carry no metadata at all and so behave exactly as
    they did. For a Freestream run the difference is substantive: the
    recorded air state no longer misreads a fan-running tare as an
    air-on point, and ``config_name`` separates two model configurations
    recorded into one folder — which the ``run_<NNNN>`` filename counter
    cannot do, since it collapses every run to 'Unknown'.

    Parameters
    ----------
    filepath : str
        Path to the run file

    Returns
    -------
    FileInfo
        Parsed file information
    """
    filepath = Path(filepath)
    meta = read_run_metadata(str(filepath))

    configuration = _meta_str(meta, 'config_name')
    if configuration is None:
        configuration = extract_configuration_from_filename(str(filepath))

    air_state = _normalize_air_state(meta.get('air_state'))
    if air_state is None:
        air_state = extract_air_state_from_filename(str(filepath))

    alpha = _meta_float(meta, 'alpha')
    beta = _meta_float(meta, 'beta')
    if alpha is None or beta is None:
        file_alpha, file_beta = extract_alpha_beta_from_filename(str(filepath))
        alpha = file_alpha if alpha is None else alpha
        beta = file_beta if beta is None else beta

    speed = _meta_float(meta, 'speed_value')
    speed_unit = _meta_str(meta, 'speed_unit')
    if speed is None or speed_unit is None:
        file_speed, file_unit = extract_speed_from_filename(str(filepath))
        speed = file_speed if speed is None else speed
        speed_unit = file_unit if speed_unit is None else speed_unit
    if speed_unit is not None:
        speed_unit = speed_unit.lower()

    run_number = _meta_int(meta, 'run_number')
    if run_number is None:
        run_number = extract_run_number_from_filename(str(filepath))

    sweep_dir = _meta_str(meta, 'sweep_dir')
    if sweep_dir is None:
        sweep_dir = extract_sweep_dir_from_filename(str(filepath))
    sweep_dir = sweep_dir.lower()
    if sweep_dir not in ('up', 'dn'):
        sweep_dir = ''

    return FileInfo(
        filepath=filepath,
        configuration=configuration,
        air_state=air_state,
        alpha=alpha,
        beta=beta,
        speed=speed,
        speed_unit=speed_unit,
        run_number=run_number,
        sweep_dir=sweep_dir,
    )


def _speed_group_suffix(info: 'FileInfo') -> Optional[str]:
    """The filename speed-step key for a file (e.g. 'Hz_20', 'mach_0.30'),
    or None when the file carries no resolvable tunnel-speed token. Uses the
    filename TARGET setpoint — the bulletproof grouping cue Casey asked for."""
    if info.speed is None or info.speed_unit is None:
        return None
    return speed_condition_key(info.speed, info.speed_unit)


def group_files_by_configuration(files: list) -> Dict[str, Dict[str, list]]:
    """
    Group run files by configuration and air state.

    Each file is indexed with :func:`parse_run_file`, so the
    configuration comes from ``meta.run.config_name`` when the file
    records one: two model configurations recorded into a single folder
    separate correctly instead of collapsing into one 'Unknown' group.

    A velocity/Mach sweep records every speed step into ONE folder. ALL of a
    configuration's speed steps stay in a SINGLE group (one case): the speed
    steps are kept DISTINCT downstream as separate Mach values (per-point,
    from the reduced tunnel conditions) that the GUI Mach filter selects
    between, and the MATLAB export lays out as an (alpha, beta, mach) 3-D
    array. Grouping is NOT split by speed — Casey wants one config per
    configuration, not one per speed increment.

    Files are ordered alpha -> beta -> speed setpoint within each group (the
    filename speed token is the robust ordering cue).

    Parameters
    ----------
    files : list
        List of file paths, or of already-parsed :class:`FileInfo`
        objects (as :func:`scan_run_directory` returns, so a
        manifest-indexed directory is not re-parsed here)

    Returns
    -------
    dict
        Nested dictionary: {config_name: {'AirOn': [files], 'AirOff': [files],
        'Unknown': []}}
    """
    grouped: Dict[str, Dict[str, list]] = {}
    for f in files:
        info = f if isinstance(f, FileInfo) else parse_run_file(str(f))
        cfg = grouped.setdefault(
            info.configuration, {'AirOn': [], 'AirOff': [], 'Unknown': []})
        cfg.setdefault(info.air_state, []).append(info)

    # Sort files within each group by alpha, then beta, then speed setpoint.
    for config in grouped:
        for air_state in grouped[config]:
            grouped[config][air_state].sort(
                key=lambda x: (x.alpha, x.beta,
                               x.speed if x.speed is not None else 0.0))

    return grouped


def run_balance_type(directory: str) -> str:
    """Balance type recorded by the runs in a directory.

    Returns ``'external'`` when the runs carry resolved loads from an
    external balance (the ATE), ``'internal'`` when they carry bridge
    volts needing a ``.vol``, or ``''`` when nothing says.

    Freestream stamps ``balance_type`` into every run file's root
    metadata, and the directory manifest repeats it when a calibration
    was staged. Metadata is read without unpacking channel arrays, and
    the first file that answers wins: a directory mixing balance types
    is not a thing.
    """
    try:
        manifest = read_run_manifest(directory)
    except Exception:                                  # noqa: BLE001
        manifest = {}
    entry = (manifest or {}).get('balance_cal') or {}
    btype = str(entry.get('balance_type') or '').strip().lower()
    if btype in ('external', 'internal'):
        return btype

    for path in find_run_files(directory)[:8]:
        try:
            meta = read_run_metadata(str(path))
        except Exception:                              # noqa: BLE001
            continue
        btype = str(meta.get('balance_type') or '').strip().lower()
        if btype in ('external', 'internal'):
            return btype
    return ''


def run_span_config(directory: str) -> str:
    """Model-span configuration recorded by the runs in a directory.

    Returns ``'half'`` for a semispan model on the turntable,
    ``'full'`` for a full-span model on the incidence strut, or ``''``
    when nothing says.

    The marker decides which resolution algorithm an external-balance
    run needs (see
    :func:`~.external_balance.resolve_external_wrf`), so the GUI shows
    it rather than letting a silent default pick for the operator.
    Freestream stamps ``span_config`` into every run file's root
    metadata from the positioner's own mapping.
    """
    for path in find_run_files(directory)[:8]:
        try:
            meta = read_run_metadata(str(path))
        except Exception:                              # noqa: BLE001
            continue
        span = str(meta.get('span_config') or '').strip().lower()
        if span in ('full', 'half'):
            return span
    return ''
