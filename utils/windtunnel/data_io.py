"""
Data I/O Module
===============

Functions for reading wind tunnel data files (TDMS, HDF5 and MATLAB .mat
formats) and exporting processed data.
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


def _finalize_load_units(raw: 'RawData',
                         channel_units: Dict[str, Any]) -> None:
    """
    Record a ``load_units`` marker for external-balance files.

    Freestream's ATE_Balance channels carry a per-channel ``unit``
    attribute ('N' / 'N*m'); the downstream reduction chain works in
    lb / in-lb (deprecated/scripts/calc_coeffs.m 'External'), so the
    unit system is surfaced in ``raw.properties['load_units']`` for
    the reducers to convert on. Files without unit metadata get no
    marker (treated as already lb / in-lb, the legacy behavior).
    """
    if raw.properties.get('balance_type') != 'external':
        return
    if 'load_units' in raw.properties:
        return
    for ch in _EXTERNAL_UNIT_PROBE_CHANNELS:
        unit = channel_units.get(ch)
        if isinstance(unit, bytes):
            unit = unit.decode('utf-8', errors='replace')
        if isinstance(unit, str) and unit.strip():
            u = unit.strip().lower()
            raw.properties['load_units'] = (
                'N' if u.startswith('n') else 'lb')
            return


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
    slower channels are cubic-interpolated onto it with extrapolation,
    mirroring the historical read_tdms_file behavior.
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
            interp_func = interp1d(
                ch['time'], ch['data'],
                kind='cubic' if n >= 4 else 'linear',
                bounds_error=False,
                fill_value='extrapolate'
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
    the SI -> lb/in-lb conversion in
    :func:`~.external_balance.external_loads_to_ips`, and the speed
    markers (``speed_value`` / ``speed_unit`` / ``speed_setpoints``)
    carry the tunnel-speed sweep dimension through to the reducers.
    """
    for key in ('balance_type', 'load_units',
                'speed_value', 'speed_unit', 'speed_setpoints'):
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

    Legacy TDMS runs carry an explicit ``AirOn``/``AirOff`` substring in
    the filename and are classified by that (unchanged). Newer Freestream
    runs drop that token and instead encode the condition in the speed
    setting: the filename ``{Hz|ftps|mps|RPM|mach}_<value>`` token (or the
    legacy ``mach`` token). A zero speed (``Hz_0.0`` / ``mach_0.00``) is a
    tare/air-off point; any nonzero speed is air-on.

    The returned dict always carries the ``'AirOn'`` / ``'AirOff'`` keys
    (``'AirOn'`` collects every nonzero-speed file). In addition, each
    distinct nonzero speed gets its own condition key (e.g. ``'Hz_30.0'``)
    listing just that speed's files, so a velocity sweep surfaces as its
    distinct speeds rather than a single collapsed condition. Files with
    no cue at all are left unclassified.

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
        fname = str(f).lower()
        if 'airoff' in fname:                       # legacy token wins first
            classified['AirOff'].append(f)
            continue
        if 'airon' in fname:
            classified['AirOn'].append(f)
            continue

        value, unit = extract_speed_from_filename(str(f))
        if value is None:                           # fall back to bare mach
            mach = extract_mach_from_filename(str(f))
            if mach is not None:
                value, unit = mach, 'mach'
        if value is None:
            continue                                # neither cue -> skip

        if abs(value) < 1e-6:                       # speed 0 -> air off/tare
            classified['AirOff'].append(f)
        else:
            classified['AirOn'].append(f)
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
