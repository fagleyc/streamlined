"""
COE File Reader
===============

Parser for the legacy Reduce2 .COE file format.  Reads the header
sections and the [TEST RUN] table into a structured object.  Designed
to be the entry point for downstream post-processing (stability
derivative plots, blockage corrections, etc.) without any Excel /
macro dependency.

Format reference: see brandt/5gat_Atail_B0.COE and the format docs
in coe_writer.py.

Author: C. Fagley
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# The 35 columns in the [TEST RUN] table, in order.  Match the
# canonical field names emitted by coe_writer._format_row so reader
# and writer round-trip identically.
COE_COLUMNS = [
    'Alpha', 'Beta', 'Ma', 'Re',
    'T0Stil', 'Tinf', 'a', 'Vinf',
    'p0Stil', 'pInf', 'qInf', 'pDiff',
    'N1', 'N2', 'N',
    'Y1', 'Y2', 'Y',
    'Ax',
    'PiMom', 'YaMom', 'RoMom',
    'CN', 'CY', 'CAx', 'CAxBD',
    'CPiMom', 'CYaMom', 'CRoMom',
    'CLift', 'CDrag', 'LD',
    'CPiMomSh', 'CYaMomSh', 'CRoMomSh',
]


@dataclass
class COEHeader:
    """Header metadata parsed from a .COE file."""
    coe_filename: str = ''
    date: str = ''
    time: str = ''
    red_filename: str = ''
    comment: str = ''
    atm_pressure_psia: float = 0.0
    cal_voltage_file: str = ''
    cal_mcf_file: str = ''
    cal_date: str = ''
    balance_serial: str = ''
    dx1: float = 0.0
    dx2: float = 0.0
    dy1: float = 0.0
    dy2: float = 0.0
    ref_length_in: float = 1.0
    ref_area_in2: float = 1.0
    mac_in: float = 1.0
    span_in: float = 1.0
    mrc_x_in: float = 0.0
    mrc_z_in: float = 0.0
    alpha_offset_deg: float = 0.0


@dataclass
class COEData:
    """Full contents of a parsed .COE file."""
    filepath: str = ''
    header: COEHeader = field(default_factory=COEHeader)
    # data[column_name] -> 1-D np.ndarray, length n_rows
    data: Dict[str, np.ndarray] = field(default_factory=dict)
    column_names: List[str] = field(default_factory=lambda: list(COE_COLUMNS))

    @property
    def n_rows(self) -> int:
        if not self.data:
            return 0
        first = next(iter(self.data.values()))
        return len(first)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def get(self, key: str, default=None):
        return self.data.get(key, default)


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------

_KV_RE = re.compile(r'^\s*([^-]+?)\s*-->\s*(.*?)\s*$')


def _parse_kv(line: str) -> Optional[Tuple[str, str]]:
    """Return (key, value) if the line is a 'Key --> value' entry."""
    m = _KV_RE.match(line)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def _parse_number_with_units(s: str) -> float:
    """Parse '11.47 [psia]' -> 11.47 ; '1.500 [in]' -> 1.500"""
    s = s.strip()
    # Strip a bracketed unit suffix if present
    m = re.match(r'^\s*(-?\d+\.?\d*(?:[eE][+\-]?\d+)?)', s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return 0.0
    return 0.0


def read_coe_file(filepath: str) -> COEData:
    """
    Parse a Reduce2 .COE file into a COEData object.

    Parameters
    ----------
    filepath : str
        Path to the .COE file.

    Returns
    -------
    COEData
        Parsed file contents.  Raises ValueError on a malformed file
        (missing [TEST RUN] section or unparseable data rows).
    """
    fp = Path(filepath)
    if not fp.exists():
        raise FileNotFoundError(f'COE file not found: {filepath}')

    text = fp.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()

    coe = COEData(filepath=str(fp))
    section = 'preamble'
    comment_lines: List[str] = []
    data_lines: List[str] = []
    in_data_table = False
    data_table_skip_header_lines = 0

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            if in_data_table:
                # Blank lines inside data are tolerated but skipped
                continue
            continue

        stripped = line.strip()
        # True section header: starts and ends with brackets AND has no
        # commas (otherwise it's the units banner like "[ ° ], [ ° ]...")
        if (stripped.startswith('[') and stripped.endswith(']')
                and ',' not in stripped):
            section = stripped[1:-1].strip()
            if section.lower() in ('test run', 'testrun'):
                in_data_table = True
                # Next 3 lines are column-name banners, parse them
                data_table_skip_header_lines = 3
            else:
                in_data_table = False
            continue

        if in_data_table:
            if data_table_skip_header_lines > 0:
                data_table_skip_header_lines -= 1
                continue
            # Skip lines that look like banner repeats (no commas)
            if ',' not in line:
                continue
            # Real data row
            data_lines.append(line)
            continue

        # Header sections - parse key/value pairs
        if section == 'preamble':
            kv = _parse_kv(line)
            if kv is None:
                if line.strip() == '*****':
                    continue
                if line.lower().startswith('reduce'):
                    continue
                continue
            key, val = kv
            key_lower = key.lower()
            if 'coe data' in key_lower:
                coe.header.coe_filename = val
            elif key_lower == 'date':
                coe.header.date = val
            elif key_lower == 'time':
                coe.header.time = val
            elif 'red data' in key_lower:
                coe.header.red_filename = val
            continue

        if section.lower() == 'comment':
            comment_lines.append(line.strip())
            continue

        if section.lower() == 'conditions':
            kv = _parse_kv(line)
            if kv is None:
                continue
            key, val = kv
            if 'atmosphere' in key.lower():
                coe.header.atm_pressure_psia = (
                    _parse_number_with_units(val))
            continue

        if section.lower().startswith('force'):
            kv = _parse_kv(line)
            if kv is None:
                continue
            key, val = kv
            key_lower = key.lower()
            if 'calibration voltage' in key_lower:
                coe.header.cal_voltage_file = val
            elif 'moment calibration' in key_lower:
                coe.header.cal_mcf_file = val
            elif 'date' in key_lower:
                coe.header.cal_date = val
            elif 'serial' in key_lower:
                coe.header.balance_serial = val
            elif 'n1' in key_lower:
                coe.header.dx1 = _parse_number_with_units(val)
            elif 'n2' in key_lower:
                coe.header.dx2 = _parse_number_with_units(val)
            elif 'y1' in key_lower:
                coe.header.dy1 = _parse_number_with_units(val)
            elif 'y2' in key_lower:
                coe.header.dy2 = _parse_number_with_units(val)
            continue

        if section.lower() == 'reference length':
            kv = _parse_kv(line)
            if kv is None:
                continue
            key, val = kv
            v = _parse_number_with_units(val)
            key_lower = key.lower()
            if key_lower == 'referencelength':
                coe.header.ref_length_in = v
            elif key_lower == 'referencearea':
                coe.header.ref_area_in2 = v
            elif key_lower == 'meanareachord':
                coe.header.mac_in = v
            elif key_lower == 'spanwidth':
                coe.header.span_in = v
            elif key_lower == 'momentxshift':
                coe.header.mrc_x_in = v
            elif key_lower == 'momentzshift':
                coe.header.mrc_z_in = v
            continue

        if section.lower() == 'alpha effective':
            kv = _parse_kv(line)
            if kv is None:
                continue
            key, val = kv
            if 'offset' in key.lower():
                coe.header.alpha_offset_deg = _parse_number_with_units(val)
            continue

    coe.header.comment = '\n'.join(comment_lines)

    if not data_lines:
        raise ValueError(
            f"No data rows found in '{filepath}'. "
            "Verify the file contains a [TEST RUN] section with rows.")

    # Parse each data row
    parsed_rows: List[List[float]] = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < len(COE_COLUMNS):
            # Skip malformed rows but warn
            continue
        try:
            row = [float(parts[i]) for i in range(len(COE_COLUMNS))]
        except ValueError:
            continue
        parsed_rows.append(row)

    if not parsed_rows:
        raise ValueError(
            f"No numeric data rows could be parsed from '{filepath}'.")

    arr = np.array(parsed_rows, dtype=float)
    for i, name in enumerate(COE_COLUMNS):
        coe.data[name] = arr[:, i]

    return coe
