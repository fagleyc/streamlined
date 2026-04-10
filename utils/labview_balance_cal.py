"""
LabVIEW Balance Calibration Interface
======================================

Standalone module for parsing .vol calibration files and computing
balance calibration coefficients. Designed to be called from LabVIEW
via the Python Node or command line.

Usage from LabVIEW Python Node:
    import labview_balance_cal
    coeffs, r_squared, bias, channels, distances = labview_balance_cal.get_calibration(
        vol_filepath, cal_type='Linear'
    )

Usage from command line:
    python labview_balance_cal.py <vol_filepath> [--cal_type Linear|Quadratic|Cubic] [--output json|csv]

Author: C. Fagley
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path so we can import the existing modules
_PROJECT_ROOT = str(Path(__file__).parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.windtunnel.calibration import read_vol_file, calc_coeffs


def get_calibration(vol_filepath: str,
                    cal_type: str = 'Linear') -> tuple:
    """
    Parse a .vol calibration file and compute balance calibration coefficients.

    This is the primary entry point for LabVIEW Python Node calls.

    Parameters
    ----------
    vol_filepath : str
        Full path to the .vol calibration file.
    cal_type : str
        Calibration fit type: 'Linear', 'Quadratic', or 'Cubic'.
        Default is 'Linear'.

    Returns
    -------
    coeffs : list of list of float
        Calibration coefficient matrix [n_terms x 6].
        For Linear: n_terms = 6 (one per voltage channel).
        For Quadratic: n_terms = 12 (6 linear + 6 squared).
        For Cubic: n_terms = 18 (6 linear + 6 squared + 6 cubed).
        Each column corresponds to a balance element:
        [N1, N2, Y1, Y2, Axial, Roll] (Force config) or
        [AftPitch, AftYaw, FwdPitch, FwdYaw, Axial, Roll] (Moment config).
    r_squared : list of float
        R-squared goodness-of-fit for each of the 6 channels.
    bias : list of float
        RMSE bias error for each of the 6 channels.
    channels : list of str
        Names of the 6 balance channels.
    distances : dict
        Balance distance measurements from the calibration file.
    """
    cal = read_vol_file(vol_filepath)
    cal = calc_coeffs(cal, cal_type)

    return (
        cal.coeffs.tolist(),
        cal.r_squared.tolist(),
        cal.bias.tolist(),
        cal.force_channels,
        dict(cal.distances.values),
    )


def get_calibration_detailed(vol_filepath: str,
                             cal_type: str = 'Linear') -> dict:
    """
    Parse a .vol file and return all calibration data as a dictionary.

    Useful for JSON serialization or detailed inspection.

    Parameters
    ----------
    vol_filepath : str
        Full path to the .vol calibration file.
    cal_type : str
        Calibration fit type: 'Linear', 'Quadratic', or 'Cubic'.

    Returns
    -------
    dict
        Complete calibration results including:
        - coeffs: coefficient matrix [n_terms x 6]
        - r_squared: goodness of fit per channel
        - bias: RMSE per channel
        - channels: channel names
        - distances: balance arm distances
        - max_loads: maximum rated loads and units
        - balance_info: balance description metadata
        - force_matrix: applied force matrix from cal file
        - volts_matrix: measured voltage matrix from cal file
        - force_est: estimated forces from fit
        - cal_type: fit type used
    """
    cal = read_vol_file(vol_filepath)
    cal = calc_coeffs(cal, cal_type)

    return {
        'coeffs': cal.coeffs.tolist(),
        'r_squared': cal.r_squared.tolist(),
        'bias': cal.bias.tolist(),
        'channels': cal.force_channels,
        'distances': dict(cal.distances.values),
        'max_loads': {
            'values': cal.max_loads.values,
            'units': cal.max_loads.units,
        },
        'balance_info': {
            'type': cal.description.balance_type,
            'serial_number': cal.description.serial_number,
            'outer_diameter': cal.description.outer_diameter,
        },
        'force_matrix': cal.force.tolist(),
        'volts_matrix': cal.volts.tolist(),
        'force_est': cal.force_est.tolist(),
        'cal_type': cal.cal_type,
    }


def apply_calibration(vol_filepath: str,
                      raw_voltages: list,
                      excitation: float = 1.0,
                      cal_type: str = 'Linear') -> list:
    """
    Apply balance calibration to raw voltage readings.

    Converts raw balance voltages to engineering-unit forces/moments
    (balance elements) using the calibration coefficients.

    Parameters
    ----------
    vol_filepath : str
        Full path to the .vol calibration file.
    raw_voltages : list of float or list of list of float
        Raw voltage readings [N1, N2, Y1, Y2, Axial, Roll].
        - Single sample: [v1, v2, v3, v4, v5, v6]
        - Multiple samples: [[v1,v2,v3,v4,v5,v6], ...]
    excitation : float
        Excitation voltage. Raw voltages are divided by this value.
        Default is 1.0 (no normalization, assumes already normalized).
    cal_type : str
        Calibration fit type: 'Linear', 'Quadratic', or 'Cubic'.

    Returns
    -------
    elements : list of list of float
        Calibrated forces/moments [n_samples x 6].
        Columns: [N1, N2, Y1, Y2, Axial, Roll] in engineering units.
    """
    from utils.windtunnel.calibration import form_higher_order_terms

    cal = read_vol_file(vol_filepath)
    cal = calc_coeffs(cal, cal_type)

    v = np.atleast_2d(raw_voltages).astype(float)
    v = v / excitation

    order_map = {'Linear': 1, 'Quadratic': 2, 'Cubic': 3}
    order = order_map.get(cal_type, 1)
    X = form_higher_order_terms(v, order)
    elements = X @ cal.coeffs

    return elements.tolist()


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Parse a .vol balance calibration file and compute coefficients.')
    parser.add_argument('vol_file', help='Path to the .vol calibration file')
    parser.add_argument('--cal_type', default='Linear',
                        choices=['Linear', 'Quadratic', 'Cubic'],
                        help='Calibration fit type (default: Linear)')
    parser.add_argument('--output', default='json', choices=['json', 'csv'],
                        help='Output format (default: json)')
    args = parser.parse_args()

    if not Path(args.vol_file).exists():
        print(f"Error: File not found: {args.vol_file}", file=sys.stderr)
        sys.exit(1)

    result = get_calibration_detailed(args.vol_file, args.cal_type)

    if args.output == 'json':
        print(json.dumps(result, indent=2))
    elif args.output == 'csv':
        coeffs = np.array(result['coeffs'])
        channels = result['channels']
        print(f"Calibration Type: {result['cal_type']}")
        print(f"Channels: {', '.join(channels)}")
        print()
        print("Coefficient Matrix:")
        header = ','.join(channels) if channels else ','.join(
            [f'Ch{i}' for i in range(coeffs.shape[1])])
        print(f"Term,{header}")
        for i, row in enumerate(coeffs):
            print(f"{i},{','.join(f'{v:.10e}' for v in row)}")
        print()
        print("R-squared:")
        print(','.join(f'{v:.8f}' for v in result['r_squared']))
        print()
        print("Bias (RMSE):")
        print(','.join(f'{v:.8e}' for v in result['bias']))
