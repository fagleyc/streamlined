"""Balance calibration (.vol) reader regression tests.

Real .vol files in CalFiles/ come out of several different calibration
GUIs: most are comma delimited ASCII, but at least one is tab delimited
and another is UTF-16.  read_vol_file() used to raise on those, and the
GUI swallowed the exception and substituted demo data, so a bad parse
looked exactly like a successful load.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.windtunnel.calibration import read_vol_file  # noqa: E402


# Load, then 6 bridge volts, then excitation volts
_LOADS = [
    [0.0, 0.0, 8.6e-05, 8.0e-05, 3.7e-05, -7.8e-05, -2.4e-04, 5.0015],
    [6.435, 2.4e-04, 8.8e-05, 7.8e-05, 3.7e-05, -8.0e-05, -2.4e-04, 5.0015],
    [12.87, 4.8e-04, 9.2e-05, 7.7e-05, 3.6e-05, -8.1e-05, -2.5e-04, 5.0015],
]


def _vol_text(delim: str) -> str:
    """A minimal one-channel .vol file using the given column delimiter."""
    lines = [
        "Voltage Calibration File",
        "Date-->28-Feb-2023",
        "",
        "[Balance Description]",
        "Balance Type-->1-Force / 5-Moment",
        "Balance Serial Number-->94017",
        "Balance outer Diameter-->.75 in.",
        "",
        "[Maximal Balance Loads]",
        "Aft_Pitch-->75 in-lb",
        "",
        "[Distances]",
        "Distance from Aft_Pitch to center of Balance x1 in inches--> 1.287",
        "",
        "[Aft_Pitch pos]",
        f"Number of Loads --> {len(_LOADS)}",
        delim.join(["Load[in-lb]"] + [f"ch{i}[Volt]" for i in range(7)]),
    ]
    for row in _LOADS:
        lines.append(delim.join(f"{v:.8f}" for v in row))
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str, encoding: str = "latin-1") -> str:
    path.write_bytes(text.encode(encoding))
    return str(path)


class TestDelimiters:

    def test_comma_delimited(self, tmp_path):
        cal = read_vol_file(_write(tmp_path / "comma.vol", _vol_text(",")))
        assert cal.force_channels == ["Aft_Pitch"]
        assert cal.force.shape == (len(_LOADS), 6)
        np.testing.assert_allclose(
            cal.force[:, 0], [row[0] for row in _LOADS])

    def test_tab_delimited_matches_comma_delimited(self, tmp_path):
        comma = read_vol_file(_write(tmp_path / "comma.vol", _vol_text(",")))
        tabbed = read_vol_file(_write(tmp_path / "tab.vol", _vol_text("\t")))
        np.testing.assert_allclose(tabbed.force, comma.force)
        np.testing.assert_allclose(tabbed.volts, comma.volts)
        assert tabbed.force_channels == comma.force_channels

    def test_trailing_delimiter_is_ignored(self, tmp_path):
        text = "\n".join(ln + "\t" for ln in _vol_text("\t").splitlines())
        cal = read_vol_file(_write(tmp_path / "trailing.vol", text + "\n"))
        assert cal.force.shape == (len(_LOADS), 6)


class TestEncodings:

    def test_utf16_file(self, tmp_path):
        comma = read_vol_file(_write(tmp_path / "comma.vol", _vol_text(",")))
        wide = read_vol_file(_write(tmp_path / "utf16.vol",
                                    _vol_text("\t"), "utf-16"))
        np.testing.assert_allclose(wide.force, comma.force)
        assert wide.description.serial_number == "94017"

    def test_utf8_bom_file(self, tmp_path):
        cal = read_vol_file(_write(tmp_path / "bom.vol",
                                   _vol_text(","), "utf-8-sig"))
        assert cal.force_channels == ["Aft_Pitch"]


class TestInvalidFiles:

    def test_file_without_channels_raises(self, tmp_path):
        """An unparseable file must fail loudly, not return an empty cal."""
        path = tmp_path / "empty.vol"
        path.write_text("not a calibration file\n", encoding="latin-1")
        with pytest.raises(ValueError):
            read_vol_file(str(path))


class TestShippedCalFiles:
    """Every .vol shipped in CalFiles/ must parse."""

    @pytest.mark.parametrize(
        "name", sorted(p.name for p in (ROOT / "CalFiles").glob("*.vol"))
        or ["<none>"])
    def test_parses(self, name):
        if name == "<none>":
            pytest.skip("no CalFiles/*.vol in this checkout")
        cal = read_vol_file(str(ROOT / "CalFiles" / name))
        assert len(cal.force_channels) == 6
        assert cal.force.shape[1] == 6
        assert cal.force.shape[0] > 0
        assert np.all(np.isfinite(cal.volts))
