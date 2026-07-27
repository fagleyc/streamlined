"""Data-load failure reporting tests.

ProcessingWorker used to catch every reduction failure (and every
missing balance calibration) and return _create_case_from_files()
output instead: analytic CL = 0.1 + 0.11*alpha with no tunnel
conditions.  The load then reported success, so fabricated numbers were
indistinguishable from a real reduction.  Failures must reach the user.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

pytest.importorskip("PyQt6")

from utils.gui.controllers.data_controller import (  # noqa: E402
    BACKEND_AVAILABLE, FileInfoSimple, ProcessingWorker,
)

pytestmark = pytest.mark.skipif(
    not BACKEND_AVAILABLE, reason='windtunnel backend not importable')

GEOMETRY = {'mac': 1.0, 'ref_area': 1.0, 'span': 1.0,
            'mrc': [0.0, 0.0, 0.0], 'units': 'IPS'}
SETTINGS = {'facility': 'LSWT', 'balance_config': 'Force',
            'cal_type': 'Linear'}


def _worker(directory: str) -> ProcessingWorker:
    return ProcessingWorker(
        directories=[directory], balance_cal=None, pressure_cal=None,
        balance_cal_file=None, pressure_cal_file=None,
        geometry=GEOMETRY, settings=SETTINGS)


def _junk_run(tmp_path: Path) -> Path:
    """A file named like a run file whose contents cannot be read."""
    path = tmp_path / "AirOn_Model_Alpha_0.0_Beta_0.0.mat"
    path.write_bytes(b"not a MATLAB file")
    return path


class TestBackendFailuresPropagate:

    def test_unreadable_run_file_raises(self, tmp_path):
        path = _junk_run(tmp_path)
        info = FileInfoSimple(filepath=path, configuration='Unknown',
                              air_state='AirOn', alpha=0.0, beta=0.0)
        worker = _worker(str(tmp_path))
        with pytest.raises(RuntimeError):
            worker._process_with_backend(
                'Case', 'Unknown', [info], [], [0.0], [0.0], 1, 1, 0,
                str(tmp_path))

    def test_demo_data_is_not_substituted(self, tmp_path):
        """The backend path must never reach the demo-case builder."""
        path = _junk_run(tmp_path)
        info = FileInfoSimple(filepath=path, configuration='Unknown',
                              air_state='AirOn', alpha=0.0, beta=0.0)
        worker = _worker(str(tmp_path))

        def _fail(*args, **kwargs):
            raise AssertionError("demo data must not be fabricated")

        worker._create_case_from_files = _fail
        with pytest.raises(RuntimeError):
            worker._process_with_backend(
                'Case', 'Unknown', [info], [], [0.0], [0.0], 1, 1, 0,
                str(tmp_path))


class TestWorkerReportsFailures:

    def test_error_emitted_and_no_case_ready(self, tmp_path):
        _junk_run(tmp_path)
        worker = _worker(str(tmp_path))

        errors = []
        cases = []
        finished = []
        worker.signals.error.connect(
            lambda title, msg: errors.append((title, msg)))
        worker.signals.case_ready.connect(cases.append)
        worker.signals.finished.connect(finished.append)

        worker.run()

        assert cases == [], "no case may be emitted for an unreadable run"
        assert errors, "the failure must be reported to the user"
        assert finished == [], "a total failure must not report success"
