"""Verify the solver runtime installed by Scripts/Setup.cmd."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
_WHEELS = {
    item['package']: item
    for item in json.loads((ROOT / 'runtime-artifacts.json').read_text(encoding='utf-8-sig'))['wheels']
    if item['package'] in ('numpy', 'scipy', 'manifold3d')
}
# The editor preflight uses these two names without installing anything.
EXPECTED = {name: (item['version'], item['sha256']) for name, item in _WHEELS.items()}


def installed_record(name):
    from setup_runtime import installed_wheel
    item = _WHEELS[name]
    if not installed_wheel(item):
        return None
    return {'version': item['version'], 'source_wheel_sha256': item['sha256']}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify installed dependencies without changes (the default).')
    parser.parse_args()
    from setup_runtime import ROOT, installed_wheel, manifest
    if sys.version_info[:2] != (3, 10) or Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
        raise RuntimeError('Use the Python 3.10 environment created by Scripts/Setup.cmd.')
    selected = [item for item in manifest()['wheels'] if item['package'] in ('numpy', 'scipy', 'manifold3d')]
    for item in selected:
        if not installed_wheel(item):
            raise RuntimeError('Missing solver dependency; run Scripts/Setup.cmd: ' + item['package'])
    import manifold3d
    from scipy.optimize import least_squares
    trial = least_squares(lambda x: x - 1., [0.], method='trf', loss='huber')
    if not trial.success or abs(trial.x[0] - 1.) > 1e-7 or not hasattr(manifold3d, 'Mesh64'):
        raise RuntimeError('Solver verification failed.')
    print(json.dumps({'status': 'ready', 'dependencies': selected,
                      'verification': 'Source checksums, imports and a scalar TRF/Huber problem; no video inference.'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
