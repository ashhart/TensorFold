"""Run the frozen model byte-parity helper with the compact candidate revision."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

HELPER_SHA256 = '9dbc83074647aa5741e7ad1424bd4e673a82d2a48dfc9fdf3fad6a33d36389fd'
BASE = 'f350cd3ea253a042e205746928271f9d8f71a82d'
CANDIDATE = '8d00b490f5ddf45721546e9b9816a083a46e5fa2'


class JsonOverride:
    """Override the candidate revision only in the frozen helper's environment."""
    applied = False
    dumps = staticmethod(json.dumps)

    @classmethod
    def loads(cls, data: str) -> object:
        value = json.loads(data)
        if isinstance(value, dict) and 'checkpoint_assets_sha256' in value and 'candidate' in value:
            if value['base'] != BASE:
                raise ValueError('unexpected base revision in frozen environment')
            value['candidate'] = CANDIDATE
            cls.applied = True
        return value


def main() -> int:
    """Write a new result without editing the frozen helper or earlier receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--zig', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    helper_path = Path(__file__).resolve().parent.parent / 'model/verify_template.py'
    if hashlib.sha256(helper_path.read_bytes()).hexdigest() != HELPER_SHA256:
        raise ValueError('frozen model parity helper differs')
    spec = importlib.util.spec_from_file_location('frozen_model_verify', helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load frozen model parity helper')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.json = JsonOverride
    with tempfile.TemporaryDirectory(prefix='compact-template-parity-') as tmp:
        output = Path(tmp) / 'oracle'
        saved = sys.argv
        try:
            sys.argv = ['verify_template.py', '--template', str(args.template.resolve()), '--zig', args.zig,
                        '--out', str(output)]
            status = helper.main()
        finally:
            sys.argv = saved
        if status != 0 or not JsonOverride.applied:
            raise RuntimeError('byte parity failed or candidate override was not applied')
        summary = json.loads((output / 'summary.json').read_text())
    summary.update({'base_revision': BASE, 'candidate_revision': CANDIDATE,
                    'helper_sha256': HELPER_SHA256, 'jinja_version': __import__('jinja2').__version__,
                    'zig_version': '0.17.0', 'provenance': 'not checked by this byte-parity test'})
    with args.out.open('x') as out:
        json.dump(summary, out, indent=2)
        out.write('\n')
    print('PASS: six Jinja byte comparisons per revision; provenance not checked')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
