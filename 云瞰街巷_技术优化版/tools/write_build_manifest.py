"""Record exact local source hashes before building, and bind them to the EXE."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATHS = [
    'main.py', 'Cloudlane.spec', 'build.ps1', 'requirements.txt',
    'requirements-dev.txt', 'requirements-lock.txt',
    'tools/frozen_runtime.py', 'tools/validate_packaged.py',
    'tools/write_build_manifest.py',
]


def source_hashes() -> dict[str, str]:
    paths = sorted(SOURCE_PATHS + [p.relative_to(ROOT).as_posix() for p in (ROOT / 'traffic_app').glob('*.py')])
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['before', 'after'])
    args = parser.parse_args()
    destination = ROOT / 'build' / 'source_manifest.json'
    if args.phase == 'before':
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': source_hashes()}
    else:
        payload = json.loads(destination.read_text(encoding='utf-8'))
        if payload['source_sha256'] != source_hashes():
            raise RuntimeError('源码在构建过程中发生变化；请重新构建，不能沿用本次构建记录。')
        executable = ROOT / 'dist' / '云瞰街巷' / '云瞰街巷.exe'
        payload.update({'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                        'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
                        'model_sha256': hashlib.sha256((ROOT / 'models' / 'best.pt').read_bytes()).hexdigest(),
                        'sources_unchanged_during_build': True})
        destination = executable.parent / '构建信息.json'
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'构建记录：{destination}')


if __name__ == '__main__':
    main()
