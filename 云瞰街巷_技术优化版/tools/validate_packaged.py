"""Exercise the distributed EXE using the explicitly synthetic software fixture."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from write_build_manifest import source_hashes

ROOT = Path(__file__).resolve().parent.parent


def rows(path):
    assert path.read_bytes().startswith(b'\xef\xbb\xbf'), f'Missing BOM: {path}'
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu'], default='auto')
    args = parser.parse_args()
    executable = ROOT / 'dist' / '云瞰街巷' / '云瞰街巷.exe'
    fixture = ROOT / 'validation' / 'synthetic_photos_software_validation.avi'
    tag = 'gpu' if args.device == 'auto' else 'cpu'
    expected_device = 'cuda:0' if tag == 'gpu' else 'cpu'
    report_path = ROOT / 'validation' / f'packaged_{tag}.json'
    screenshot_path = ROOT / 'validation' / f'packaged_{tag}_ui.png'
    environment = os.environ.copy()
    environment['QT_QPA_PLATFORM'] = 'offscreen'
    environment['YOLO_AUTOINSTALL'] = 'false'
    command = [str(executable), '--smoke-test', str(fixture), '--device', args.device,
               '--max-frames', '12', '--data-root', str(ROOT / 'validation' / f'packaged_{tag}_data'),
               '--report', str(report_path), '--screenshot', str(screenshot_path)]
    result = subprocess.run(command, cwd=ROOT, env=environment, timeout=240)
    assert result.returncode == 0, f'Packaged EXE failed: {result.returncode}'
    summary = json.loads(report_path.read_text(encoding='utf-8'))
    session = Path(summary['session_dir'])
    saved = json.loads((session / 'summary.json').read_text(encoding='utf-8'))
    metadata = json.loads((session / 'session.json').read_text(encoding='utf-8'))
    frames, detections, events = [rows(session / name) for name in ('frames.csv', 'detections.csv', 'events.csv')]
    model = Path(summary['config_final']['model_path'])
    build = json.loads((executable.parent / '构建信息.json').read_text(encoding='utf-8'))
    expected_model_hash = hashlib.sha256((ROOT / 'models' / 'best.pt').read_bytes()).hexdigest()
    expected_runtime = {'device': expected_device, 'half': False}
    final_config = summary['config_final']
    frame_parameters_match = all(
        float(row['conf']) == final_config['conf'] and float(row['iou']) == final_config['iou']
        and int(row['imgsz']) == final_config['imgsz']
        and int(row['density_threshold']) == final_config['density_threshold']
        for row in frames
    )
    checks = {
        'exe_exit_zero': result.returncode == 0,
        'status_success': summary['status'] in ('completed', 'limited'),
        'device_expected': summary['device'] == expected_device,
        'default_fp32': summary['config_initial']['half'] is False,
        'actual_fp32_runtime_matches_metadata': summary['inference_runtime'] == saved['inference_runtime'] == metadata['metadata']['inference_runtime'] == expected_runtime,
        'version_2_1_matches_metadata': summary['app_version'] == saved['app_version'] == metadata['metadata']['app_version'] == '2.1.0',
        'model_sha256_matches_metadata': summary['model_sha256'] == saved['model_sha256'] == metadata['metadata']['model_sha256'] == expected_model_hash,
        'schema_2_matches_metadata': saved['schema_version'] == metadata['schema_version'] == 2,
        'per_frame_effective_parameters': frame_parameters_match,
        'processed_12_frames': summary['processed_frames'] == summary['analyzed_frames'] == 12,
        'csv_12_frames': len(frames) == saved['recorded_frames'] == 12,
        'frame_indices_complete': [int(row['frame_index']) for row in frames] == list(range(1, 13)),
        'source_timestamps': all(abs(float(row['source_time_seconds']) - i / 6) < 1e-5 for i, row in enumerate(frames)),
        'detection_rows_match_frames': len(detections) == sum(int(row['current_count']) for row in frames),
        'line_events_match_summary': sum(row['event_type'] == 'line_crossing' for row in events) == sum(saved['line_counts'].values()),
        'roi_events_match_summary': sum(row['event_type'] == 'roi_dwell' for row in events) == saved['roi_dwell_events'],
        'last_cumulative_matches_summary': int(frames[-1]['cumulative_unique_count']) == saved['cumulative_unique_count'],
        'status_matches_metadata': saved['status'] == metadata['status'] == summary['status'],
        'no_reported_error': not summary.get('error'),
        'packaged_model_preserved': hashlib.sha256(model.read_bytes()).hexdigest() == '975aefac6e72860b240d7c9ef6f4785899725373a45e4c7ab8bfa57a27c2a384',
        'source_matches_build': source_hashes() == build['source_sha256'],
        'exe_matches_build': hashlib.sha256(executable.read_bytes()).hexdigest() == build['executable_sha256'],
        'model_matches_build': expected_model_hash == build['model_sha256'],
        'screenshot_saved': screenshot_path.is_file() and screenshot_path.stat().st_size > 10000,
    }
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Packaged EXE software integration check; synthetic still-photo fixture, not accuracy or traffic evaluation.',
        'executable': str(executable),
        'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
        'requested_device': args.device, 'actual_device': summary['device'],
        'app_version': summary['app_version'], 'model_sha256': summary['model_sha256'],
        'source_sha256': build['source_sha256'],
        'summary': str(report_path), 'screenshot': str(screenshot_path),
        'checks': checks, 'passed': all(checks.values()),
    }
    destination = ROOT / 'validation' / f'packaged_{tag}_verification.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
