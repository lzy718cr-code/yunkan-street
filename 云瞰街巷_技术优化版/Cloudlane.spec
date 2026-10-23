# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH)
datas = collect_data_files('ultralytics')
datas += copy_metadata('ultralytics', recursive=True)
datas += [(str(root / 'models' / 'best.pt'), 'models')]
datas += [(str(root / 'models' / 'space_best.pt'), 'models')]
datas += [(str(root / 'models' / 'yolo11s.pt'), 'models')]
demo_clip = root / 'demo' / 'demo_60s.mp4'
if demo_clip.is_file():
    datas += [(str(demo_clip), 'demo')]
hiddenimports = collect_submodules('ultralytics')
hiddenimports += ['lap', 'lap._lapjv', 'torchvision', 'torchvision._C',
                  'scipy.optimize._lsap', 'scipy.special._ufuncs']
# SciPy 1.18 vendors array-api-compat under _external and imports the fft
# namespaces dynamically. Existing SciPy hooks do not discover these imports.
hiddenimports += collect_submodules('scipy._external.array_api_compat.numpy')
hiddenimports += collect_submodules('scipy._external.array_api_compat.torch')

a = Analysis(
    [str(root / 'main.py')],
    pathex=[str(root)],
    binaries=[], datas=datas, hiddenimports=hiddenimports,
    hookspath=[], hooksconfig={'matplotlib': {'backends': ['Agg']}},
    runtime_hooks=[str(root / 'tools' / 'frozen_runtime.py')],
    excludes=['tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'pytest'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name='云瞰街巷', debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False, disable_windowed_traceback=False,
    argv_emulation=False, target_arch=None, codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='云瞰街巷')
