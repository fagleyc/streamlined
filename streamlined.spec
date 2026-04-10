# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['streamlined.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'nptdms',
        'scipy',
        'scipy.signal',
        'pyqtgraph',
        'openpyxl',
        'PyQt6',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'PyQt5', 'tkinter', '_tkinter',
        'yt_dlp', 'Cryptodome', 'websockets', 'requests',
        'curl_cffi', 'brotli', 'secretstorage', 'certifi',
        'mutagen', 'IPython', 'jupyter', 'notebook', 'nbformat',
        'jedi', 'parso', 'zmq', 'tornado', 'cryptography',
        'PIL', 'cv2', 'sklearn', 'tensorflow', 'torch',
    ],
    noarchive=False,
)

# Exclude unnecessary files
a.datas = [d for d in a.datas if not any(
    x in d[0] for x in ['scripts/', 'examples/', 'Run01/', 'CalFiles/', '.m']
)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Streamlined',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
