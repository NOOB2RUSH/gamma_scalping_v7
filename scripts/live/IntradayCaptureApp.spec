# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


project_root = Path(SPECPATH).resolve().parents[1]
entry = project_root / "scripts" / "live" / "intraday_capture_app.py"


a = Analysis(
    [str(entry)],
    pathex=[str(project_root), str(project_root / "scripts" / "live")],
    binaries=[],
    datas=collect_data_files("akshare", includes=["file_fold/*"]),
    hiddenimports=[
        "akshare",
        "pyarrow",
        "pyarrow.parquet",
    ]
    + collect_submodules("core.configs"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="IntradayCaptureApp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="IntradayCaptureApp",
)
