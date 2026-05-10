# PyInstaller spec for the audio-separator add-on.
# Build with: pyinstaller audio-separator-addon.spec --distpath dist/

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


def _safe_collect(fn, name):
    try:
        return fn(name)
    except Exception:
        return []


# python-audio-separator pulls onnxruntime + torch + librosa + samplerate
# + (optionally) demucs. Each backend reaches into its own dynamic-import
# soup, so we sweep submodules + data files broadly. The big offenders
# historically:
#   * `torch` — without `collect_submodules` PyInstaller misses dispatcher
#     plug-ins loaded by string name in `torch._C._load_*`.
#   * `onnxruntime` — ships its own .so/.dll alongside Python wrappers;
#     `collect_data_files` picks them up.
#   * `librosa` — uses `pkg_resources` to find PEP 561 stub files for the
#     beat tracker; without `collect_data_files` the freeze crashes when
#     downsampling certain models' output.
#   * `samplerate` — ships a libsamplerate dylib that PyInstaller doesn't
#     auto-detect on macOS without the explicit binaries collection
#     (handled by collect_data_files at the package level).
#   * `demucs` — `htdemucs_ft.yaml` references `bag_of_models` which loads
#     submodels at runtime via importlib; collect_submodules helps.
hiddenimports = (
    _safe_collect(collect_submodules, 'audio_separator')
    + _safe_collect(collect_submodules, 'onnxruntime')
    + _safe_collect(collect_submodules, 'torch')
    + _safe_collect(collect_submodules, 'librosa')
    + _safe_collect(collect_submodules, 'samplerate')
    + _safe_collect(collect_submodules, 'soundfile')
    + _safe_collect(collect_submodules, 'demucs')
    + _safe_collect(collect_submodules, 'omegaconf')
    + _safe_collect(collect_submodules, 'einops')
    + _safe_collect(collect_submodules, 'pydub')
)
datas = (
    _safe_collect(collect_data_files, 'audio_separator')
    + _safe_collect(collect_data_files, 'onnxruntime')
    + _safe_collect(collect_data_files, 'torch')
    + _safe_collect(collect_data_files, 'librosa')
    + _safe_collect(collect_data_files, 'samplerate')
    + _safe_collect(collect_data_files, 'soundfile')
    + _safe_collect(collect_data_files, 'demucs')
    + [('manifest.json', '.')]
)

block_cipher = None

a = Analysis(
    ['audio_separator_addon.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Speech recognition / LLM frameworks the addon doesn't need —
    # excluding these saves ~1 GB on the freeze.
    excludes=['tensorflow', 'jax', 'flax', 'gradio', 'transformers',
              'matplotlib', 'IPython', 'jupyter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='audio-separator-addon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, upx_exclude=[],
    name='audio-separator-addon',
)
