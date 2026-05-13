"""Subtitld add-on entry point for `python-audio-separator`.

Wraps [nomadkaraoke/python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
(MIT). Drop-in alternative to the host's built-in `ffmpeg-separator`
(stereo mid/side trick) — same `audio.separate` task contract, much
better quality at the cost of a one-shot ~50–500 MB model download
and a few seconds of inference per minute.

Result frame must match what `AddonAudioSeparatorProvider` expects:

    {"vocals": "<path>", "background": "<path>"}

`python-audio-separator` writes 2 (or 4 for Demucs) stems with names
like `<input>_(Vocals)_<model>.flac` and `<input>_(Instrumental)_<model>.flac`.
We normalize: pick the vocals stem out by name match, treat anything
else as the "background" — for 4-stem Demucs we fall back to mixing the
non-vocal stems together so callers always get a single instrumental
file (matches the 2-stem contract the host expects).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
from pathlib import Path

log = logging.getLogger('audio-separator')
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[audio-separator] %(levelname)s %(message)s')

PROTOCOL = 1
ADDON_ID = 'audio-separator'
VERSION = '0.0.4'

# Host (`audioengine.py`) hard-requires 48 kHz mono audio: any cached
# stem that round-trips through it must match or it raises
# `ValueError: Sample rate mismatch`. The built-in `ffmpeg-separator`
# already writes 48 kHz mono — mirror its contract here so add-on
# outputs are drop-in compatible. MDX/UVR models natively emit
# 44.1 kHz stereo; Demucs emits 44.1 kHz stereo too. We post-process
# every stem before handing it back to the host.
_HOST_SAMPLE_RATE = 48000
_HOST_CHANNELS = 1


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------
_write_lock = threading.Lock()


def write_frame(frame: dict) -> None:
    line = json.dumps(frame, ensure_ascii=False)
    with _write_lock:
        sys.stdout.write(line + '\n')
        sys.stdout.flush()


def emit_progress(rid, value, message=''):
    write_frame({'id': rid, 'type': 'progress',
                 'data': {'value': max(0.0, min(1.0, float(value))), 'message': message}})


def emit_error(rid, code, message, retryable=False):
    write_frame({'id': rid, 'type': 'error',
                 'data': {'code': code, 'message': message, 'retryable': retryable}})


def emit_result(rid, data):
    write_frame({'id': rid, 'type': 'result', 'data': data})


# ---------------------------------------------------------------------------
# Separator cache — `Separator` is expensive to construct (loads models,
# allocates ONNX/torch sessions). Keep one alive per (model, device,
# output_format, output_dir) tuple so back-to-back calls reuse it.
# ---------------------------------------------------------------------------
_sep_lock = threading.Lock()
_separators: dict[tuple, object] = {}
_pending_cancel: set[str] = set()
_pending_cancel_lock = threading.Lock()


def _model_dir() -> Path:
    """Where downloaded model weights live. Per-user cache so reinstalling
    the addon doesn't re-download. We also let `AUDIO_SEPARATOR_MODEL_DIR`
    override for CI / tests."""
    override = os.environ.get('AUDIO_SEPARATOR_MODEL_DIR')
    if override:
        path = Path(override)
    else:
        # Mirror the platform-appropriate user-cache for the addon.
        # Falls back to ~/.cache on Linux, ~/Library/Caches on macOS,
        # %LOCALAPPDATA% on Windows.
        try:
            from platformdirs import user_cache_dir  # type: ignore
            path = Path(user_cache_dir('subtitld')) / 'addons' / 'audio-separator' / 'models'
        except Exception:
            path = Path.home() / '.cache' / 'subtitld' / 'addons' / 'audio-separator' / 'models'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_separator(model_filename: str, device: str, output_format: str,
                   output_dir: str):
    key = (model_filename, device, output_format, output_dir)
    with _sep_lock:
        cached = _separators.get(key)
        if cached is not None:
            return cached

        try:
            from audio_separator.separator import Separator  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f'python-audio-separator package not available: {exc}') from exc

        log.info('building Separator model=%s device=%s format=%s',
                 model_filename, device, output_format)

        # `Separator(...)` constructor signature is stable across the 0.17+
        # line. Older versions may have `use_cuda=True` instead of the
        # device strings — catch and fall back if needed.
        try:
            sep = Separator(
                model_file_dir=str(_model_dir()),
                output_dir=output_dir,
                output_format=output_format,
                use_cuda=(device == 'cuda'),
            )
        except TypeError:
            # Newer (1.x) signature uses keyword `cuda`/`mps` differently.
            sep = Separator(
                model_file_dir=str(_model_dir()),
                output_dir=output_dir,
                output_format=output_format,
            )

        sep.load_model(model_filename=model_filename)
        _separators[key] = sep
        return sep


# ---------------------------------------------------------------------------
# Stem-name disambiguation
# ---------------------------------------------------------------------------
# python-audio-separator names its outputs like:
#   <input_stem>_(Vocals)_<model>.flac
#   <input_stem>_(Instrumental)_<model>.flac
# Demucs additionally emits Drums/Bass/Other stems. Match case-insensitively.
_VOCAL_RE = re.compile(r'\(vocals?\)', re.IGNORECASE)
_INSTRUMENTAL_RE = re.compile(r'\(instrumental\)', re.IGNORECASE)
_NO_VOCALS_RE = re.compile(r'\(no\s*vocals?\)', re.IGNORECASE)
_DRUMS_RE = re.compile(r'\(drums?\)', re.IGNORECASE)
_BASS_RE = re.compile(r'\(bass\)', re.IGNORECASE)
_OTHER_RE = re.compile(r'\(other\)', re.IGNORECASE)


def _classify_stems(paths: list[str]) -> tuple[str | None, list[str], list[str]]:
    """Return (vocals_path, background_paths, leftover) given a list of
    stem files. `background_paths` is the set of non-vocal stems we want
    to keep; `leftover` is anything we don't recognize (e.g. an
    `(Echo)` stem, ignored)."""
    vocals = None
    background = []
    leftover = []
    for p in paths:
        name = os.path.basename(p)
        if _VOCAL_RE.search(name):
            vocals = p
        elif (_INSTRUMENTAL_RE.search(name)
              or _NO_VOCALS_RE.search(name)
              or _DRUMS_RE.search(name)
              or _BASS_RE.search(name)
              or _OTHER_RE.search(name)):
            background.append(p)
        else:
            leftover.append(p)
    return vocals, background, leftover


# libsndfile (used by python-audio-separator's metadata probe) knows
# WAV / FLAC / OGG / AIFF / AU / W64 / etc., but throws
# `Format not recognised` on video containers (MKV / MP4 / WebM / MOV /
# AVI / …). python-audio-separator still limps along via its own ffmpeg
# fallback, but the bit-depth probe silently defaults to 16-bit and the
# logs are noisy. Fix it at the source: any extension we don't recognise
# as audio, transcode to a temp WAV via ffmpeg up front and hand that
# to the Separator. ffmpeg is a hard dep already (`_mix_to_single_track`
# below relies on it too).
_LIBSNDFILE_FRIENDLY = {
    '.wav', '.wave', '.flac', '.ogg', '.oga', '.opus', '.aiff', '.aif',
    '.au', '.snd', '.w64', '.caf', '.mp3', '.m4a',
}


def _prepare_input_audio(input_path: str, work_dir: str) -> tuple[str, str | None]:
    """Return `(path_to_feed_separator, temp_to_cleanup_or_None)`.

    If the input is already an audio container libsndfile speaks, returns
    `(input_path, None)`. Otherwise extracts the first audio stream to a
    temp WAV in `work_dir` and returns `(temp_path, temp_path)` — caller
    is responsible for unlinking the temp file once the Separator is done.
    """
    import subprocess
    import uuid

    ext = os.path.splitext(input_path)[1].lower()
    if ext in _LIBSNDFILE_FRIENDLY:
        return input_path, None

    temp_path = os.path.join(work_dir, f'_input_{uuid.uuid4().hex}.wav')
    # `-vn` drops video; `-map 0:a:0?` grabs the first audio stream if
    # present (the `?` makes the mapping optional so we get a clear
    # ffmpeg error instead of a Python crash when the file has no audio).
    # pcm_s16le matches what libsndfile would have produced via its
    # 16-bit default, so no quality regression vs the warning path.
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-i', input_path,
        '-vn', '-map', '0:a:0?',
        '-acodec', 'pcm_s16le',
        temp_path,
    ]
    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if rc.returncode != 0:
        raise RuntimeError(
            f'ffmpeg decode failed: '
            f'{rc.stderr.decode("utf-8", "replace")[:200]}')
    if not os.path.isfile(temp_path) or os.path.getsize(temp_path) == 0:
        raise RuntimeError(
            f'ffmpeg produced no audio output for {input_path!r}')
    return temp_path, temp_path


def _mix_to_single_track(stems: list[str], output_path: str,
                         output_format: str) -> str:
    """Demucs gives us 3 instrumental stems (drums, bass, other). The host
    contract is one `background` file — mix them together with ffmpeg's
    `amix` filter. The result lives next to the source stems so the
    addon stays self-contained.

    We fold the host's 48 kHz mono normalization into the same ffmpeg
    invocation (rather than running a separate resample pass) so the
    Demucs path is a single decode/encode cycle.
    """
    import subprocess

    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y']
    for s in stems:
        cmd += ['-i', s]
    # Equal-weight mix; `normalize=0` so we don't quietly attenuate.
    cmd += [
        '-filter_complex',
        f'amix=inputs={len(stems)}:normalize=0',
        '-ar', str(_HOST_SAMPLE_RATE),
        '-ac', str(_HOST_CHANNELS),
        output_path,
    ]
    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if rc.returncode != 0:
        raise RuntimeError(
            f'ffmpeg amix failed: '
            f'{rc.stderr.decode("utf-8", "replace")[:200]}')
    return output_path


def _normalize_to_host_format(path: str) -> str:
    """Resample/downmix a stem in-place to 48 kHz mono.

    The host's `audioengine.load_audio` (see Subtitld
    `src/subtitld/modules/audioengine.py`) raises
    `ValueError: Sample rate mismatch` if the file isn't 48 kHz.
    `python-audio-separator` writes at the model's training rate
    (44.1 kHz for MDX/UVR/Roformer, 44.1 kHz stereo for Demucs).
    Cheaper than a probe + branch: unconditionally re-encode each
    stem to 48 kHz mono. ffmpeg's resampler short-circuits when the
    rate already matches, so this is a near-noop on the rare model
    that already produces 48 kHz.

    We write to a sibling temp file and rename atomically so a
    crashed ffmpeg never leaves a half-written stem in the cache
    (where it'd survive across runs and break the next playback).
    """
    import subprocess

    root, ext = os.path.splitext(path)
    temp_path = f'{root}._normalize{ext}'
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-i', path,
        '-ar', str(_HOST_SAMPLE_RATE),
        '-ac', str(_HOST_CHANNELS),
        temp_path,
    ]
    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if rc.returncode != 0:
        # Best-effort cleanup of the partial temp before surfacing the
        # error — leaving stray `._normalize.flac` files in the cache
        # would confuse future cache-hit checks.
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise RuntimeError(
            f'ffmpeg resample failed for {os.path.basename(path)!r}: '
            f'{rc.stderr.decode("utf-8", "replace")[:200]}')
    os.replace(temp_path, path)
    return path


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------
def handle_audio_separate(rid: str, params: dict, defaults: dict) -> None:
    input_path = params.get('input_path')
    output_dir = params.get('output_dir')
    if not input_path or not output_dir:
        emit_error(rid, 'bad_params', 'input_path and output_dir are required')
        return
    if not os.path.isfile(input_path):
        emit_error(rid, 'bad_params', f'input_path not found: {input_path!r}')
        return

    options = params.get('options') or {}
    model_filename = options.get('model_filename') or defaults['model_filename']
    device = options.get('device') or defaults['device']
    output_format = (options.get('output_format')
                     or defaults['output_format']).upper()

    with _pending_cancel_lock:
        if rid in _pending_cancel:
            _pending_cancel.discard(rid)
            emit_error(rid, 'cancelled', 'cancelled before separation started')
            return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    emit_progress(rid, 0.05, 'Loading separator model...')
    try:
        sep = _get_separator(model_filename, device, output_format, output_dir)
    except Exception as exc:
        log.exception('separator load failed')
        emit_error(rid, 'internal', f'failed to load separator: {exc}')
        return

    # Pre-decode video containers (mkv / mp4 / etc.) into a temp WAV — the
    # Separator's metadata probe goes through libsndfile, which only
    # speaks audio formats. Tracked in `temp_decoded` so the finally below
    # can clean up regardless of which branch returns/raises after this
    # point.
    temp_decoded: str | None = None
    try:
        emit_progress(rid, 0.2, 'Preparing input audio...')
        try:
            feed_path, temp_decoded = _prepare_input_audio(input_path, output_dir)
        except Exception as exc:
            log.exception('input decode failed')
            emit_error(rid, 'internal', f'failed to decode input: {exc}')
            return

        emit_progress(rid, 0.4, 'Separating audio...')
        try:
            # `Separator.separate(...)` is synchronous and returns the list of
            # output filenames it just wrote. Some versions return basenames
            # relative to `output_dir`; normalize.
            raw = sep.separate(feed_path)
            outputs = [
                p if os.path.isabs(p) else os.path.join(output_dir, p)
                for p in (raw or [])
            ]
        except Exception as exc:
            log.exception('separation failed')
            emit_error(rid, 'internal', f'separation failed: {exc}')
            return
    finally:
        if temp_decoded is not None:
            try:
                os.remove(temp_decoded)
            except OSError:
                pass

    if not outputs:
        emit_error(rid, 'internal', 'separator returned no output files')
        return

    emit_progress(rid, 0.9, 'Finalizing stems...')
    vocals_path, background_paths, leftover = _classify_stems(outputs)
    if vocals_path is None:
        emit_error(rid, 'internal',
                   f'no vocals stem found in outputs: {outputs!r}')
        return

    if not background_paths:
        emit_error(rid, 'internal',
                   f'no instrumental stem found in outputs: {outputs!r}')
        return

    if len(background_paths) == 1:
        background_path = background_paths[0]
        # 2-stem path: separator wrote a stem directly to disk at the
        # model's native rate (44.1 kHz stereo for MDX/UVR/Roformer).
        # Resample/downmix to the host's 48 kHz mono in place.
        try:
            _normalize_to_host_format(background_path)
        except Exception as exc:
            log.exception('background normalize failed')
            emit_error(rid, 'internal',
                       f'failed to normalize background to 48 kHz mono: {exc}')
            return
    else:
        # 4-stem Demucs case — mix drums/bass/other into one file. We pick
        # the model name suffix from one of them so the merged file stays
        # next to its source stems (and gets cleaned up alongside them when
        # the cache rolls over). `_mix_to_single_track` already emits at
        # 48 kHz mono, no separate normalize pass needed here.
        merged_name = (Path(background_paths[0]).stem.split('_(')[0]
                       + '_(Background)_'
                       + os.path.splitext(model_filename)[0]
                       + '.' + output_format.lower())
        background_path = os.path.join(output_dir, merged_name)
        try:
            _mix_to_single_track(background_paths, background_path, output_format)
        except Exception as exc:
            log.exception('background mix failed')
            emit_error(rid, 'internal',
                       f'failed to mix Demucs stems: {exc}')
            return

    # Vocals stem always comes straight off the separator at the model's
    # native rate — normalize unconditionally so the host's load_audio
    # contract (48 kHz mono) holds for both stems.
    try:
        _normalize_to_host_format(vocals_path)
    except Exception as exc:
        log.exception('vocals normalize failed')
        emit_error(rid, 'internal',
                   f'failed to normalize vocals to 48 kHz mono: {exc}')
        return

    emit_progress(rid, 1.0, 'Done')
    emit_result(rid, {
        'vocals': vocals_path,
        'background': background_path,
    })


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    manifest_path = Path(__file__).resolve().parent / 'manifest.json'
    config_defaults: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            config_defaults = {
                f.get('key'): f.get('default')
                for f in (manifest.get('config_schema') or {}).get('fields', [])
                if f.get('default') is not None
            }
        except Exception:
            log.exception('manifest parse failed')

    defaults = {
        'model_filename': (os.environ.get('AUDIO_SEPARATOR_MODEL')
                           or config_defaults.get('model_filename',
                                                  'Kim_Inst.onnx')),
        'device': (os.environ.get('AUDIO_SEPARATOR_DEVICE')
                   or config_defaults.get('device', 'cpu')),
        'output_format': (os.environ.get('AUDIO_SEPARATOR_OUTPUT_FORMAT')
                          or config_defaults.get('output_format', 'FLAC')),
    }

    write_frame({
        'type': 'hello',
        'protocol': PROTOCOL,
        'addon': ADDON_ID,
        'version': VERSION,
        'capabilities': [
            {'task': 'audio.separate'},
        ],
    })

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            frame = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        ftype = frame.get('type')
        rid = frame.get('id', '')

        if ftype == 'shutdown':
            log.info('shutdown received; exiting')
            return 0
        if ftype == 'cancel':
            target = (frame.get('data') or {}).get('target') or frame.get('target')
            if target:
                with _pending_cancel_lock:
                    _pending_cancel.add(target)
            continue
        if ftype == 'audio.separate':
            threading.Thread(
                target=handle_audio_separate,
                args=(rid, frame.get('params') or {}, defaults),
                daemon=True,
            ).start()
            continue
        # Host control frames (`ready`, future-proof) carry no request id and
        # expect no response. Ignore unless a request id was supplied.
        if not rid:
            log.debug('ignoring host control frame: %s', ftype)
            continue

        emit_error(rid, 'bad_params', f'unknown request type: {ftype!r}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
