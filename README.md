# audio-separator add-on for Subtitld

High-quality vocal/music separation backed by
[python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
(MIT). Drop-in alternative to Subtitld's built-in `ffmpeg-separator`
("ffmpeg (fast)") which uses the stereo mid/side trick: this add-on
trades a few seconds of inference per minute and a one-shot model
download (~50–500 MB) for substantially cleaner vocal isolation.

## When to install

The built-in ffmpeg separator is plenty when:

- the source already has a clean stereo mix (vocals dead-center),
- you only need a rough vocal-vs-music split for transcription
  preview, or
- you don't want a multi-hundred-MB model download.

Switch to this add-on when:

- the built-in lets through too much music (kick / bass / strings
  panned to center bleed into "vocals" with the mid/side trick), or
- the built-in pulls non-vocal content out the sides (panned
  backing vocals leak into "background"), or
- you're producing release-quality dubs and want the cleanest
  source for `voice_ref_audio` extraction.

## Models

Selected via the addon's config panel (Global tab → Audio separator
→ Configure):

| Model                                               | Size  | Notes                                                              |
| --------------------------------------------------- | ----- | ------------------------------------------------------------------ |
| `UVR-MDX-NET-Inst_HQ_3.onnx` (default)              | ~64 MB | Fast, balanced 2-stem (vocals / instrumental). Good first pick.    |
| `UVR_MDXNET_KARA_2.onnx`                            | ~64 MB | Tuned to split *main* vocals from backing vocals.                  |
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt`         | ~396 MB | Top-quality 2-stem, SDR 12.98. Slower; pick when quality matters. |
| `htdemucs_ft.yaml`                                  | ~320 MB | Demucs fine-tuned 4-stem (vocals/drums/bass/other). 4-stem mode is internally collapsed back to 2 — vocals + mixed-instrumental — to match the host contract. |

Models live in
`~/.cache/subtitld/addons/audio-separator/models/`
(or platform equivalent) and are downloaded on first use.
`AUDIO_SEPARATOR_MODEL_DIR=...` overrides the location.

## Building

```bash
pip install pyinstaller
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install "audio-separator[cpu]>=0.17,<1.0" platformdirs
pyinstaller audio-separator-addon.spec --distpath dist/
cd dist/audio-separator-addon
zip -r ../audio-separator-0.0.4-linux-x86_64.zip . ../../manifest.json ../../LICENSE ../../README.md
```

## Output contract

The host (`AddonAudioSeparatorProvider`) expects a single result
frame:

```json
{"vocals": "<path>", "background": "<path>"}
```

For 4-stem Demucs the wrapper mixes drums/bass/other together with
ffmpeg's `amix` filter so callers always see one `background` file.

Both stems are post-normalized to **48 kHz** before the result frame
fires (the host's `audioengine.load_audio` raises `Sample rate mismatch`
on anything else). Channel layout is split by stem:

- **background**: stereo. MDX/UVR/Demucs models natively emit 44.1 kHz
  stereo and the spatial information (drum spread, instrument panning,
  ambience) is musically meaningful. The host's stem playback path
  (`Clip` and the bounce mixer) is already stereo-aware.
- **vocals**: mono. Speech is functionally mono and the dubbing
  pipeline's clone-reference extraction expects a single-channel
  signal.

MDX/UVR/Roformer models natively emit 44.1 kHz, so the resample is
unconditional — cheap with ffmpeg's short-circuit when rates already
match.

## License

Wrapper code: MIT. python-audio-separator: MIT. Individual model
weights have their own licenses — check each model's card on
HuggingFace before commercial use.
