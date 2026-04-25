#!/usr/bin/env python
"""Run Demucs while saving WAV output with soundfile.

Recent torchaudio versions can route `torchaudio.save()` through TorchCodec.
That is fragile when TorchCodec, PyTorch, and FFmpeg versions are not aligned.
Demucs separation itself can still work, so this runner patches only the save
function used by Demucs' CLI.
"""

from __future__ import annotations

from pathlib import Path


def save_audio_with_soundfile(
    wav,
    path,
    samplerate: int,
    bitrate: int = 320,
    clip: str = "rescale",
    bits_per_sample: int = 16,
    as_float: bool = False,
    preset: int = 2,
    **_unused_kwargs,
) -> None:
    from demucs.audio import encode_mp3, prevent_clip
    import soundfile as sf

    wav = prevent_clip(wav, mode=clip)
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".mp3":
        encode_mp3(wav, path, samplerate, bitrate, preset=preset, verbose=True)
        return

    audio = wav.detach().cpu().numpy()
    if audio.ndim == 2:
        audio = audio.T

    if suffix == ".wav":
        if as_float:
            subtype = "FLOAT"
        elif bits_per_sample == 24:
            subtype = "PCM_24"
        elif bits_per_sample == 32:
            subtype = "PCM_32"
        else:
            subtype = "PCM_16"
        sf.write(str(path), audio, samplerate, subtype=subtype)
        return

    if suffix == ".flac":
        subtype = "PCM_24" if bits_per_sample == 24 else "PCM_16"
        sf.write(str(path), audio, samplerate, subtype=subtype)
        return

    raise ValueError(f"Invalid suffix for path: {suffix}")


def main() -> None:
    import demucs.audio
    import demucs.separate

    demucs.audio.save_audio = save_audio_with_soundfile
    demucs.separate.save_audio = save_audio_with_soundfile
    demucs.separate.main()


if __name__ == "__main__":
    main()
