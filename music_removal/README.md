# Music Removal

Utilities for removing or reducing background music before speech enhancement.

## Demucs vocal extraction

Install Demucs separately:

```cmd
pip install -U demucs soundfile
```

Or install the folder-specific requirements:

```cmd
pip install -r music_removal/requirements.txt
```

Extract vocal stems:

```cmd
python music_removal/extract_vocals_demucs.py ^
  --input audios\my_dataset_numbered ^
  --output audios\my_dataset_vocals ^
  --device cuda ^
  --model htdemucs ^
  --overwrite
```

Then pass the resulting vocal stem folder to `audio_enhancement/enhance_audio.py`.

`htdemucs` is a good default. Try `htdemucs_ft` for better quality when slower
processing is acceptable.
