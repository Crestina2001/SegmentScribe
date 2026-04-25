# MossFormer Folder Enhancer

Portable MossFormer2 speech enhancement code for enhancing a folder of audio files.

## Install dependencies

From inside this `MossFormer` folder:

```bash
pip install -r ../requirements.txt
pip install -r requirements.txt
```

For MP3/AAC/M4A output support, install FFmpeg so `pydub` can export those formats.

## Run

```bash
python enhance_folder.py --input-dir path/to/input_audios --output-dir path/to/output_audios --model-path checkpoints/MossFormer2_SE_48K
```

The script writes enhanced files directly into `path/to/output_audios`. It
always flattens ClearVoice's temporary `MossFormer2_SE_48K` subfolder.

Useful options:

```bash
python enhance_folder.py --input-dir input_audios --output-dir output_audios --model-path checkpoints/MossFormer2_SE_48K
python enhance_folder.py --input-dir input_audios --output-dir output_audios --model-path checkpoints/MossFormer2_SE_48K --overwrite
python enhance_folder.py --input-dir input_audios --output-dir output_audios --model-path checkpoints/MossFormer2_SE_48K --cpu
python enhance_folder.py --input-dir input_audios --output-dir output_audios --model-path checkpoints/MossFormer2_SE_48K --chunk-seconds 4
python enhance_folder.py --input-dir input_audios --output-dir output_audios --model-path checkpoints/MossFormer2_SE_48K --preprocess-workers 4
python enhance_folder.py --input-dir input_audios --output-dir output_audios --model-path checkpoints/MossFormer2_SE_48K --inference-batch-size 4
```

## Checkpoints

You can download checkpoints separately:

```bash
python download_checkpoints.py --model-path checkpoints/MossFormer2_SE_48K
```

This writes to:

```text
<project-root>/checkpoints/MossFormer2_SE_48K/
```

`<project-root>` means the parent directory that contains the `MossFormer`
folder.

You can choose any model path:

```bash
python download_checkpoints.py --model-path models/MossFormer2_SE_48K
python enhance_folder.py --input-dir input_audios --output-dir output_audios --model-path models/MossFormer2_SE_48K
```

Relative `--model-path` values are resolved from `<project-root>`. Absolute
paths are used exactly as provided.

`--model-path` is required for both downloading and enhancement. The enhancement
script validates the checkpoint directory before loading the model.

For offline use, make sure your checkpoint directory contains:

```text
last_best_checkpoint
last_best_checkpoint.pt
```

## Copying To Another Project

Copy the whole `MossFormer` folder. It contains the needed ClearVoice source code
for `MossFormer2_SE_48K`, the enhancement script, and dependency list.
