# SegmentScribe

Small audio prep utilities for converting source audio to numbered WAV files,
removing music, and enhancing speech with ModelScope ZipEnhancer or
MossFormer2.

## Setup

```cmd
conda create -n segmentScribe python=3.11 -y
conda activate segmentScribe
```

Install PyTorch with CUDA. Pick the CUDA build that matches your machine from
https://pytorch.org/get-started/locally/. Example:

```cmd
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Install the Python dependencies:

```cmd
pip install -r requirements.txt
```

Install `ffmpeg` for audio conversion and non-WAV inputs:

```cmd
conda install -c conda-forge ffmpeg -y
```

For audio with background music, install Demucs as an optional vocal-separation
preprocessor:

```cmd
pip install -r music_removal/requirements.txt
```

For MossFormer2 enhancement, install the additional dependencies:

```cmd
pip install -r MossFormer/requirements.txt
```

## Preprocessing of dirty audio data(optional)

Convert a folder of audio files to numbered 16 kHz mono WAV files. This is
useful when source files have unusual formats or filenames:

```
python utils/convert_to_numbered_wav.py \
  --input my_wav_folder \
  --output numbered_wavs \
  --recursive \
  --overwrite
```

## Remove background music(optional)

Usual denoising/audio enhancing models are trained on DNS2020-like dataset, which is designed to remove environmental noise, NOT background music. So we need additional steps if the source audios contain background music.

Command to use demucs

```cmd
python music_removal/extract_vocals_demucs.py \
  --input my_wav_folder \
  --output vocal_stems \
  --device cuda \
  --model htdemucs \
  --overwrite
```

Better quality but slower

```cmd
python music_removal/extract_vocals_demucs.py \
  --input my_wav_folder \
  --output vocal_stems \
  --device cuda \
  --model htdemucs_ft \
  --overwrite
```

## Denoising method 1: ZipEnhancer

CMD:

```cmd
python zip_enhancer/enhance_audio.py \
  --input my_wav_folder \
  --output enhanced_wavs \
  --device cuda:0 \
  --silence-aware-splits \
  --preprocess-workers 4 \
  --normalize match_original \
  --alignment-metric rms \
  --overwrite
```

About volume normalization: The modelscope code has internal volume normalization, which has been eliminated, but the model itself may still change the volume.

--normalize has these options: 1, None - no normalization of the model output; 2, match_original - in alignment of the original volume; 3, a value - a numeric dBFS value such as -20; 4, library_median - estimate one shared corpus RMS target using a duration-weighted speech median. Note that for safety issues, we will keep the value not too far away from the lowest volume one. So if there are certain audios with very low volumes, the final target value will be very low.

--alignment-metric has these options: 1, rms; 2, peak

About model downloading

The default enhancement model is **automatively** downloaded by ModelScope:

```text
iic/speech_zipenhancer_ans_multiloss_16k_base
```

Comment: ZipEnhancer is a newer model, but after some trials, I don't think it performs well. It is recommended to use the second method, which is more steady.

## Denoising method 2: MossFormer 2

MossFormer2 is bundled in `MossFormer/` and uses a downloaded checkpoint path
explicitly. First download the checkpoint:

```cmd
python MossFormer/download_checkpoints.py --model-path checkpoints/MossFormer2_SE_48K
```

Then enhance a folder:

```cmd
python MossFormer/enhance_folder.py \
  --input-dir my_wav_folder \
  --output-dir mossformer_enhanced \
  --model-path checkpoints/MossFormer2_SE_48K \
  --preprocess-workers 4 \
  --inference-batch-size 4 \
  --overwrite
```

`--preprocess-workers` loads and resamples upcoming files in background threads.
GPU inference still runs in the main process.
`--inference-batch-size` batches equal-length MossFormer windows inside each
long file to improve GPU occupancy; increase it until VRAM use is healthy.
