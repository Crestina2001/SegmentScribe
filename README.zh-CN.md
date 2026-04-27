# SegmentScribe

[English](README.md) | [简体中文](README.zh-CN.md)

SegmentScribe 用于将原始训练音频转换为 VoxCPM 兼容的训练片段。它会把源音频整理成编号 WAV，按需去除背景音乐，使用 ModelScope ZipEnhancer 或 MossFormer2 增强人声，并通过 `slide_rule` 切分出带转写对齐信息和清单文件的下游训练片段。

## 环境安装

创建 Conda 环境：

```cmd
conda create -n segmentScribe python=3.11 -y
conda activate segmentScribe
```

安装带 CUDA 的 PyTorch。请根据本机 CUDA 版本从 <https://pytorch.org/get-started/locally/> 选择对应命令。例如：

```cmd
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

安装 Python 依赖：

```cmd
pip install -r requirements.txt
```

## WebUI

启动本地浏览器界面：

```cmd
python webui.py
```

WebUI 可以运行下文介绍的主要流程：

- 将源音频转换为编号的 16 kHz 单声道 WAV；
- 使用 Demucs 去除背景音乐；
- 使用 MossFormer2 或 ZipEnhancer 做语音增强；
- 通过 `slide_rule` 切分音频；
- 查看 `summary.json`、`manifest.tsv`、转写文本和片段音频；
- 从 VoxCPM JSONL 中过滤疑似错误说话人的片段；
- 将最终片段音量归一化到新的安全数据集目录。

下载 WebUI 默认使用的模型 checkpoint：

```cmd
python utils/download_model_webui.py --models all
```

该命令会在 `checkpoints/` 下准备 MossFormer2、Qwen3-ASR 和 Qwen3 forced aligner
的默认路径。WebUI 的 **Models** 标签页中也提供同样的下载入口。

安装 `ffmpeg`，用于音频格式转换和处理非 WAV 输入：

```cmd
conda install -c conda-forge ffmpeg -y
```

如果音频中有背景音乐，可以额外安装 Demucs 作为人声分离预处理：

```cmd
pip install -r music_removal/requirements.txt
```

如果需要使用 MossFormer2 做语音增强，请安装额外依赖：

```cmd
pip install -r MossFormer/requirements.txt
```

## 脏音频预处理（可选）

把一个目录中的音频转换为编号的 16 kHz 单声道 WAV。适合处理格式混乱或文件名不规则的原始数据：

```cmd
python utils/convert_to_numbered_wav.py \
  --input my_wav_folder \
  --output numbered_wavs \
  --recursive \
  --overwrite
```

## 去除背景音乐（可选）

常见降噪或语音增强模型通常训练在类似 DNS2020 的数据上，主要处理环境噪声，不一定能可靠去除背景音乐。如果源音频包含明显 BGM，建议先做人声分离。

使用 Demucs：

```cmd
python music_removal/extract_vocals_demucs.py \
  --input my_wav_folder \
  --output vocal_stems \
  --device cuda \
  --model htdemucs \
  --overwrite
```

质量更好但速度更慢的模型：

```cmd
python music_removal/extract_vocals_demucs.py \
  --input my_wav_folder \
  --output vocal_stems \
  --device cuda \
  --model htdemucs_ft \
  --overwrite
```

## 降噪方法 1：ZipEnhancer

命令示例：

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

关于音量归一化：ModelScope 原始代码内部有音量归一化逻辑，本仓库中已去除这部分，但模型本身仍可能改变输出音量。

`--normalize` 可选值：

- `None`: 不对模型输出做音量归一化。
- `match_original`: 对齐到原始音频音量。
- 数值: 指定目标 dBFS，例如 `-20`。
- `library_median`: 使用语音片段的时长加权中位 RMS 估计整个数据集的共享目标音量。为了安全，目标值不会离最低音量样本太远；如果数据中有特别低音量的音频，最终目标也可能偏低。

`--alignment-metric` 可选值：

- `rms`
- `peak`

默认增强模型会通过 ModelScope 自动下载：

```text
iic/speech_zipenhancer_ans_multiloss_16k_base
```

备注：ZipEnhancer 是较新的模型，但实际试用后稳定性不如第二种方法。更推荐使用 MossFormer2。

## 降噪方法 2：MossFormer2

MossFormer2 已放在 `MossFormer/` 目录下，但需要显式下载 checkpoint。先下载模型：

```cmd
python MossFormer/download_checkpoints.py --model-path checkpoints/MossFormer2_SE_48K
```

增强整个目录：

```cmd
python MossFormer/enhance_folder.py \
  --input-dir my_wav_folder \
  --output-dir mossformer_enhanced \
  --model-path checkpoints/MossFormer2_SE_48K \
  --preprocess-workers 4 \
  --inference-batch-size 4 \
  --overwrite
```

说明：

- `--preprocess-workers` 使用后台线程提前加载和重采样后续文件。
- GPU 推理仍在主进程中运行。
- `--inference-batch-size` 会把单个长文件中等长的 MossFormer 窗口合批，提高 GPU 利用率。可以逐步增大，直到显存使用比较健康。

## 下载 Qwen3 ASR 模型

运行 `slide_rule` 前需要下载 ASR 和 forced aligner checkpoint。默认使用 ModelScope：

```cmd
python utils/download_qwen3_asr.py \
  --provider modelscope \
  --output-root checkpoints
```

也可以使用 Hugging Face：

```cmd
python utils/download_qwen3_asr.py \
  --provider hf \
  --output-root checkpoints
```

下载后会得到：

- `checkpoints/Qwen3-ASR-1.7B`
- `checkpoints/Qwen3-ForcedAligner-0.6B`

## 基于规则的切分：slide_rule

`slide_rule` 会把准备好的语音音频切成转写对齐的片段，适合后续 TTS 数据准备。它依次执行 ASR/forced alignment、规则标点处理、粗切分、静音感知细切分，以及 manifest 写入。

确保 `slide_LLM` 和 `slide_workflow` 在 `PYTHONPATH` 中可用，因为 `slicing_utils` 会复用其中的 ASR、prepass、rough-cut 和输出写入辅助逻辑。

默认 `transformers` 后端：

```cmd
python -m slide_rule \
  --input mossformer_enhanced \
  --output-dir sliced_segments \
  --model-path checkpoints/Qwen3-ASR-1.7B \
  --aligner-path checkpoints/Qwen3-ForcedAligner-0.6B \
  --asr-backend transformers \
  --device cuda:0 \
  --dtype bfloat16 \
  --asr-max-batch-size 1 \
  --min-seg-sec 3 \
  --max-seg-sec 10 \
  --vad-backend auto \
  --overwrite
```

可选 `vllm` 后端：

```cmd
python -m slide_rule \
  --input mossformer_enhanced \
  --output-dir sliced_segments \
  --model-path checkpoints/Qwen3-ASR-1.7B \
  --aligner-path checkpoints/Qwen3-ForcedAligner-0.6B \
  --asr-backend vllm \
  --device cuda:0 \
  --dtype bfloat16 \
  --asr-max-batch-size 1 \
  --min-seg-sec 3 \
  --max-seg-sec 6 \
  --overwrite
```

`vllm` 在 `requirements.txt` 中只会在 Linux/WSL 环境安装。原生 Windows 环境建议使用 `transformers` 后端。

常用参数：

- `--input`: 单个音频文件或音频目录。
- `--output-dir`: 输出目录，包含音频片段、转写、trace、manifest 和 VoxCPM 风格 JSONL。
- `--model-path`: ASR 模型路径。
- `--aligner-path`: forced aligner 模型路径。
- `--language`: 可选语言提示，会传给 ASR 后端。
- `--dry-run`: 运行流程但不写最终音频片段。
- `--enable-punctuation-correction`: 启用规则标点修正。

## 最终音量归一化（可选）

`slide_rule` 生成短训练片段后，可以按片段测量响度，把安全片段归一化到固定 LUFS
目标，并写入新的数据集目录：

```cmd
python utils/normalize_corpus_volume.py \
  --input sliced_segments \
  --output normalized_segments \
  --target-lufs -20 \
  --max-volume-change-db 12 \
  --overwrite
```

归一化工具会读取 `*_voxcpm.jsonl`，只处理其中列出的片段音频。需要过大向上增益、
归一化后可能削波、有效语音太少，或片段内部音量变化过大的行会被剪掉；过响片段可以被任意幅度衰减。
被剪掉的行写入 `pruned_volume_segments.jsonl`，报告写入
`volume_analysis.csv` 和 `normalized_manifest.csv`。

## 过滤说话人异常片段（可选）

如果数据集应该主要来自同一个说话人，可以用 SpeechBrain ECAPA 说话人嵌入对 VoxCPM JSONL
做保守过滤：

```cmd
python speaker_outlier_filter/filter_speaker_outliers.py \
  --input-jsonl audios/zhouquanquan_final_normalized/zhouquanquan.jsonl \
  --output-jsonl audios/zhouquanquan_final_normalized/zhouquanquan_speaker_filtered.jsonl \
  --dataset-root audios/zhouquanquan_final_normalized \
  --device cuda:0 \
  --overwrite
```

工具会保留原始音频文件，只写新的 JSONL 和审计报告。过滤结果写入
`zhouquanquan_speaker_filtered.jsonl`，被剪掉的行写入
`pruned_speaker_outliers.jsonl`，相似度明细和摘要分别写入
`speaker_similarity_report.csv` 与 `speaker_filter_summary.json`。

## 输出结果

`slide_rule` 输出目录通常包含：

- `audios/`: 保留下来的切分音频。
- `transcripts/`: 每个音频片段对应的转写文本。
- `traces/`: 每个阶段和每个片段的调试 trace。
- `discarded_audios/`: 因时长或其他规则被丢弃的片段。
- `prepass/`: ASR、时间戳、VAD chunk、停顿统计等预处理结果。
- `intermediate/`: 各阶段完整中间结果。
- `manifest.tsv`: 训练数据清单。
- `summary.json`: 本次运行的总体统计和配置。
- `<input_stem>_voxcpm.jsonl`: VoxCPM 风格数据文件。

更详细的切分机制说明见 [`technical_report.md`](technical_report.md)。
