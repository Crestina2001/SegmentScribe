from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from slicing_utils.asr import CharToken
from slicing_utils.preprocess import empty_prepass
from slide_LLM.config import LLMWorkflowConfig
from slide_LLM.punctuation import LLMPunctuationResult
from slide_LLM.rough_cut import Phase3Result


class _DummyAsrBackend:
    def __init__(self, **kwargs):
        pass


class _FakeUnifiedClient:
    def __init__(self, *, env_path=None):
        self.closed = False

    async def send_prompt(self, *args, **kwargs):
        return SimpleNamespace(text="ok", tool_calls=[])

    async def close(self):
        self.closed = True


def _prepass(text: str = "a"):
    base = empty_prepass()
    return SimpleNamespace(
        strong=base.strong,
        weak=base.weak,
        no_punc=base.no_punc,
        global_chars=[CharToken(idx=0, char="a", start_sec=0.0, end_sec=0.1)],
        full_text=text,
        warnings=[],
        chunk_spans=[],
        total_speech_sec=0.1,
        purged_count=0,
        pause_ms_by_token={},
        to_stream_dump=lambda: {
            "summary": {},
            "full_text": text,
            "chunks": [],
            "global_chars": [{"idx": 0, "char": "a", "start_sec": 0.0, "end_sec": 0.1}],
            "warnings": [],
            "pause_ms_by_token": {},
        },
    )


def _base_config(input_path, output_dir, *, asr_max_batch_size=1, llm_concurrency=1):
    return LLMWorkflowConfig(
        input_path=input_path,
        output_dir=output_dir,
        model_path="model",
        aligner_path="aligner",
        llm_model="fake",
        asr_max_batch_size=asr_max_batch_size,
        llm_concurrency=llm_concurrency,
    )


@pytest.mark.asyncio
async def test_slide_llm_batches_asr_chunks_across_sources(tmp_path, monkeypatch):
    source_dir = tmp_path / "in"
    source_dir.mkdir()
    for name in ("a.wav", "b.wav", "c.wav"):
        (source_dir / name).write_bytes(b"not audio")

    import slide_LLM.pipeline as pipeline
    from slide_rule.thin_cut import Phase4Result
    from slicing_utils.filter_write import Phase5Result

    lock = threading.Lock()
    counts = {
        "llm_active": 0,
        "max_llm": 0,
    }
    asr_batches: list[list[int]] = []
    events: list[tuple[str, str]] = []

    def fake_load_audio(path, target_sample_rate):
        source_id = {"a.wav": 1, "b.wav": 2, "c.wav": 3}[path.name]
        return np.full(1600, source_id, dtype=np.float32), 16000, 0.1

    def fake_prepare_full_prepass_plan(audio, sample_rate, **kwargs):
        source_id = int(audio[0])
        if source_id == 1:
            chunks = [(0, 10), (10, 20), (20, 30)]
        else:
            chunks = [(0, 10)]
        return SimpleNamespace(chunks=chunks)

    def fake_asr_only(self, windows):
        ids = [int(audio[0]) for audio, _sample_rate in windows]
        with lock:
            asr_batches.append(ids)
            events.append(("asr_batch", ",".join(str(item) for item in ids)))
        time.sleep(0.03)
        return [
            SimpleNamespace(text="a", language="zh")
            for _audio, _sample_rate in windows
        ]

    def fake_align_draft_chunk_windows(windows, drafts, asr_backend):
        return [
            (
                draft.text,
                [CharToken(idx=0, char="a", start_sec=0.0, end_sec=0.1)],
                draft.span,
            )
            for draft in drafts
        ]

    class TrackingClient(_FakeUnifiedClient):
        async def send_prompt(self, *args, **kwargs):
            with lock:
                counts["llm_active"] += 1
                counts["max_llm"] = max(counts["max_llm"], counts["llm_active"])
                events.append(("llm_start", "event-loop"))
            await pipeline.asyncio.sleep(0.08)
            with lock:
                counts["llm_active"] -= 1
                events.append(("llm_end", "event-loop"))
            return SimpleNamespace(text="ok", tool_calls=[])

    async def fake_punctuation(prepass, *, llm_client, model, provider=None):
        await llm_client.send_prompt(None, "punctuation", model=model)
        return LLMPunctuationResult(
            corrected_full_text=prepass.full_text,
            corrected_token_to_positions={0: [0]},
        )

    async def fake_rough_cut(**kwargs):
        await kwargs["llm_client"].send_prompt(None, "rough", model=kwargs["model"])
        return Phase3Result()

    monkeypatch.setattr(pipeline, "AsrBackend", _DummyAsrBackend)
    monkeypatch.setattr(_DummyAsrBackend, "transcribe_windows_asr_only", fake_asr_only, raising=False)
    monkeypatch.setattr(pipeline, "UnifiedClient", TrackingClient)
    monkeypatch.setattr(pipeline, "load_audio_mono", fake_load_audio)
    monkeypatch.setattr(pipeline, "prepare_full_prepass_plan", fake_prepare_full_prepass_plan)
    monkeypatch.setattr(pipeline, "align_draft_chunk_windows", fake_align_draft_chunk_windows)
    monkeypatch.setattr(pipeline, "assemble_full_prepass", lambda *args: _prepass())
    monkeypatch.setattr(pipeline, "run_llm_punctuation_phase", fake_punctuation)
    monkeypatch.setattr(pipeline, "run_llm_rough_cut_phase", fake_rough_cut)
    monkeypatch.setattr(pipeline, "run_thin_cut_phase", lambda **kwargs: Phase4Result())
    monkeypatch.setattr(pipeline, "run_filter_write_phase", lambda **kwargs: Phase5Result())

    cfg = _base_config(
        source_dir,
        tmp_path / "out",
        asr_max_batch_size=4,
        llm_concurrency=2,
    )

    summary = await pipeline.run_llm_workflow(cfg)

    assert max(len(batch) for batch in asr_batches) <= 4
    assert any(len(batch) == 4 and set(batch) == {1, 2} for batch in asr_batches)
    assert counts["max_llm"] <= 2
    assert [item["source_path"] for item in summary["per_source"]] == [
        str(source_dir / "a.wav"),
        str(source_dir / "b.wav"),
        str(source_dir / "c.wav"),
    ]
    assert events


def test_slide_llm_rejects_duplicate_source_stems(tmp_path):
    from slide_LLM.pipeline import _raise_for_duplicate_stems

    first = tmp_path / "one" / "sample.wav"
    second = tmp_path / "two" / "sample.mp3"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"a")
    second.write_bytes(b"b")

    with pytest.raises(SystemExit, match="Duplicate source filename stems"):
        _raise_for_duplicate_stems([first, second])


def test_slide_llm_cli_validates_pool_sizes(tmp_path):
    from slide_LLM import cli

    source = tmp_path / "input.wav"
    source.write_bytes(b"not audio")
    base = [
        "--input",
        str(source),
        "--output-dir",
        str(tmp_path / "out"),
        "--model-path",
        "model",
        "--aligner-path",
        "aligner",
        "--llm-model",
        "fake",
    ]

    args = cli.build_parser().parse_args(base + ["--asr-max-batch-size", "2", "--llm-concurrency", "4"])
    config = cli.args_to_config(args)
    assert config.asr_max_batch_size == 2
    assert config.llm_concurrency == 4

    bad_asr = cli.build_parser().parse_args(base + ["--asr-max-batch-size", "0"])
    with pytest.raises(SystemExit, match="--asr-max-batch-size must be > 0"):
        cli.args_to_config(bad_asr)

    bad_llm = cli.build_parser().parse_args(base + ["--llm-concurrency", "0"])
    with pytest.raises(SystemExit, match="--llm-concurrency must be > 0"):
        cli.args_to_config(bad_llm)
