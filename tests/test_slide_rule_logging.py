from __future__ import annotations

import json
import logging

import numpy as np

from slide_rule.config import RuleWorkflowConfig


def _reset_slide_rule_logger() -> None:
    for name in ("slide_rule", "slicing_utils"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(logging.NOTSET)
        logger.propagate = True


def test_cli_keeps_stdout_as_json(tmp_path, monkeypatch, capsys):
    _reset_slide_rule_logger()
    input_path = tmp_path / "input.wav"
    input_path.write_bytes(b"not real audio")
    output_dir = tmp_path / "out"

    from slide_rule import cli
    import slide_rule.pipeline as pipeline

    expected_summary = {
        "pipeline_mode": "rule",
        "source_audio_count": 1,
        "kept_segment_count": 0,
    }
    monkeypatch.setattr(pipeline, "run_rule_workflow_sync", lambda config: expected_summary)

    rc = cli.main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--model-path",
            "model",
            "--aligner-path",
            "aligner",
        ]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out) == expected_summary


def test_pipeline_logs_discovery_source_and_final(tmp_path, monkeypatch, caplog):
    _reset_slide_rule_logger()
    source = tmp_path / "sample.wav"
    source.write_bytes(b"not real audio")
    output_dir = tmp_path / "out"

    import slide_rule.pipeline as pipeline
    from slicing_utils.preprocess import empty_prepass

    class DummyAsrBackend:
        def __init__(self, **kwargs):
            pass

    def fake_load_audio(path, target_sample_rate):
        return np.zeros(1600, dtype=np.float32), 16000, 0.1

    def fake_compute_full_prepass(*args, **kwargs):
        base = empty_prepass()
        return pipeline.FullPrepass(
            strong=base.strong,
            weak=base.weak,
            no_punc=base.no_punc,
            global_chars=[],
            full_text="",
            warnings=[],
            chunk_spans=[],
            total_speech_sec=0.0,
            purged_count=0,
        )

    monkeypatch.setattr(pipeline, "AsrBackend", DummyAsrBackend)
    monkeypatch.setattr(pipeline, "load_audio_mono", fake_load_audio)
    monkeypatch.setattr(pipeline, "compute_full_prepass", fake_compute_full_prepass)

    cfg = RuleWorkflowConfig(
        input_path=source,
        output_dir=output_dir,
        model_path="model",
        aligner_path="aligner",
    )

    caplog.set_level(logging.INFO)
    summary = pipeline.run_rule_workflow_sync(cfg)

    messages = [record.getMessage() for record in caplog.records]
    assert summary["per_source"][0]["status"] == "asr_empty"
    assert any("Discovered 1 source audio file(s)" in message for message in messages)
    assert any("Source 1/1 started" in message for message in messages)
    assert any("Source 1/1 asr_empty" in message for message in messages)
    assert any("Finished slide_rule" in message for message in messages)
