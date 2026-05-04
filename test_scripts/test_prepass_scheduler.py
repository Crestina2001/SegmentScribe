from __future__ import annotations

import asyncio
import time

import numpy as np

from slicing_utils.asr import WindowTranscript
from slicing_utils.prepass import AsyncPrepassScheduler
from slide_LLM.cli import args_to_config as llm_args_to_config
from slide_LLM.cli import build_parser as build_llm_parser
from slide_rule.cli import args_to_config as rule_args_to_config
from slide_rule.cli import build_parser as build_rule_parser


class FakeAlignerWorker:
    def __init__(self, owner: "FakeAsrBackend", worker_index: int) -> None:
        self.owner = owner
        self.worker_index = worker_index

    def align_window_transcripts(self, windows, transcripts):
        self.owner.aligner_batches.append((self.worker_index, len(windows)))
        time.sleep(0.02)
        return [
            WindowTranscript(
                text=transcript.text,
                language=transcript.language,
                chars=[],
                duration_sec=0.0,
            )
            for transcript in transcripts
        ]


class FakeAsrBackend:
    def __init__(self) -> None:
        self.asr_batches: list[int] = []
        self.aligner_batches: list[tuple[int, int]] = []
        self.aligner_worker_indices: list[int] = []

    def transcribe_windows_asr_only(self, windows):
        self.asr_batches.append(len(windows))
        return [
            WindowTranscript(
                text=f"text-{index}",
                language="",
                chars=[],
                duration_sec=0.0,
            )
            for index, _window in enumerate(windows)
        ]

    def create_aligner_worker(self, *, worker_index: int = 1):
        self.aligner_worker_indices.append(worker_index)
        return FakeAlignerWorker(self, worker_index)


def _audio(seconds: float = 3.0) -> np.ndarray:
    return np.zeros(int(16000 * seconds), dtype=np.float32)


def test_scheduler_uses_single_asr_worker_and_bounded_batches():
    async def run():
        backend = FakeAsrBackend()
        scheduler = AsyncPrepassScheduler(
            backend,
            asr_max_batch_size=2,
            aligner_num_workers=3,
            aligner_max_batch_size=2,
        )
        scheduler.start()
        try:
            chunks = [(0, 100), (100, 200), (200, 300), (300, 400), (400, 500)]
            await scheduler.run_source_prepass(
                audio=_audio(),
                sample_rate=16000,
                chunks=chunks,
                source_label="fake",
            )
        finally:
            await scheduler.close()
        return backend

    backend = asyncio.run(run())

    assert backend.aligner_worker_indices == [1, 2, 3]
    assert max(backend.asr_batches) <= 2
    assert max(batch_size for _worker_index, batch_size in backend.aligner_batches) <= 2


def test_rule_and_llm_cli_accept_shared_scheduler_knobs(tmp_path):
    rule_args = build_rule_parser().parse_args(
        [
            "--input",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "rule_out"),
            "--model-path",
            "asr",
            "--aligner-path",
            "aligner",
            "--source-concurrency",
            "2",
            "--aligner-num-workers",
            "3",
        ]
    )
    llm_args = build_llm_parser().parse_args(
        [
            "--input",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "llm_out"),
            "--model-path",
            "asr",
            "--aligner-path",
            "aligner",
            "--llm-model",
            "model",
            "--source-concurrency",
            "2",
            "--aligner-num-workers",
            "3",
        ]
    )

    rule_cfg = rule_args_to_config(rule_args)
    llm_cfg = llm_args_to_config(llm_args)

    assert rule_cfg.source_concurrency == 2
    assert rule_cfg.aligner_num_workers == 3
    assert llm_cfg.source_concurrency == 2
    assert llm_cfg.aligner_num_workers == 3


def test_cli_rejects_invalid_shared_scheduler_knobs(tmp_path):
    for parser in (build_rule_parser(), build_llm_parser()):
        args = [
            "--input",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--model-path",
            "asr",
            "--aligner-path",
            "aligner",
            "--source-concurrency",
            "0",
        ]
        if parser.prog == "python -m slide_LLM":
            args.extend(["--llm-model", "model"])
        parsed = parser.parse_args(args)
        try:
            if parser.prog == "python -m slide_LLM":
                llm_args_to_config(parsed)
            else:
                rule_args_to_config(parsed)
        except SystemExit:
            continue
        raise AssertionError("Expected invalid scheduler knobs to raise SystemExit")
