"""Logging utility: writes to a file under the log dir, no duplicate handlers."""

import logging

from agentstat.logging_utils import get_logger


def test_writes_log_file(tmp_path):
    log = get_logger("test-run", run_id="abc123", log_dir=tmp_path)
    log.info("hello world")
    for h in log.handlers:
        h.flush()
    log_file = tmp_path / "test-run-abc123.log"
    assert log_file.exists()
    assert "hello world" in log_file.read_text()


def test_no_file_handler_without_run_id():
    log = get_logger("test-no-file", run_id=None)
    assert not any(isinstance(h, logging.FileHandler) for h in log.handlers)


def test_idempotent_no_duplicate_handlers(tmp_path):
    log1 = get_logger("test-idem", run_id="r1", log_dir=tmp_path)
    n_after_first = len(log1.handlers)
    log2 = get_logger("test-idem", run_id="r1", log_dir=tmp_path)
    assert log1 is log2
    assert len(log2.handlers) == n_after_first  # no stacking
