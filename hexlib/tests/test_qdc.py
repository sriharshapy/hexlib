# hexlib/tests/test_qdc.py
"""QDC submission and completion detection, against a fake API client.

WHY COMPLETION IS DETECTED FROM LOG FILES. `get_job_status` returns
`state=None`; `get_jobs_list` lagged more than 30 minutes on both observed jobs.
Polling either means either hanging forever or declaring success early. A job
that ran zero tests once reported passing on this account, which is the failure
mode all of this exists to make impossible.
"""
import zipfile

import pytest

from hexlib.device.qdc import artifact, job


def test_target_is_sm8650_and_the_id_is_pinned():
    assert job.DEVICE == "SM8650"
    assert job.TARGET_ID == 3625030


def test_stage_produces_a_zip_containing_every_binary(tmp_path):
    (tmp_path / "hexlib_run").write_bytes(b"\x7fELF fake")
    (tmp_path / "libhexlib_skel.so").write_bytes(b"\x7fELF fake")
    test_py = tmp_path / "test_on_device.py"
    test_py.write_text("def test_x(): pass\n")
    z = artifact.stage(
        [str(tmp_path / "hexlib_run"), str(tmp_path / "libhexlib_skel.so")],
        str(test_py), str(tmp_path / "job"),
    )
    names = zipfile.ZipFile(z).namelist()
    assert any("hexlib_run" in n for n in names)
    assert any("libhexlib_skel.so" in n for n in names)
    assert any("test_on_device.py" in n for n in names)
    assert any("pytest.ini" in n for n in names)


def test_stage_refuses_a_missing_binary(tmp_path):
    with pytest.raises(artifact.StagingError, match="not found"):
        artifact.stage([str(tmp_path / "nope")], None, str(tmp_path / "job"))


def test_submission_requires_an_explicit_timeout(tmp_path):
    z = tmp_path / "a.zip"
    z.write_bytes(b"PK")
    with pytest.raises(TypeError):
        job.submit(str(z))  # timeout_min is required, not defaulted


def test_submission_refuses_a_zero_or_absurd_timeout(tmp_path, monkeypatch):
    z = tmp_path / "a.zip"
    z.write_bytes(b"PK")
    with pytest.raises(job.QdcError, match="timeout"):
        job.submit(str(z), timeout_min=0)
    with pytest.raises(job.QdcError, match="timeout"):
        job.submit(str(z), timeout_min=10000)


def test_wait_polls_log_files_and_never_job_status(monkeypatch):
    calls = {"logs": 0}

    class F:
        filename = "TestLogs/results.xml"

    def fake_logs(client, job_id):
        calls["logs"] += 1
        return [F()] if calls["logs"] >= 2 else []

    def boom(*a, **k):
        raise AssertionError("get_job_status must never be polled")

    monkeypatch.setattr(job, "_client", lambda: object())
    monkeypatch.setattr(job.qdc_api, "get_job_log_files", fake_logs, raising=False)
    monkeypatch.setattr(job.qdc_api, "get_job_status", boom, raising=False)
    monkeypatch.setattr(job, "POLL_S", 0)
    assert job.wait(1234, cap_s=10) is True
    assert calls["logs"] >= 2


def test_wait_returns_false_at_the_cap_rather_than_hanging(monkeypatch):
    monkeypatch.setattr(job, "_client", lambda: object())
    monkeypatch.setattr(job.qdc_api, "get_job_log_files",
                        lambda c, j: [], raising=False)
    monkeypatch.setattr(job, "POLL_S", 0)
    assert job.wait(1234, cap_s=0) is False


def test_a_job_with_no_results_xml_is_not_complete(monkeypatch):
    class F:
        filename = "TestLogs/logcat.txt"

    monkeypatch.setattr(job, "_client", lambda: object())
    monkeypatch.setattr(job.qdc_api, "get_job_log_files",
                        lambda c, j: [F()], raising=False)
    monkeypatch.setattr(job, "POLL_S", 0)
    assert job.wait(1234, cap_s=0) is False


def test_the_api_key_is_read_from_the_environment_not_committed(monkeypatch):
    monkeypatch.delenv("QDC_API_KEY", raising=False)
    monkeypatch.setattr(job, "_key_file", lambda: None)
    with pytest.raises(job.QdcError, match="QDC_API_KEY"):
        job._api_key()


def test_no_credential_appears_anywhere_in_the_source():
    import pathlib
    for p in pathlib.Path("hexlib/device").rglob("*.py"):
        src = p.read_text()
        assert "qdc_api_key" not in src.lower() or "environ" in src or "home()" in src
        assert "Bearer " not in src
