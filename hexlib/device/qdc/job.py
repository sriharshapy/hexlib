# hexlib/device/qdc/job.py
"""Submit a staged artifact to Qualcomm Device Cloud and detect completion.

THREE MEASURED FACTS, do not re-derive or contradict them:

  1. DEVICE = "SM8650", TARGET_ID = 3625030. That device's CDSP reads
     ARCH_VER == 0x8c75, bit-identical to the simulator -- no microarch
     confound between stage 1 and stage 3.
  2. NEVER poll get_job_status or get_jobs_list for completion.
     get_job_status returns state=None on this account. get_jobs_list
     lagged more than 30 minutes on both jobs observed. Completion is
     detected by the *appearance* of TestLogs/results.xml among a job's
     log files. A job that ran zero tests once reported passing on this
     account because something declared success on weaker evidence than
     that -- wait() exists to make that impossible.
  3. Artifact is a zip, TestFramework.APPIUM, entry_script=None, extracted
     at /qdc/appium, logs collected from /data/local/tmp/QDC_logs. On-farm
     scripts have a plain `adb`.

CREDENTIALS. The QDC API key is the operator's own. It is read from the
QDC_API_KEY environment variable, or from ~/.qdc_api_key as a fallback, and
is never logged, never printed (including in an exception message), and
never committed. There is no code path in this module that holds a
credential on anyone's behalf.

TIMEOUTS. submit() takes timeout_min as a required keyword argument with no
default, refuses values outside 1..240, and never guesses on the caller's
behalf. A runaway job spends real money; a missing timeout must be a
TypeError raised by Python itself before this module runs a single line.

qualcomm_device_cloud_sdk IS IMPORTED LAZILY. Every function in this module
that needs it imports it inside its own body, not at module scope, so that
importing hexlib.device.qdc.job -- and running every test in
hexlib/tests/test_qdc.py -- works whether or not that package is installed.
The real SDK is only required once code actually reaches the network, which
none of the offline tests do.
"""
from __future__ import annotations

import os
import pathlib
import time
import types

DEVICE = "SM8650"
TARGET_ID = 3625030
POLL_S = 30
RESULTS_MARKER = "TestLogs/results.xml"

_MIN_TIMEOUT_MIN = 1
_MAX_TIMEOUT_MIN = 240

_API_KEY_ENV = "QDC_API_KEY"
_BASE_URL_ENV = "QDC_BASE_URL"


class QdcError(Exception):
    """Anything that must stop a submission before it spends device
    minutes: a bad timeout, a missing credential, a missing artifact, or
    the API itself refusing the request."""


def _key_file() -> pathlib.Path | None:
    p = pathlib.Path.home() / ".qdc_api_key"
    return p if p.is_file() else None


def _api_key() -> str:
    """The key is the operator's own, from the environment or
    ~/.qdc_api_key. Never committed, never in CI, never held on anyone's
    behalf, and never included in any error this function raises."""
    key = os.environ.get(_API_KEY_ENV)
    if not key:
        f = _key_file()
        key = f.read_text().strip() if f else None
    if not key:
        raise QdcError(
            f"no QDC credential: set {_API_KEY_ENV} or create ~/.qdc_api_key. "
            "Credentials are personal and are never stored in this repository."
        )
    return key


def _base_url() -> str:
    url = os.environ.get(_BASE_URL_ENV)
    if not url:
        raise QdcError(
            f"no QDC endpoint: set {_BASE_URL_ENV} to this account's API base URL."
        )
    return url


def _client():
    """Build an authenticated SDK client. Imports the real package lazily
    so that nothing above this line requires it to be installed."""
    from qualcomm_device_cloud_sdk import AuthenticatedClient

    return AuthenticatedClient(base_url=_base_url(), token=_api_key())


# --- qdc_api: a thin, monkeypatchable facade over the real SDK ------------
#
# Tests patch attributes directly onto this namespace (e.g.
# `job.qdc_api.get_job_log_files = fake`) without the real
# qualcomm_device_cloud_sdk package ever being imported. Each wrapper below
# imports the real SDK lazily, inside itself, for the same reason job()
# does: importing this module must never require the package to exist.
#
# NOTE ON PRODUCTION READINESS: get_job_log_files and get_job_status are
# exercised against the real /jobs/{id}/logs and /jobs/{id} endpoints and
# their response shapes are confirmed from the installed SDK's models
# (JobLogsType0.filename, JobType0.state). submit_job and upload_artifact
# below are best-effort against the same models but have never been run
# against a live account -- every historical submission from this codebase
# has gone through the QDC web console instead. Confirm field names end to
# end on a single, cheap, short-timeout dry run before trusting this path
# with anything that matters.


def _real_get_job_log_files(client, job_id):
    from qualcomm_device_cloud_sdk.api.jobs import get_jobs_job_id_logs

    return get_jobs_job_id_logs.sync(job_id=job_id, client=client) or []


def _real_get_job_status(client, job_id):
    from qualcomm_device_cloud_sdk.api.jobs import get_jobs_job_id

    return get_jobs_job_id.sync(job_id=job_id, client=client)


def _real_upload_artifact(client, zip_path: str) -> str:
    from qualcomm_device_cloud_sdk.models.artifact_type import ArtifactType
    from qualcomm_device_cloud_sdk.models.post_artifacts_upload_body import (
        PostArtifactsUploadBody,
    )
    from qualcomm_device_cloud_sdk.types import File
    from qualcomm_device_cloud_sdk.api.artifacts import post_artifacts_upload

    name = os.path.basename(zip_path)
    with open(zip_path, "rb") as fh:
        body = PostArtifactsUploadBody(file=File(payload=fh, file_name=name))
        result = post_artifacts_upload.sync(
            client=client,
            body=body,
            filename=name,
            artifact_type=ArtifactType.TESTPACKAGE,
        )
    uuid = getattr(result, "uuid", None)
    if not uuid:
        raise QdcError("QDC accepted the artifact upload but returned no uuid")
    return uuid


def _real_submit_job(client, artifact_uuid: str, timeout_min: int):
    from qualcomm_device_cloud_sdk.models.create_job_type_0 import CreateJobType0
    from qualcomm_device_cloud_sdk.models.job_type import JobType
    from qualcomm_device_cloud_sdk.models.job_mode import JobMode
    from qualcomm_device_cloud_sdk.models.test_framework import TestFramework
    from qualcomm_device_cloud_sdk.api.jobs import post_jobs

    body = CreateJobType0(
        target_id=str(TARGET_ID),
        job_type=JobType.AUTOMATED,
        job_mode=JobMode.APPLICATION,
        timeout_in_minutes=timeout_min,
        test_framework=TestFramework.APPIUM,
        entry_script=None,
        job_artifacts=[artifact_uuid],
    )
    return post_jobs.sync(client=client, body=body)


def _real_download_log_file(client, job_id, filename: str) -> bytes:
    from qualcomm_device_cloud_sdk.api.jobs import get_jobs_download_logs

    resp = get_jobs_download_logs.sync_detailed(
        job_id=job_id, client=client, filename=filename
    )
    return resp.content


qdc_api = types.SimpleNamespace(
    get_job_log_files=_real_get_job_log_files,
    get_job_status=_real_get_job_status,
    upload_artifact=_real_upload_artifact,
    submit_job=_real_submit_job,
    download_log_file=_real_download_log_file,
)


def submit(zip_path: str, *, timeout_min: int) -> int:
    """Submit `zip_path` (from artifact.stage) as a job on TARGET_ID and
    return the job id. timeout_min is required -- there is no default --
    and must be in 1..240; a runaway job spends real money."""
    if not _MIN_TIMEOUT_MIN <= timeout_min <= _MAX_TIMEOUT_MIN:
        raise QdcError(
            f"timeout_min must be {_MIN_TIMEOUT_MIN}..{_MAX_TIMEOUT_MIN}, "
            f"got {timeout_min}"
        )
    if not os.path.isfile(zip_path):
        raise QdcError(f"artifact not found: {zip_path}")

    client = _client()
    artifact_uuid = qdc_api.upload_artifact(client, zip_path)
    job = qdc_api.submit_job(client, artifact_uuid, timeout_min)
    job_id = getattr(job, "job_id", None)
    if job_id is None:
        raise QdcError("QDC accepted the submission but returned no job_id")
    return job_id


def _has_results(files) -> bool:
    return any(RESULTS_MARKER in (getattr(f, "filename", "") or "") for f in files)


def wait(job_id: int, cap_s: int = 1800) -> bool:
    """Block until TestLogs/results.xml appears among job_id's log files,
    or until cap_s seconds have passed.

    Returns True only once results.xml has actually appeared -- never a
    guess. Returns False at the cap rather than hanging forever; False at
    the cap must never be confused with success, and nothing here lets it
    be. Never touches get_job_status or the jobs list: see the module
    docstring for why.
    """
    client = _client()
    deadline = time.monotonic() + cap_s
    while True:
        files = qdc_api.get_job_log_files(client, job_id)
        if _has_results(files):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_S)


def fetch(job_id: int, dest: str) -> list[str]:
    """Download every log file QDC has for job_id into dest, and return the
    local paths written. Only meaningful after wait() has returned True --
    fetching before results.xml exists proves nothing."""
    client = _client()
    files = qdc_api.get_job_log_files(client, job_id)
    os.makedirs(dest, exist_ok=True)

    paths = []
    for f in files:
        name = getattr(f, "filename", None)
        if not name:
            continue
        data = qdc_api.download_log_file(client, job_id, name)
        local = os.path.join(dest, os.path.basename(name))
        with open(local, "wb") as fh:
            fh.write(data)
        paths.append(local)
    return paths
