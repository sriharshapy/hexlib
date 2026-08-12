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
     log files.

     WHAT wait() IS AND IS NOT. It is a COMPLETION DETECTOR, not a verdict.
     `wait() -> True` means one thing and only one thing: a log file whose
     name ends in TestLogs/results.xml showed up. It does not open that file,
     so it cannot distinguish a real report from a zero-byte placeholder, and
     an earlier version of this docstring claiming "wait() exists to make
     [a false pass] impossible" was overclaiming: it makes DECLARING
     COMPLETION EARLY impossible, which is a different (and narrower) thing.
     A job that ran zero tests once reported passing on this account, and the
     check that actually rules that out lives in
     `hexlib/cli.py::_qdc_check_results` -- it parses the report, requires
     tests > 0 with no failures/errors/skips, and requires hexlib's own
     measurement lines in the fetched logs. Any new caller of wait() needs
     that check too; wait()'s True is its precondition, never its conclusion.
  3. Artifact is a zip, TestFramework.APPIUM, entry_script=None, extracted
     at /qdc/appium, logs collected from /data/local/tmp/QDC_logs. On-farm
     scripts have a plain `adb`.

CREDENTIALS.
  - QDC_API_KEY (required, or ~/.qdc_api_key as a fallback): the operator's
    own API key. Never logged, never printed (including in an exception
    message), never committed, and never held on anyone's behalf.
  - QDC_BASE_URL (optional override): defaults to Qualcomm's own public
    endpoint, qualcomm_device_cloud_sdk.api.qdc_api.API_BASE_URL. Set this
    only for a private or government tenant with a different endpoint --
    the default is correct for the account this module was built against.

AUTH HEADER: this module authenticates through the SDK's own
get_public_api_client_using_api_key(), not a hand-rolled client, so the
header scheme (a raw Authorization header plus X-QCOM-TokenType: apikey --
NOT an OAuth-style token prefix) comes from Qualcomm's code. See _client()
below.

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

# PUBLIC on purpose: hexlib/cli.py range-checks `--timeout-min` against these
# before it runs a full SDK build, and it must not respell the bounds. This
# module stays the single authority for them.
MIN_TIMEOUT_MIN = 1
MAX_TIMEOUT_MIN = 240

_API_KEY_ENV = "QDC_API_KEY"
_BASE_URL_ENV = "QDC_BASE_URL"

# Labels the SDK's own client constructor requires. None of these are
# secrets -- they identify the caller to QDC's own logging, nothing more.
_APP_NAME = "hexlib"
_CLIENT_TYPE = "Python"


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


def _on_behalf_of() -> str:
    """A caller label for QDC's own logging, not a credential. Best-effort
    from the environment; falls back to a fixed string rather than failing
    a submission over a cosmetic field."""
    return (
        os.environ.get("QDC_OPERATOR")
        or os.environ.get("USERNAME")
        or os.environ.get("USER")
        or "hexlib"
    )


def _base_url_override() -> str | None:
    """QDC_BASE_URL is optional: unset, the SDK's own public endpoint
    (qualcomm_device_cloud_sdk.api.qdc_api.API_BASE_URL) is used, via the
    SDK's own client constructor. Set only for a private or government
    tenant with a different endpoint."""
    return os.environ.get(_BASE_URL_ENV)


def _client():
    """Build the QDC client the way the SDK itself does, so the auth header
    scheme, the default base URL, and any header Qualcomm adds later all
    come from their code, not a hand-rolled guess.

    Imports the real package lazily so that nothing above this line, and no
    test in hexlib/tests/test_qdc.py, requires it to be installed.
    """
    from qualcomm_device_cloud_sdk.api import qdc_api as _vendor

    key = _api_key()
    override = _base_url_override()
    if not override:
        return _vendor.get_public_api_client_using_api_key(
            app_name_header=_APP_NAME,
            on_behalf_of_header=_on_behalf_of(),
            client_type_header=_CLIENT_TYPE,
            api_key_header=key,
        )

    # QDC_BASE_URL was set: same header recipe as
    # get_public_api_client_using_api_key (copied from
    # qualcomm_device_cloud_sdk/api/qdc_api.py since that function does not
    # take a base_url argument), pointed at a different endpoint. This path
    # is exercised even less than the default one -- confirm it against the
    # tenant's own documentation before trusting it with a real submission.
    from qualcomm_device_cloud_sdk import Client

    return Client(
        base_url=override,
        headers={
            "Authorization": key,
            "X-QCOM-TokenType": "apikey",
            "X-QCOM-AppName": _APP_NAME,
            "X-QCOM-ClientType": _CLIENT_TYPE,
            "X-QCOM-OnBehalfOf": _on_behalf_of(),
        },
    )


# --- qdc_api: a thin, monkeypatchable facade over the SDK's own qdc_api ---
#
# Tests patch attributes directly onto this namespace (e.g.
# `job.qdc_api.get_job_log_files = fake`) without the real
# qualcomm_device_cloud_sdk package ever being imported. Each wrapper below
# imports the real SDK lazily, inside itself, for the same reason _client()
# does: importing this module must never require the package to exist.
#
# EVERY WRAPPER BELOW DELEGATES TO qualcomm_device_cloud_sdk.api.qdc_api,
# Qualcomm's own high-level module (get_job_log_files, get_job_status,
# upload_file, submit_job, download_job_log_files, get_jobs_list) rather
# than hand-rolling calls to the low-level generated endpoints. That module
# already builds the right request bodies and already raises on a non-200
# response (via its own try_call helper) -- there is no reason to duplicate
# that logic here and every reason not to, since duplicating it is exactly
# how field names or status-code handling drift out of sync with what
# Qualcomm ships.
#
# NOTE ON PRODUCTION READINESS: get_job_log_files and get_job_status go
# through the same vendor code Qualcomm's own client uses, so their shapes
# are as trustworthy as the SDK itself. upload_file/submit_job have never
# been exercised against a live account from this codebase -- every
# historical submission went through the QDC web console instead. Confirm
# end to end on a single, cheap, short-timeout dry run before trusting this
# path with anything that matters. get_jobs_list is wired up here only so a
# test can prove it is never called by wait() -- see the guard test in
# hexlib/tests/test_qdc.py and fact 2 in this module's docstring.


def _real_get_job_log_files(client, job_id):
    from qualcomm_device_cloud_sdk.api import qdc_api as _vendor

    return _vendor.get_job_log_files(client, job_id) or []


def _real_get_job_status(client, job_id):
    from qualcomm_device_cloud_sdk.api import qdc_api as _vendor

    return _vendor.get_job_status(client, job_id)


def _real_get_jobs_list(client, page_number=0, page_size=20):
    # Wired up for completeness and for the guard test only. wait() must
    # never call this -- get_jobs_list lagged more than 30 minutes on both
    # jobs observed on this account.
    from qualcomm_device_cloud_sdk.api import qdc_api as _vendor

    return _vendor.get_jobs_list(client, page_number, page_size)


def _real_upload_artifact(client, zip_path: str) -> str:
    from qualcomm_device_cloud_sdk.api import qdc_api as _vendor
    from qualcomm_device_cloud_sdk.models.artifact_type import ArtifactType

    # TESTSCRIPT, NOT TESTPACKAGE, and this was measured rather than reasoned.
    # Uploading the same zip as TESTPACKAGE authenticates and uploads fine and
    # is then refused at submission with HTTP 400 "Appium tests requires 1 test
    # script and at most 1 test package" -- the package is the OPTIONAL half and
    # the script is the required one, so one package and zero scripts is not a
    # valid Appium job. llama.cpp's own QDC runner
    # (scripts/snapdragon/qdc/run_qdc_jobs.py) uploads its Appium zip as
    # TESTSCRIPT against this same API, which is the working reference this was
    # matched to. Costs no device minutes to get wrong -- the job is rejected
    # before dispatch -- but it does cost a round trip.
    uuid = _vendor.upload_file(client, zip_path, ArtifactType.TESTSCRIPT)
    if not uuid:
        raise QdcError("QDC accepted the artifact upload but returned no uuid")
    return uuid


def _real_submit_job(client, artifact_uuid: str, timeout_min: int):
    from qualcomm_device_cloud_sdk.api import qdc_api as _vendor
    from qualcomm_device_cloud_sdk.models.job_type import JobType
    from qualcomm_device_cloud_sdk.models.job_mode import JobMode
    from qualcomm_device_cloud_sdk.models.test_framework import TestFramework

    return _vendor.submit_job(
        client,
        target_id=TARGET_ID,
        job_name="hexlib",
        external_job_id=None,
        job_type=JobType.AUTOMATED,
        job_mode=JobMode.APPLICATION,
        timeout=timeout_min,
        test_framework=TestFramework.APPIUM,
        entry_script=None,
        job_artifacts=[artifact_uuid],
        monkey_events=None,
        monkey_session_timeout=None,
    )


def _real_download_log_file(client, filename: str, local_path: str) -> bool:
    from qualcomm_device_cloud_sdk.api import qdc_api as _vendor

    return bool(_vendor.download_job_log_files(client, filename, local_path))


qdc_api = types.SimpleNamespace(
    get_job_log_files=_real_get_job_log_files,
    get_job_status=_real_get_job_status,
    get_jobs_list=_real_get_jobs_list,
    upload_artifact=_real_upload_artifact,
    submit_job=_real_submit_job,
    download_log_file=_real_download_log_file,
)


def submit(zip_path: str, *, timeout_min: int) -> int:
    """Submit `zip_path` (from artifact.stage) as a job on TARGET_ID and
    return the job id. timeout_min is required -- there is no default --
    and must be in 1..240; a runaway job spends real money."""
    if not MIN_TIMEOUT_MIN <= timeout_min <= MAX_TIMEOUT_MIN:
        raise QdcError(
            f"timeout_min must be {MIN_TIMEOUT_MIN}..{MAX_TIMEOUT_MIN}, "
            f"got {timeout_min}"
        )
    if not os.path.isfile(zip_path):
        raise QdcError(f"artifact not found: {zip_path}")

    client = _client()
    artifact_uuid = qdc_api.upload_artifact(client, zip_path)
    job_id = qdc_api.submit_job(client, artifact_uuid, timeout_min)
    if job_id is None:
        raise QdcError("QDC accepted the submission but returned no job_id")
    return job_id


def _results_filename(name: object) -> bool:
    """True if `name` IS the results file, by path suffix -- not merely a name
    that CONTAINS the marker somewhere. `RESULTS_MARKER in name` also matched
    `TestLogs/results.xml.part` and `TestLogs/results.xml.tmp`, i.e. exactly
    the half-written intermediate whose appearance is the one thing a
    completion detector must not fire on. Separators are normalized because
    QDC's own filenames use forward slashes and nothing guarantees a future
    field will."""
    if not isinstance(name, str) or not name:
        return False
    return name.replace("\\", "/").endswith(RESULTS_MARKER)


def _has_results(files) -> bool:
    """PURELY A FILENAME TEST -- it never opens anything. A zero-byte
    results.xml satisfies it. That is deliberate (this module cannot read a
    file it has not downloaded yet) and it is why `wait()` is not a verdict;
    see `wait()` and fact 2 in the module docstring."""
    return any(_results_filename(getattr(f, "filename", None)) for f in files)


def wait(job_id: int, cap_s: int = 1800) -> bool:
    """Block until a log file named TestLogs/results.xml APPEARS among
    job_id's log files, or until cap_s seconds have passed.

    WHAT True GUARANTEES, EXACTLY: that a file with that name now exists in
    QDC's log listing for this job. NOTHING MORE. This function does not
    download it, does not open it, does not parse it, and cannot tell a real
    JUnit report from a zero-byte file with the right name -- `_has_results`
    is a filename test (see its own docstring). True therefore means
    "finished, probably" and is a PRECONDITION for judging the job, never the
    judgement.

    WHERE THE REAL CHECK LIVES: `hexlib/cli.py::_qdc_check_results`, called by
    `_qdc_submit` after `fetch()`. It parses the report and requires
    tests > 0, no failures, no errors, no skips, and hexlib's own measurement
    lines (`hexlib: --self-test: PASS`, a positive `cycles_total=`) in the
    fetched logs. A caller that treats this function's True as a pass
    reintroduces this account's own history -- a job that ran zero tests and
    reported passing -- with the check one level further away.

    THE CAP IS THE CALLER'S TO CHOOSE, and the default is not a safe one for
    every job: 1800 s is SMALLER than the 240 minutes `submit()` accepts, so
    passing a job's own timeout through is the caller's responsibility (cli.py
    derives it in `_qdc_wait_cap_s`). Returns False at the cap rather than
    hanging forever; False must never be confused with success, and it also
    does not mean the job failed -- it means this function stopped watching,
    which is why cli.py fetches whatever logs exist before reporting it.

    Never touches get_job_status or the jobs list: see the module docstring.
    """
    client = _client()
    deadline = time.monotonic() + cap_s
    errors = 0
    while True:
        # A TRANSIENT API ERROR IS NOT A VERDICT. This call used to be
        # unguarded, so one HTTP 502 anywhere in ~60 polls raised straight out
        # of wait(), through _qdc_submit, and killed the CLI -- discarding a
        # job whose minutes were already spent, before anything was fetched or
        # checked. Observed on job 756206 (2026-08-12), and a second 502 hit
        # get_job_log_upload_status for 756159 the same afternoon, so this is
        # a property of the service rather than one bad moment. llama.cpp's
        # runner retries these calls for the same reason.
        #
        # Swallowed only until the cap, never forever, and counted so that
        # "the API was down the whole time" cannot masquerade as the ordinary
        # "no results appeared" answer -- those are different findings and the
        # caller is told which one it got.
        try:
            files = qdc_api.get_job_log_files(client, job_id)
            if _has_results(files):
                return True
        except Exception as e:  # noqa: BLE001 - any transport failure retries
            errors += 1
            print(
                f"qdc: polling job {job_id} log files failed "
                f"({errors} time(s)), retrying: {e}"
            )
        if time.monotonic() >= deadline:
            if errors:
                print(
                    f"qdc: gave up on job {job_id} after {cap_s}s with "
                    f"{errors} failed poll(s) -- if that is every poll, this "
                    f"is an API outage, not a job that produced nothing"
                )
            return False
        time.sleep(POLL_S)


def _local_log_path(dest: str, name: str) -> str | None:
    """Where QDC's log file `name` should land under `dest`, PRESERVING QDC's
    own directory structure, or None if `name` cannot be mapped safely.

    `os.path.join(dest, os.path.basename(name))` flattened it, and QDC's
    listings are not flat: `TestLogs/results.xml` and `logs/results.xml` both
    became `dest/results.xml`, so one silently overwrote the other and the
    returned `paths` list held two entries pointing at one file -- with
    cli.py's results check then reading whichever download happened to finish
    last. Flattening is not a cosmetic problem here: the one file this whole
    path is built to read is identified by that basename.

    Returns None for anything that would escape `dest` -- an absolute path, a
    drive letter, or a `..` component. QDC's own names have never looked like
    that, which is exactly why nothing would notice if one did.

    THE CHECK MUST NOT DEPEND ON THE HOST OS, and it did. This used
    `os.path.isabs` and `os.path.splitdrive`, both of which are the RUNNING
    platform's notion of a path. So `C:/Windows/System32/evil.txt` was refused
    on Windows and ACCEPTED on Linux -- where it is not absolute at all, merely
    a relative path whose first component happens to be named `C:` -- and the
    file landed in `dest/C:/Windows/System32/evil.txt`. The mirror case is
    `/etc/passwd`, absolute on POSIX and not on Windows.

    That asymmetry matters because the two ends run on different systems: the
    developer machine here is Windows, CI is Linux, and the name being screened
    arrives from a remote service. A traversal guard that only holds on the
    platform you happened to test on is the "claim of protection that protects
    nothing" pattern, in a place where the input is not ours.

    Both flavours are now consulted, so a name is refused if EITHER OS would
    read it as absolute or drive-qualified, regardless of where this runs.
    Found by CI on the first push after CI itself was repaired.
    """
    unified = name.replace("\\", "/").strip()
    if not unified:
        return None
    parts = [p for p in unified.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    for flavour in (pathlib.PureWindowsPath, pathlib.PurePosixPath):
        p = flavour(unified)
        if p.is_absolute() or p.anchor or p.drive:
            return None
    # A bare drive-relative name (`C:foo`) has no anchor under PureWindowsPath
    # but still names a drive, and a component containing `:` cannot be a
    # legitimate QDC log path segment on any platform we target.
    if any(":" in p for p in parts):
        return None
    return os.path.join(dest, *parts)


def fetch(job_id: int, dest: str) -> list[str]:
    """Download every log file QDC has for job_id into dest, MIRRORING QDC's
    own directory layout beneath it, and return the local paths written.

    Only meaningful after wait() has returned True -- and note that wait()
    returning True proves only that a file with the right NAME appeared, so
    even a full fetch is not a verdict: `hexlib/cli.py::_qdc_check_results`
    is what judges the job.
    """
    client = _client()
    files = qdc_api.get_job_log_files(client, job_id)
    os.makedirs(dest, exist_ok=True)

    paths = []
    for f in files:
        name = getattr(f, "filename", None)
        if not name or not isinstance(name, str):
            continue
        local = _local_log_path(dest, name)
        if local is None:
            continue
        os.makedirs(os.path.dirname(local), exist_ok=True)
        if qdc_api.download_log_file(client, name, local):
            paths.append(local)
    return paths
