# hexlib/tests/test_qdc.py
"""QDC submission and completion detection, against a fake API client.

WHY COMPLETION IS DETECTED FROM LOG FILES. `get_job_status` returns
`state=None`; `get_jobs_list` lagged more than 30 minutes on both observed jobs.
Polling either means either hanging forever or declaring success early. A job
that ran zero tests once reported passing on this account, which is the failure
mode all of this exists to make impossible.
"""
import ast
import collections
import math
import pathlib
import re
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


# --- 0-byte inputs -------------------------------------------------------
#
# `stage` checked existence only, and was VERIFIED to accept four 0-byte files
# and produce a perfectly submittable zip. A link or copy that fails part way
# leaves exactly that: a `libhexlib_skel.so` of length zero, present, correctly
# named, and completely unrunnable -- discovered on the device as a dlopen
# failure with no obvious cause, after the minutes are spent.


def test_stage_refuses_a_zero_byte_binary(tmp_path):
    (tmp_path / "hexlib_run").write_bytes(b"\x7fELF fake")
    (tmp_path / "libhexlib_skel.so").write_bytes(b"")        # a truncated link
    with pytest.raises(artifact.StagingError, match="0 bytes"):
        artifact.stage(
            [str(tmp_path / "hexlib_run"), str(tmp_path / "libhexlib_skel.so")],
            None, str(tmp_path / "job"),
        )


def test_stage_refuses_a_zero_byte_test_script(tmp_path):
    (tmp_path / "hexlib_run").write_bytes(b"\x7fELF fake")
    (tmp_path / "test_on_device.py").write_text("")
    with pytest.raises(artifact.StagingError, match="0 bytes"):
        artifact.stage(
            [str(tmp_path / "hexlib_run")],
            str(tmp_path / "test_on_device.py"), str(tmp_path / "job"),
        )


def test_stage_produces_no_zip_at_all_when_an_input_is_empty(tmp_path):
    """The refusal must happen BEFORE anything submittable exists on disk --
    a zip left behind by a failed staging run is a zip somebody can submit."""
    (tmp_path / "hexlib_run").write_bytes(b"")
    out_base = tmp_path / "job"
    with pytest.raises(artifact.StagingError):
        artifact.stage([str(tmp_path / "hexlib_run")], None, str(out_base))
    assert not (tmp_path / "job.zip").exists()


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


def test_wait_never_polls_the_jobs_list_either(monkeypatch):
    # get_jobs_list lagged more than 30 minutes on both jobs observed on
    # this account -- as dangerous as get_job_status, and nothing stops a
    # future edit from wiring it into wait() by mistake. This guard exists
    # so that edit fails a test instead of silently reintroducing the
    # exact failure mode this module was built to make impossible.
    calls = {"logs": 0}

    class F:
        filename = "TestLogs/results.xml"

    def fake_logs(client, job_id):
        calls["logs"] += 1
        return [F()] if calls["logs"] >= 2 else []

    def boom(*a, **k):
        raise AssertionError("get_jobs_list must never be polled")

    monkeypatch.setattr(job, "_client", lambda: object())
    monkeypatch.setattr(job.qdc_api, "get_job_log_files", fake_logs, raising=False)
    monkeypatch.setattr(job.qdc_api, "get_jobs_list", boom, raising=False)
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


# --- wait() is a completion detector, NOT a verdict ----------------------
#
# `_has_results` was `RESULTS_MARKER in filename`, a bare substring test, so
# `TestLogs/results.xml.part` -- the half-written intermediate whose appearance
# is the one thing a completion detector must not fire on -- counted as
# "finished". And even a genuine match proves only that a NAME appeared:
# `wait()` never opens the file, so a zero-byte results.xml satisfies it. That
# is why the real check lives in `hexlib/cli.py::_qdc_check_results`, and why
# job.py's docstrings now say so instead of claiming wait() makes a false pass
# impossible.


@pytest.mark.parametrize("name", [
    "TestLogs/results.xml.part",
    "TestLogs/results.xml.tmp",
    "TestLogs/results.xml.gz",
    "TestLogs/results.xmlx",
    "TestLogs/my_results.xml.bak",
])
def test_a_partially_written_results_file_is_not_completion(monkeypatch, name):
    class F:
        filename = None

    F.filename = name
    monkeypatch.setattr(job, "_client", lambda: object())
    monkeypatch.setattr(job.qdc_api, "get_job_log_files",
                        lambda c, j: [F()], raising=False)
    monkeypatch.setattr(job, "POLL_S", 0)
    assert job.wait(1234, cap_s=0) is False, (
        f"{name!r} is not TestLogs/results.xml -- a substring match on it "
        "declares a job complete off a half-written file"
    )


@pytest.mark.parametrize("name", [
    "TestLogs/results.xml",
    "job-1234/TestLogs/results.xml",
    "job-1234\\TestLogs\\results.xml",
])
def test_the_real_results_file_is_recognized_however_it_is_pathed(monkeypatch, name):
    class F:
        filename = None

    F.filename = name
    monkeypatch.setattr(job, "_client", lambda: object())
    monkeypatch.setattr(job.qdc_api, "get_job_log_files",
                        lambda c, j: [F()], raising=False)
    monkeypatch.setattr(job, "POLL_S", 0)
    assert job.wait(1234, cap_s=0) is True


def test_a_log_entry_with_no_filename_at_all_does_not_crash_wait(monkeypatch):
    class F:
        filename = None

    class G:
        pass

    monkeypatch.setattr(job, "_client", lambda: object())
    monkeypatch.setattr(job.qdc_api, "get_job_log_files",
                        lambda c, j: [F(), G()], raising=False)
    monkeypatch.setattr(job, "POLL_S", 0)
    assert job.wait(1234, cap_s=0) is False


# --- fetch() mirrors QDC's directory layout ------------------------------
#
# `os.path.join(dest, os.path.basename(name))` FLATTENED it. QDC's listings are
# not flat, so `TestLogs/results.xml` and `logs/results.xml` both became
# `dest/results.xml`: one silently overwrote the other, the returned `paths`
# held two entries pointing at ONE file, and cli.py's results check then read
# whichever download happened to land last. The one file this whole path exists
# to read is identified by that basename.


def _fake_downloads(monkeypatch, names):
    """A fake QDC that lists `names` and 'downloads' each by writing its own
    remote name into the local file, so a collision is detectable by content."""
    class F:
        def __init__(self, filename):
            self.filename = filename

    monkeypatch.setattr(job, "_client", lambda: object())
    monkeypatch.setattr(job.qdc_api, "get_job_log_files",
                        lambda c, j: [F(n) for n in names], raising=False)

    def fake_download(client, remote, local):
        with open(local, "w", encoding="utf-8") as f:
            f.write(remote)
        return True

    monkeypatch.setattr(job.qdc_api, "download_log_file", fake_download,
                        raising=False)


def test_fetch_does_not_let_two_logs_with_one_basename_overwrite_each_other(
    monkeypatch, tmp_path
):
    _fake_downloads(monkeypatch, ["TestLogs/results.xml", "logs/results.xml"])
    paths = job.fetch(1234, str(tmp_path / "d"))

    assert len(set(paths)) == 2, (
        f"two distinct remote logs collapsed onto one local path: {paths}"
    )
    contents = sorted(open(p, encoding="utf-8").read() for p in paths)
    assert contents == ["TestLogs/results.xml", "logs/results.xml"], (
        "one download overwrote the other -- the returned paths pointed at a "
        "single file holding whichever finished last"
    )


def test_fetch_keeps_the_results_basename_findable_by_the_cli(monkeypatch, tmp_path):
    """cli._qdc_check_results locates the report with
    `os.path.basename(p) == "results.xml"`, so mirroring the directory layout
    must not change what that sees."""
    import os as _os

    _fake_downloads(monkeypatch, ["TestLogs/results.xml", "TestLogs/logcat.txt"])
    paths = job.fetch(1234, str(tmp_path / "d"))
    assert any(_os.path.basename(p) == "results.xml" for p in paths)
    assert all(_os.path.isfile(p) for p in paths)


@pytest.mark.parametrize("evil", [
    "../../escaped.txt",
    "TestLogs/../../escaped.txt",
    "/etc/passwd",
    "C:/Windows/System32/evil.txt",
    "",
    "   ",
])
def test_fetch_refuses_a_remote_name_that_would_escape_the_destination(
    monkeypatch, tmp_path, evil
):
    """QDC's own names have never looked like this, which is exactly why
    nothing would notice if one did."""
    dest = tmp_path / "d"
    _fake_downloads(monkeypatch, [evil])
    paths = job.fetch(1234, str(dest))
    assert paths == []
    assert not (tmp_path / "escaped.txt").exists()


def _inject_fake_sdk(monkeypatch, *, get_public_api_client_using_api_key, client_ctor=None):
    """Inject a fake qualcomm_device_cloud_sdk package into sys.modules so
    job._client()'s internal lazy imports resolve to fakes -- proving
    _client()'s default-vs-override branching without the real SDK
    installed or reachable, and without network access."""
    import sys
    import types as _types

    vendor = _types.SimpleNamespace(
        get_public_api_client_using_api_key=get_public_api_client_using_api_key,
    )
    api_pkg = _types.SimpleNamespace(qdc_api=vendor)
    sdk_pkg = _types.ModuleType("qualcomm_device_cloud_sdk")
    sdk_pkg.api = api_pkg
    if client_ctor is not None:
        sdk_pkg.Client = client_ctor

    monkeypatch.setitem(sys.modules, "qualcomm_device_cloud_sdk", sdk_pkg)
    monkeypatch.setitem(sys.modules, "qualcomm_device_cloud_sdk.api", api_pkg)
    monkeypatch.setitem(sys.modules, "qualcomm_device_cloud_sdk.api.qdc_api", vendor)


def test_client_uses_the_sdks_default_endpoint_when_not_overridden(monkeypatch):
    monkeypatch.delenv("QDC_BASE_URL", raising=False)
    monkeypatch.setenv("QDC_API_KEY", "irrelevant-for-this-test")
    calls = {}

    def fake_default(**kwargs):
        calls["kwargs"] = kwargs
        return "vendor-client"

    _inject_fake_sdk(monkeypatch, get_public_api_client_using_api_key=fake_default)
    assert job._client() == "vendor-client"
    assert calls["kwargs"]["api_key_header"] == "irrelevant-for-this-test"


def test_client_honors_a_base_url_override_without_the_sdk_default(monkeypatch):
    monkeypatch.setenv("QDC_BASE_URL", "https://private-tenant.example/qdc")
    monkeypatch.setenv("QDC_API_KEY", "irrelevant-for-this-test")

    def boom(**kwargs):
        raise AssertionError(
            "the SDK's default-endpoint client must not be built when "
            "QDC_BASE_URL is set"
        )

    seen = {}

    class FakeClient:
        def __init__(self, base_url=None, headers=None):
            seen["base_url"] = base_url
            seen["headers"] = headers

    _inject_fake_sdk(monkeypatch, get_public_api_client_using_api_key=boom,
                     client_ctor=FakeClient)
    job._client()
    assert seen["base_url"] == "https://private-tenant.example/qdc"
    assert seen["headers"]["Authorization"] == "irrelevant-for-this-test"


def test_the_api_key_is_read_from_the_environment_not_committed(monkeypatch):
    monkeypatch.delenv("QDC_API_KEY", raising=False)
    monkeypatch.setattr(job, "_key_file", lambda: None)
    with pytest.raises(job.QdcError, match="QDC_API_KEY"):
        job._api_key()


# --- the credential scan -------------------------------------------------
#
# WHAT THIS REPLACED, AND WHY IT HAD TO GO. This was:
#
#     assert "qdc_api_key" not in src.lower() or "environ" in src or "home()" in src
#     assert "Bearer " not in src
#
# job.py contains `os.environ`, so the first disjunction was UNCONDITIONALLY
# TRUE for the only file that could ever carry a credential: adding
# `_FALLBACK_KEY = "<a realistic-looking key>"` to job.py left this file
# reporting 1 passed. Proven by mutation, not inferred. This is the only test
# guarding "no credential lands in the repository", and it could not fail.
#
# THE `Bearer ` ASSERTION IS GONE ON PURPOSE, NOT OVERLOOKED. It was a vacuous
# negative -- nothing under hexlib/device has ever spelled it, so it could
# only ever pass -- and the property it gestured at is already checked
# behaviourally, with a real binding, by
# test_client_honors_a_base_url_override_without_the_sdk_default above: that
# test asserts `seen["headers"]["Authorization"] == "irrelevant-for-this-test"`,
# i.e. the raw key with NO prefix, which is QDC's actual header scheme (a bare
# Authorization value plus X-QCOM-TokenType: apikey -- see job.py's docstring).
# An `assert "Bearer " not in src` cannot distinguish "we correctly don't use
# OAuth-style prefixes" from "this file happens not to contain that word", and
# keeping a second, weaker, source-text version of an assertion that is already
# made behaviourally is how a suite accumulates tests that only look like
# coverage.
#
# CREDENTIALS IN THIS PROJECT ARE PERSONAL AND LOCAL: read from QDC_API_KEY or
# ~/.qdc_api_key, never committed, never in CI. Nothing below reads either one.

DEVICE_DIR = pathlib.Path("hexlib/device")

# Identifier fragments that mean "this name holds a credential". Matched against
# the whole lowercased name AND against its underscore-separated words, so
# `_FALLBACK_KEY`, `apiKey`, `SECRET_TOKEN` and `qdc_api_key` all hit.
_CRED_WORDS = frozenset({
    "key", "keys", "secret", "secrets", "token", "tokens", "password",
    "passwd", "pwd", "credential", "credentials", "cred", "auth", "bearer",
    "signature", "sig",
})
_CRED_FRAGMENTS = (
    "apikey", "api_key", "access_key", "accesskey", "private_key",
    "privatekey", "secret", "password", "passwd", "credential", "authtoken",
    "auth_token", "token",
)

# Prefixes real credentials from real providers actually carry. Checked against
# every string in every file under hexlib/device REGARDLESS of what it is
# assigned to, since a key pasted into an innocuously-named variable (or a
# non-Python file) is the same leak. Kept to unmistakable markers so this
# cannot fire on a legitimate constant.
_SECRET_PREFIXES = (
    "sk-", "sk_live_", "sk_test_", "rk_live_", "ghp_", "gho_", "ghs_",
    "github_pat_", "xoxb-", "xoxp-", "xoxa-", "AKIA", "ASIA", "AIza",
    "ya29.", "eyJhbGciO",          # a JWT's own base64 header
    "-----BEGIN",                   # any PEM private key block
)

_ENV_VAR_NAME = re.compile(r"\A[A-Z][A-Z0-9_]*\Z")


def _shannon_entropy_bits_per_char(s):
    counts = collections.Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _why_this_looks_like_a_secret(value):
    """Return a reason string if `value` has the SHAPE of a credential, else
    None. Every exclusion below exists to keep a legitimate constant in this
    project from tripping it -- the point is a scan that can fail on a real
    key, not one that fails on a URL and gets deleted six weeks later.

    Deliberately shape-based and value-blind: nothing here has, or needs, any
    knowledge of what a real QDC key looks like."""
    if len(value) < 16:
        return None                      # QDC_API_KEY, apikey, X-QCOM-* etc.
    if "://" in value or value.startswith(("http", "www.", "/", ".", "-I", "--")):
        return None                      # URLs, paths, flags
    if " " in value or "\n" in value or "\t" in value:
        return None                      # prose: error messages, shell lines
    if "/" in value or "\\" in value:
        return None                      # /data/local/tmp/... and friends
    if value.isdigit():
        return None                      # TARGET_ID 3625030, timeouts, sizes
    if _ENV_VAR_NAME.match(value):
        return None                      # the NAME of an env var, e.g. QDC_API_KEY
    if re.fullmatch(r"[A-Za-z0-9_.]*\.[A-Za-z0-9]{1,6}", value):
        return None                      # filenames: results.xml, pytest.ini

    classes = sum((
        bool(re.search(r"[a-z]", value)),
        bool(re.search(r"[A-Z]", value)),
        bool(re.search(r"[0-9]", value)),
    ))
    if classes < 2:
        return None                      # all-lowercase words, SCREAMING_CASE

    bits = _shannon_entropy_bits_per_char(value)
    if bits < 3.0:
        return None
    return f"{len(value)} chars, {bits:.2f} bits/char, {classes} character classes"


def _credential_shaped(name):
    low = name.lower().lstrip("_")
    if any(f in low for f in _CRED_FRAGMENTS):
        return True
    return bool(_CRED_WORDS & set(w for w in low.split("_") if w))


def _named_string_constants(tree):
    """Yield (name, value, lineno) for every string literal in `tree` that is
    bound to a NAME: an assignment target (`X = "..."`, `self.x = "..."`,
    annotated or not), a keyword argument (`f(api_key="...")`), or a dict entry
    with a literal string key (`{"Authorization": "..."}`). Those are the
    places a credential actually gets written; using `ast` rather than text
    matching means a comment or a docstring cannot trip it, and equally cannot
    hide one."""
    def target_names(node):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, (ast.Tuple, ast.List)):
            for e in node.elts:
                yield from target_names(e)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                for t in targets:
                    for name in target_names(t):
                        yield name, node.value.value, node.value.lineno
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                if (kw.arg and isinstance(kw.value, ast.Constant)
                        and isinstance(kw.value.value, str)):
                    yield kw.arg, kw.value.value, kw.value.lineno
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if (isinstance(k, ast.Constant) and isinstance(k.value, str)
                        and isinstance(v, ast.Constant) and isinstance(v.value, str)):
                    yield k.value, v.value, v.lineno


def _scan_python_source(path, src):
    """Credential-shaped NAME bound to a secret-shaped VALUE."""
    findings = []
    for name, value, lineno in _named_string_constants(ast.parse(src)):
        if not _credential_shaped(name):
            continue
        why = _why_this_looks_like_a_secret(value)
        if why:
            findings.append(f"{path}:{lineno}: {name} = a {why} string literal")
    return findings


def _scan_raw_text(path, src):
    """A provider's own credential prefix, anywhere, under any name -- covers
    the case the AST scan cannot: a key assigned to a name nobody would flag,
    or sitting in a non-Python file."""
    findings = []
    for lineno, line in enumerate(src.split("\n"), start=1):
        for prefix in _SECRET_PREFIXES:
            idx = line.find(prefix)
            if idx != -1 and _why_this_looks_like_a_secret(line[idx:].strip().strip("'\"")):
                findings.append(f"{path}:{lineno}: a literal beginning {prefix!r}")
    return findings


def _scan_device_tree():
    """Every text file under hexlib/device, not just *.py: a credential in a
    shell script, an .ini or a .json staged into the artifact is the same leak.
    Skips __pycache__ and anything that is not decodable as UTF-8."""
    findings = []
    for p in sorted(DEVICE_DIR.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts:
            continue
        try:
            src = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        posix = p.as_posix()
        if p.suffix == ".py":
            findings += _scan_python_source(posix, src)
        findings += _scan_raw_text(posix, src)
    return findings


def test_no_credential_appears_anywhere_in_the_device_source():
    """THE ONLY test guarding "no credential lands in the repository". It has
    to be able to fail; the version this replaced could not (see the comment
    block above). Its companion below proves it can, by planting one."""
    findings = _scan_device_tree()
    assert not findings, (
        "credential-shaped literal(s) found under hexlib/device -- QDC keys are "
        "personal and are read only from QDC_API_KEY or ~/.qdc_api_key, never "
        "committed:\n  " + "\n  ".join(findings)
    )


def test_the_credential_scan_can_actually_fail(tmp_path, monkeypatch):
    """MUTATION-VERIFY, kept as a test rather than done once by hand -- same
    pattern as test_coherency_lane_classification.py's own mutation check.
    Plants the exact shape of the leak that defeated the previous assertion
    (`_FALLBACK_KEY = "<key>"` in a file that also uses `os.environ`, which was
    what made the old disjunction unconditionally true) and confirms the scan
    reports it.

    THE PLANTED VALUE IS SYNTHETIC AND OBVIOUSLY SO: a 32-char hex string whose
    nibbles simply count down and then up (0f1e2d3c...). It is not a QDC key,
    not any provider's key, and no real credential is read, constructed or
    stored anywhere in this file. It exists only to have the right SHAPE --
    length and entropy -- for the detector to bite on."""
    synthetic = "0f1e2d3c4b5a6978" + "8796a5b4c3d2e1f0"
    planted = tmp_path / "qdc" / "leak.py"
    planted.parent.mkdir(parents=True)
    planted.write_text(
        "import os\n"
        "def _api_key():\n"
        "    return os.environ.get('QDC_API_KEY') or _FALLBACK_KEY\n"
        f'_FALLBACK_KEY = "{synthetic}"\n'
    )
    monkeypatch.setattr("hexlib.tests.test_qdc.DEVICE_DIR", tmp_path)

    findings = _scan_device_tree()
    assert findings, (
        "the credential scan did not notice a planted, credential-shaped "
        "literal -- so a green result from it means nothing. This is exactly "
        "the state the assertion it replaced was in."
    )
    assert any("_FALLBACK_KEY" in f for f in findings)

    # And the clean tree really is clean for the right reason: remove the
    # planted file and the same scan over the same directory goes quiet, so the
    # assertion above is detecting THAT literal and not merely anything at all.
    planted.unlink()
    assert not _scan_device_tree()


def test_the_credential_scan_does_not_fire_on_this_projects_real_constants():
    """The other half of "can fail": a scan that flags legitimate constants
    gets deleted. Pins the specific shapes hexlib/device really contains --
    the NAME of an env var, QDC's header names, the numeric target id, a device
    path -- so a future tightening of the heuristic that breaks them fails here
    instead of in someone's unrelated PR."""
    for name, value in (
        ("_API_KEY_ENV", "QDC_API_KEY"),        # an env var's NAME, not a key
        ("_KEY_FILE_NAME", ".qdc_api_key"),     # a filename, not a key
        ("X-QCOM-TokenType", "apikey"),         # the token TYPE, not a token
        ("TARGET_ID_KEY", "3625030"),           # a measured, public target id
        ("KEY_PATH", "/data/local/tmp/hexlib"),
        ("api_key_header", "QDC_BASE_URL"),
    ):
        assert _credential_shaped(name), f"{name} should be treated as sensitive"
        assert _why_this_looks_like_a_secret(value) is None, (
            f"{name} = {value!r} is a legitimate constant in this project and "
            "must not be reported as a credential"
        )
