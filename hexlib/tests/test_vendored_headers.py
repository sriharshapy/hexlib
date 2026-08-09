"""The vendored headers must stay ggml-free and must stay attributed.

A header that acquires a ggml include has been edited in place or re-copied from
the wrong file, and hexlib's independence from llama.cpp is gone silently."""
import pathlib
import re

HVX_DIR = pathlib.Path(__file__).resolve().parents[2] / "include" / "hexlib" / "hvx"


def test_headers_are_present():
    names = {p.name for p in HVX_DIR.glob("*.h")}
    for required in ("hvx-types.h", "hvx-norm.h", "hvx-reduce.h", "hvx-sqrt.h"):
        assert required in names, f"{required} was not vendored"


def test_no_header_references_ggml():
    headers = list(HVX_DIR.glob("*.h"))
    assert headers, f"no headers found in {HVX_DIR} — the vendored set is missing"
    offenders = [
        p.name for p in headers
        if "ggml" in p.read_text(encoding="utf-8", errors="replace").lower()
    ]
    assert offenders == [], f"vendored headers must stay ggml-free: {offenders}"


def test_provenance_records_an_upstream_commit():
    readme = (HVX_DIR / "README.md").read_text(encoding="utf-8")
    assert "MIT" in readme
    assert re.search(r"`[0-9a-f]{7,40}`", readme), (
        "README.md must record the upstream commit these headers were copied from"
    )
