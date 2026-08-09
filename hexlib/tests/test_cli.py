import os

from hexlib import cli


def test_new_kernel_creates_a_valid_directory(tmp_path, capsys):
    rc = cli.main(["new-kernel", "softmax_fp16", "--kernels-root", str(tmp_path)])
    assert rc == 0
    assert os.path.isfile(tmp_path / "softmax_fp16" / "spec.json")
    assert "softmax_fp16" in capsys.readouterr().out


def test_new_kernel_refuses_to_overwrite(tmp_path, capsys):
    cli.main(["new-kernel", "softmax_fp16", "--kernels-root", str(tmp_path)])
    rc = cli.main(["new-kernel", "softmax_fp16", "--kernels-root", str(tmp_path)])
    assert rc != 0
    assert "already exists" in capsys.readouterr().err


def test_validate_reports_problems_and_exits_nonzero(tmp_path, capsys):
    d = tmp_path / "broken"
    d.mkdir()
    rc = cli.main(["validate", str(d)])
    assert rc != 0
    assert "missing required file" in capsys.readouterr().err


def test_validate_passes_a_scaffolded_kernel(tmp_path):
    cli.main(["new-kernel", "k", "--kernels-root", str(tmp_path)])
    assert cli.main(["validate", str(tmp_path / "k")]) == 0


def test_unknown_device_is_rejected(tmp_path, capsys):
    cli.main(["new-kernel", "k", "--kernels-root", str(tmp_path)])
    rc = cli.main(["test", str(tmp_path / "k"), "--device", "gpu"])
    assert rc != 0


def test_device_backends_not_in_this_plan_say_so(tmp_path, capsys):
    """local and qdc arrive in the silicon-path plan; the CLI must say that
    rather than fail obscurely."""
    cli.main(["new-kernel", "k", "--kernels-root", str(tmp_path)])
    rc = cli.main(["test", str(tmp_path / "k"), "--device", "local"])
    assert rc != 0
    assert "not implemented" in capsys.readouterr().err.lower()
