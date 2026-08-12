#!/usr/bin/env python3
"""Run hexlib's device binaries on real Snapdragon silicon via a QDC session.

FIRST WORKING PATH TO SILICON, 2026-08-13. Four Appium test-package jobs
(756124, 756159, 756206, 756221) all reached Completed with
result=UNSUCCESSFUL and published no logs of their own, so nothing inside the
package could be observed. The interactive session works instead, and this
file is the recipe.

THREE THINGS ARE ALL REQUIRED. Each was a separate discovery and none is
obvious from the SDK signature:

  1. THE KEY IS QDC'S, NOT YOURS. `submit_session(ssh_public_key=...)` refuses
     a key of your own with "The provided SSH public key could not be found or
     has not been created for user". QDC ISSUES the pair -- look for
     `~/.ssh/qdc_id_<date>.pem` dropped when a session was created from the web
     UI. Probing ours showed the 2026-08-07 key registered and the 2026-08-06
     one not, so they expire or get replaced; if this script starts failing at
     submit, make a session in the UI and use the new pem.

  2. session_parameters=[SSHONLY] IS WHAT PROVISIONS SSH. Without it the
     session is created, reaches Running, and NEVER publishes an sshConfigs
     entry. Two ten-minute polls were burned on that (sessions 756450,
     756584) before the parameter was found.

  3. IT IS AN ADB TUNNEL, NOT A SHELL. sshConfigs hands back
         ssh -i <PRIVATE_KEY_FILE_PATH> \\
             -L <ADB_PORT>:<host>:5037 -N sshtunnel@ssh.qdc.qualcomm.com
     which forwards a local port to the DEVICE'S ADB SERVER. There is no
     remote host to scp to and no remote shell. Everything runs from THIS
     machine through `adb -P <port>`.

BILLING. Sessions bill by the minute and are billed for the full timeout, not
for what you use (session 756450: 15 charged, ~10 used). complete_session runs
in a finally, and `--timeout` bounds the worst case even if this process dies.

WHAT IT FOUND ON THE FIRST REAL RUN, so nobody re-derives it:
  * `--caps` returns arch_ver 35957 (0x8c75) -- BIT-IDENTICAL to the
    simulator, which is what job.py's fact 1 asserted and had never checked.
    unsigned_pd_support=1, vtcm_total_bytes=8388608.
  * The skel is built as `libhexlib_skel.so` but FastRPC dlopens
    `libhexlib_iface_skel.so`. No simulator test can catch this: stage 1 links
    the skel directly instead of loading it by name. Push it under both names
    until runtime/build.py is fixed.
  * cycles_total=14267 in a user-mode unsigned PD -- NON-ZERO. STATE.md called
    this the most important thing a device job can report, because a dead
    PCYCLE would invalidate every cycle figure stage 1 measured.
  * The unmapped-fd refusal holds against real ION (batch status 7).
  * `--self-test` returns 3859/4100 values not bit-exact and
    `--coherency-check` exits 6 `sentinel_unchanged`. That is NOT a coherency
    diagnosis: main.c:94 says exit 6 is equally consistent with a kernel or
    generated entry returning OK without writing its output. Settling it needs
    the skel-side echo op STATE.md records as deferred.

usage:
  python scripts/qdc_interactive.py --key ~/.ssh/qdc_id_2026-8-7_847.pem \\
      --bin-dir <dir with hexlib_run and libhexlib_skel.so> [--timeout-min 15]
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

SSH_EXTRA = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "IdentitiesOnly=yes",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=15",
    "-o", "LogLevel=ERROR",
]
DEV = "/data/local/tmp/hexlib"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def adb(port: int, *args: str, timeout: int = 180):
    cmd = ["adb", "-P", str(port), *args]
    log("$ " + " ".join(cmd[:7]) + (" ..." if len(cmd) > 7 else ""))
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    out = ((p.stdout or "") + (p.stderr or "")).rstrip()
    if out:
        print(out[:4000], flush=True)
    log(f"  -> exit {p.returncode}")
    return p.returncode, out


def port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(2)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True,
                    help="QDC-ISSUED pem (~/.ssh/qdc_id_<date>.pem), not your own key")
    ap.add_argument("--bin-dir", required=True,
                    help="directory holding hexlib_run and libhexlib_skel.so")
    ap.add_argument("--timeout-min", type=int, default=15,
                    help="session ceiling; you are billed for ALL of it")
    ap.add_argument("--adb-port", type=int, default=15037)
    ap.add_argument("--ready-wait-s", type=int, default=420)
    args = ap.parse_args()

    from qualcomm_device_cloud_sdk.api import qdc_api as v
    from qualcomm_device_cloud_sdk.models.session_submission_parameter import (
        SessionSubmissionParameter,
    )
    from hexlib.device.qdc import job

    key = os.path.expanduser(args.key)
    pub = subprocess.run(["ssh-keygen", "-y", "-f", key],
                         capture_output=True, text=True).stdout.strip()
    if not pub:
        log(f"cannot derive a public key from {key}")
        return 1

    client = job._client()
    sid = v.submit_session(
        public_api_client=client,
        target_id=job.TARGET_ID,
        session_name="hexlib interactive",   # <= 32 chars or QDC answers 400
        timeout=args.timeout_min,
        ssh_public_key=pub,
        session_parameters=[SessionSubmissionParameter.SSHONLY],
    )
    if sid is None:
        log("submit_session returned None")
        return 1
    log(f"session {sid} submitted (timeout {args.timeout_min} min)")

    tunnel = None
    try:
        cmd = None
        deadline = time.time() + args.ready_wait_s
        while time.time() < deadline:
            d = json.loads(v.get_session_by_id(client, sid).content.decode())
            cfgs = d.get("sshConfigs") or []
            log(f"state={d.get('state')} sshConfigs={len(cfgs)}")
            if cfgs:
                cmd = cfgs[0].get("sshCommand") or cfgs[0].get("qualnetSshCommand")
                break
            if d.get("state") in ("Completed", "Canceled", "Failed"):
                log(f"session ended early: {d.get('state')}")
                return 1
            time.sleep(15)
        if not cmd:
            log("no sshConfigs before the cap -- is SSHONLY set and the key QDC's?")
            return 1

        real = (cmd.replace("<PRIVATE_KEY_FILE_PATH>", key)
                   .replace("<ADB_PORT>", str(args.adb_port)))
        parts = real.split()
        tunnel = subprocess.Popen([parts[0], *SSH_EXTRA, *parts[1:]],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True)
        for _ in range(30):
            if port_open(args.adb_port):
                break
            if tunnel.poll() is not None:
                log(f"tunnel died: {(tunnel.stdout.read() or '')[:800]}")
                return 1
            time.sleep(2)
        else:
            log(f"port {args.adb_port} never opened")
            return 1
        log(f"tunnel up: local {args.adb_port} -> device adb server")

        p = args.adb_port
        rc, out = adb(p, "devices")
        if rc != 0 or "\tdevice" not in out:
            log("no device through the tunnel")
            return 1
        adb(p, "shell", "getprop ro.product.model")
        adb(p, "shell", f"mkdir -p {DEV}")
        adb(p, "push", os.path.join(args.bin_dir, "hexlib_run"), f"{DEV}/")
        adb(p, "push", os.path.join(args.bin_dir, "libhexlib_skel.so"), f"{DEV}/")
        # BOTH NAMES until runtime/build.py is fixed -- FastRPC dlopens
        # libhexlib_iface_skel.so and the build emits libhexlib_skel.so.
        adb(p, "push", os.path.join(args.bin_dir, "libhexlib_skel.so"),
            f"{DEV}/libhexlib_iface_skel.so")
        adb(p, "shell", f"chmod 755 {DEV}/hexlib_run")

        env = f"cd {DEV} && ADSP_LIBRARY_PATH={DEV}"
        for mode in ("--caps",
                     "--self-test",
                     "--self-test --coherency-check",
                     "--self-test --unmapped"):
            log(f"=== hexlib_run {mode} ===")
            adb(p, "shell", f"{env} ./hexlib_run {mode}; echo RC=$?", timeout=300)
        return 0
    finally:
        if tunnel and tunnel.poll() is None:
            log("closing tunnel")
            tunnel.terminate()
        log(f"completing session {sid}")
        try:
            v.complete_session(client, sid)
            log("session completed")
        except Exception as e:  # noqa: BLE001
            log(f"complete_session FAILED: {e} -- session {sid} bills until its "
                f"{args.timeout_min}-minute timeout")


if __name__ == "__main__":
    sys.exit(main())
