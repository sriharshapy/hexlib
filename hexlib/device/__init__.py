# hexlib/device/__init__.py
"""Stage 3: getting hexlib_run and libhexlib_iface_skel.so onto a real phone.

Everything under here that talks to Qualcomm Device Cloud is exercised
offline, against a fake client, in hexlib/tests/test_qdc.py. See
hexlib/device/qdc/job.py for why completion is detected from log files
rather than by polling job status.
"""
